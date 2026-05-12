import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def recurrence_step_full(x, x_proj, h, R, gate, scale):
    """h is normalized to unit sphere, scaled by learnable scale."""
    alpha = torch.sigmoid(gate(torch.cat([x, h], dim=-1)))
    pre = (R @ h.unsqueeze(-1)).squeeze(-1) * (1.0 - alpha) + x_proj * alpha
    return F.normalize(pre, dim=-1) * scale

def compute_jacobian_analytic(x, x_proj, h_prev, R, gate, scale):
    """
    h_new = normalize(u) * scale,  u = R@h*(1-a) + x_proj*a
    dh_new/dh = scale * d(normalize(u))/du * du/dh

    d(normalize(u))/du = (I - hat_u hat_u^T) / ||u||
    du/dh = R*diag(1-a) + (x_proj - R@h) * da/dh^T
    """
    H = h_prev.shape[-1]
    W_gate = gate.weight
    W_gate_h = W_gate[:, x.shape[-1]:]

    alpha = torch.sigmoid(gate(torch.cat([x, h_prev], dim=-1)))
    da = alpha * (1.0 - alpha)

    Rh = (R @ h_prev.unsqueeze(-1)).squeeze(-1)
    u = Rh * (1.0 - alpha) + x_proj * alpha
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u_hat = u / u_norm                                         # (B, H)

    # du/dh
    residual = x_proj - Rh
    J_u1 = R * (1.0 - alpha).unsqueeze(-2)                    # (B, H, H)
    coeff = residual * da
    J_u2 = coeff.unsqueeze(-1) * W_gate_h.unsqueeze(0)        # (B, H, H)
    J_u = J_u1 + J_u2                                         # (B, H, H)

    # d(normalize)/du = (I - u_hat u_hat^T) / ||u||
    B = h_prev.shape[0]
    I = torch.eye(H, device=h_prev.device).unsqueeze(0)
    P = I - u_hat.unsqueeze(-1) * u_hat.unsqueeze(-2)         # (B, H, H)
    J_norm = P / u_norm.unsqueeze(-1)                          # (B, H, H)

    return scale * torch.bmm(J_norm, J_u)                     # (B, H, H)

def compute_jacobian_autograd(x, x_proj, h_prev, R, gate, scale):
    B, H = h_prev.shape
    h_req = h_prev.detach().requires_grad_(True)
    h_new = recurrence_step_full(x, x_proj, h_req, R, gate, scale)
    jac = torch.zeros(B, H, H, device=h_prev.device)
    for i in range(H):
        g = torch.autograd.grad(h_new[:, i].sum(), h_req, retain_graph=True)[0]
        jac[:, i, :] = g
    return jac

def newton_residual(sol, x_seq, x_proj_seq, R_seq, gate, scale):
    h_prev = torch.roll(sol, shifts=1, dims=1)
    h_prev[:, 0] = 0.0
    B, T, H = sol.shape
    h_pred = torch.zeros_like(sol)
    for t in range(T):
        h_pred[:, t] = recurrence_step_full(
            x_seq[:, t], x_proj_seq[:, t], h_prev[:, t], R_seq[:, t], gate, scale
        )
    return sol - h_pred

def parallel_reduce_dense(jacobians, rhs):
    J = jacobians.clone()
    r = rhs.clone()
    num_steps = math.ceil(math.log2(rhs.shape[1]))
    for step in range(num_steps):
        idx = 1 << step
        T = r.shape[1]
        r[:, idx:] -= torch.einsum('btij,btj->bti', J[:, idx:], r[:, :T - idx])
        J[:, idx:] = torch.einsum('btij,btjk->btik', -J[:, idx:], J[:, :T - idx])
        J[:, :idx] = 0.0
    return r

def test_newton_convergence(
    B=8, T=64, H=32,
    n_iters=6,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"Device: {device}, B={B}, T={T}, H={H}\n")
    torch.manual_seed(42)

    sys.path.insert(0, '.')
    from grnn.rotor import RotorLayer

    gate = spectral_norm(nn.Linear(H + H, H, bias=True)).to(device)
    rotor = RotorLayer(H, triton=False).to(device)
    scale = math.sqrt(H)

    x_seq = torch.randn(B, T, H, device=device)
    x_proj_seq = torch.randn(B, T, H, device=device)

    theta_flat = rotor.mlp(x_proj_seq.reshape(B * T, H))
    A = torch.zeros(B * T, H, H, device=device)
    A[:, rotor.tril_i, rotor.tril_j] = theta_flat
    A = A - A.transpose(-2, -1)
    R_seq = torch.linalg.matrix_exp(A).reshape(B, T, H, H)

    sol = torch.zeros(B, T, H, device=device)
    for t in range(T):
        sol[:, t] = recurrence_step_full(
            x_seq[:, t], x_proj_seq[:, t],
            torch.zeros(B, H, device=device), R_seq[:, t], gate, scale
        )

    print("Jacobian verification (analytic vs autograd) at t=0:")
    jac_auto = compute_jacobian_autograd(
        x_seq[:, 0], x_proj_seq[:, 0], sol[:, 0].detach(), R_seq[:, 0], gate, scale
    )
    jac_anal = compute_jacobian_analytic(
        x_seq[:, 0], x_proj_seq[:, 0], sol[:, 0].detach(), R_seq[:, 0], gate, scale
    )
    diff = (jac_auto - jac_anal).abs()
    print(f"  max diff:  {diff.max().item():.2e}")
    print(f"  mean diff: {diff.mean().item():.2e}\n")

    print(f"{'Iter':>6} | {'max_norm':>12} | {'mean_norm':>12} | {'spec_rad':>10}")
    print("-" * 52)

    for it in range(n_iters):
        res = newton_residual(sol, x_seq, x_proj_seq, R_seq, gate, scale)
        max_norm = res.abs().max().item()
        mean_norm = res.norm(dim=-1).mean().item()

        h_prev = torch.roll(sol, shifts=1, dims=1)
        h_prev[:, 0] = 0.0
        jac = torch.zeros(B, T, H, H, device=device)
        spec_rads = []
        for t in range(T):
            J_t = compute_jacobian_analytic(
                x_seq[:, t], x_proj_seq[:, t], h_prev[:, t], R_seq[:, t], gate, scale
            )
            jac[:, t] = J_t
            sv = torch.linalg.svdvals(J_t)
            spec_rads.append(sv.max().item())

        sr = sum(spec_rads) / len(spec_rads)
        print(f"{it:>6} | {max_norm:>12.6f} | {mean_norm:>12.6f} | {sr:>10.4f}")

        if max_norm < 1e-5:
            print(f"\nConverged at iteration {it}")
            break

        delta = parallel_reduce_dense(-jac, res)
        sol = sol + delta

    print("\nDone.")

if __name__ == "__main__":
    test_newton_convergence()
