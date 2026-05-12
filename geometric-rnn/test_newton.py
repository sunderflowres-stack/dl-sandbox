import sys
import math
import torch
import torch.nn as nn

def recurrence_step_full(x, x_proj, h, R, gate):
    alpha = torch.sigmoid(gate(torch.cat([x, h], dim=-1)))
    return (R @ h.unsqueeze(-1)).squeeze(-1) * (1.0 - alpha) + x_proj * alpha

def compute_jacobian_analytic(x, x_proj, h_prev, R, gate):
    """
    df/dh_{t-1}:
    h_t = R @ h * (1-a) + x_proj * a,   a = sigmoid(W_x x + W_h h + b)

    dh_t/dh = R * diag(1-a)  +  (x_proj - R@h) * da/dh^T
    da/dh[i,j] = a_i*(1-a_i) * W_h[i,j]
    """
    H = h_prev.shape[-1]
    W_gate = gate.weight                        # (H_out, H_in_x + H)
    W_gate_h = W_gate[:, x.shape[-1]:]         # (H, H) — part for h

    gate_input = torch.cat([x, h_prev], dim=-1)
    alpha = torch.sigmoid(gate(gate_input))     # (B, H)
    da = alpha * (1.0 - alpha)                  # (B, H) — dsigmoid

    Rh = (R @ h_prev.unsqueeze(-1)).squeeze(-1)  # (B, H)
    residual = x_proj - Rh                        # (B, H)

    # J1: (B, H, H) — R scaled by (1-alpha) column-wise
    J1 = R * (1.0 - alpha).unsqueeze(-2)          # (B, 1, H) broadcasts over rows

    # J2[b, i, j] = residual[b,i] * da[b,i] * W_gate_h[i,j]
    # = (residual * da)[b,i] * W_gate_h[i,j]
    coeff = residual * da                          # (B, H)
    J2 = coeff.unsqueeze(-1) * W_gate_h.unsqueeze(0)  # (B, H, H)

    return J1 + J2


def compute_jacobian_autograd(x, x_proj, h_prev, R, gate):
    B, H = h_prev.shape
    h_req = h_prev.detach().requires_grad_(True)
    h_new = recurrence_step_full(x, x_proj, h_req, R, gate)
    jac = torch.zeros(B, H, H, device=h_prev.device)
    for i in range(H):
        g = torch.autograd.grad(h_new[:, i].sum(), h_req, retain_graph=True)[0]
        jac[:, i, :] = g
    return jac


def newton_residual(sol, x_seq, x_proj_seq, R_seq, gate):
    h_prev = torch.roll(sol, shifts=1, dims=1)
    h_prev[:, 0, :] = 0.0
    B, T, H = sol.shape
    h_pred = torch.zeros_like(sol)
    for t in range(T):
        h_pred[:, t] = recurrence_step_full(
            x_seq[:, t], x_proj_seq[:, t], h_prev[:, t], R_seq[:, t], gate
        )
    return sol - h_pred


def parallel_reduce_dense(jacobians, rhs):
    """Pure PyTorch parallel reduction, dense Jacobians (B,T,H,H)."""
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
    jac_clip=0.95,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"Device: {device}, B={B}, T={T}, H={H}, jac_clip={jac_clip}\n")
    torch.manual_seed(42)

    sys.path.insert(0, '.')
    from grnn.rotor import RotorLayer

    gate = nn.Linear(H + H, H, bias=True).to(device)
    rotor = RotorLayer(H, triton=False).to(device)

    x_seq = torch.randn(B, T, H, device=device)
    x_proj_seq = torch.randn(B, T, H, device=device)

    # precompute R_seq
    theta_flat = rotor.mlp(x_proj_seq.reshape(B * T, H))
    A = torch.zeros(B * T, H, H, device=device)
    A[:, rotor.tril_i, rotor.tril_j] = theta_flat
    A = A - A.transpose(-2, -1)
    R_seq = torch.linalg.matrix_exp(A).reshape(B, T, H, H)

    # initial guess: f(0, x_t)
    sol = torch.zeros(B, T, H, device=device)
    for t in range(T):
        sol[:, t] = recurrence_step_full(
            x_seq[:, t], x_proj_seq[:, t],
            torch.zeros(B, H, device=device), R_seq[:, t], gate
        )

    print("Jacobian verification (analytic vs autograd) at t=0:")
    jac_auto = compute_jacobian_autograd(
        x_seq[:, 0], x_proj_seq[:, 0], sol[:, 0].detach(), R_seq[:, 0], gate
    )
    jac_anal = compute_jacobian_analytic(
        x_seq[:, 0], x_proj_seq[:, 0], sol[:, 0].detach(), R_seq[:, 0], gate
    )
    diff = (jac_auto - jac_anal).abs()
    print(f"  max diff:  {diff.max().item():.2e}")
    print(f"  mean diff: {diff.mean().item():.2e}\n")

    print(f"{'Iter':>6} | {'max_norm':>12} | {'mean_norm':>12} | {'spec_rad':>10}")
    print("-" * 52)

    for it in range(n_iters):
        res = newton_residual(sol, x_seq, x_proj_seq, R_seq, gate)
        max_norm = res.abs().max().item()
        mean_norm = res.norm(dim=-1).mean().item()

        # compute jacobians
        h_prev = torch.roll(sol, shifts=1, dims=1)
        h_prev[:, 0] = 0.0
        jac = torch.zeros(B, T, H, H, device=device)
        for t in range(T):
            J_t = compute_jacobian_analytic(
                x_seq[:, t], x_proj_seq[:, t], h_prev[:, t], R_seq[:, t], gate
            )
            # clip spectral radius to jac_clip for convergence
            sv = torch.linalg.svdvals(J_t)
            sr = sv[:, 0].mean().item()
            scale = min(1.0, jac_clip / (sv.max().item() + 1e-8))
            jac[:, t] = J_t * scale

        print(f"{it:>6} | {max_norm:>12.6f} | {mean_norm:>12.6f} | {sr:>10.4f}")

        if max_norm < 1e-5:
            print(f"\nConverged at iteration {it}")
            break

        delta = parallel_reduce_dense(-jac, res)
        sol = sol + delta

    print("\nDone.")


if __name__ == "__main__":
    test_newton_convergence()
