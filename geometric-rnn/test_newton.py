import sys
import math
import torch
import torch.nn.functional as F

# Trying to implement pararnn method (C) 2025 Apple Inc.


def recurrence_step(x_proj, h, R, gate_weight_h, gate_bias, use_gate=True):
    """
    h_new = R @ h * (1 - alpha) + x_proj * alpha
    R: (B, H, H) — precomputed rotation matrix
    gate_weight_h: (H, H) — W_gate[:, H:] (part for h)
    gate_weight_x: not needed here since x is fixed per step
    """
    # for convergence test we pass pre-gated alpha separately
    raise NotImplementedError


def recurrence_step_full(x, x_proj, h, R, gate, use_gate=True):
    """Full cell recurrence for Newton test."""
    if use_gate:
        gate_input = torch.cat([x, h], dim=-1)
        alpha = torch.sigmoid(gate(gate_input))
        h_new = (R @ h.unsqueeze(-1)).squeeze(-1) * (1.0 - alpha) + x_proj * alpha
    else:
        h_new = (R @ h.unsqueeze(-1)).squeeze(-1) + x_proj
    return h_new


def compute_jacobian_autograd(x, x_proj, h_prev, R, gate, use_gate=True):
    """
    Compute df/dh_{t-1} via autograd (reference implementation).
    Returns dense (B, H, H) Jacobian.
    """
    B, H = h_prev.shape
    h_prev_req = h_prev.detach().requires_grad_(True)

    h_new = recurrence_step_full(x, x_proj, h_prev_req, R, gate, use_gate)

    jac = torch.zeros(B, H, H, device=h_prev.device)
    for i in range(H):
        grad = torch.autograd.grad(
            outputs=h_new[:, i].sum(),
            inputs=h_prev_req,
            retain_graph=True,
        )[0]
        jac[:, i, :] = grad

    return jac


def compute_jacobian_analytic(x, x_proj, h_prev, R, gate, use_gate=True):
    """
    Analytic Jacobian df/dh_{t-1}.
    J = R @ diag(1 - alpha) + (x_proj - R @ h) * alpha*(1-alpha) @ W_gate_h
    """
    B, H = h_prev.shape

    if use_gate:
        gate_input = torch.cat([x, h_prev], dim=-1)
        alpha = torch.sigmoid(gate(gate_input))           # (B, H)
        da_dh = alpha * (1.0 - alpha)                     # (B, H) — dsigmoid

        # W_gate_h: (H, H) — columns corresponding to h part of [x, h]
        W_gate = gate.weight                               # (H, 2H)
        W_gate_h = W_gate[:, x.shape[-1]:]                # (H, H)

        Rh = (R @ h_prev.unsqueeze(-1)).squeeze(-1)       # (B, H)
        residual = x_proj - Rh                            # (B, H)

        # J = R @ diag(1-alpha) + outer(residual, da_dh) @ W_gate_h
        # component 1: (B, H, H)
        J1 = R * (1.0 - alpha).unsqueeze(-2)              # broadcast: (B, 1, H) * (B, H, H)

        # component 2: d(alpha)/dh = da_dh[:, None, :] * W_gate_h
        # contribution: residual[:, :, None] * (da_dh[:, None, :] * W_gate_h[None])
        # shape: (B, H, 1) * (B, 1, H) * (1, H, H) -> (B, H, H)
        J2 = (residual.unsqueeze(-1) * da_dh.unsqueeze(-2)) * W_gate_h.unsqueeze(0)

        return J1 + J2
    else:
        return R.clone()


def newton_residual(sol, x_seq, x_proj_seq, R_seq, gate, use_gate=True):
    """
    Residual for Newton: r_t = h_t - f(h_{t-1}, x_t)
    sol: (B, T, H)
    Returns residual (B, T, H)
    """
    B, T, H = sol.shape
    h_prev = torch.roll(sol, shifts=1, dims=1)
    h_prev[:, 0, :] = 0.0

    h_pred = torch.zeros_like(sol)
    for t in range(T):
        h_pred[:, t, :] = recurrence_step_full(
            x_seq[:, t, :], x_proj_seq[:, t, :],
            h_prev[:, t, :], R_seq[:, t, :, :],
            gate, use_gate
        )

    return sol - h_pred


def parallel_reduce_dense(jacobians, rhs):
    """
    Pure PyTorch parallel reduction for dense Jacobians.
    jacobians: (B, T, H, H)
    rhs:       (B, T, H)
    Returns delta_h: (B, T, H)
    """
    J = jacobians.clone()
    r = rhs.clone()
    num_steps = math.ceil(math.log2(rhs.shape[1]))

    for step in range(num_steps):
        idx = 1 << step
        r[:, idx:, :] -= torch.einsum('btij,btj->bti', J[:, idx:], r[:, :r.shape[1]-idx])
        J[:, idx:] = torch.einsum('btij,btjk->btik', -J[:, idx:], J[:, :J.shape[1]-idx])
        J[:, :idx] = 0.0

    return r


def test_newton_convergence(
    B=8, T=64, H=32,
    n_iters=5,
    use_gate=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"Device: {device}, B={B}, T={T}, H={H}")
    print()

    torch.manual_seed(42)

    import torch.nn as nn
    gate = nn.Linear(H + H, H, bias=True).to(device)

    from grnn.rotor import RotorLayer
    rotor = RotorLayer(H, triton=False).to(device)

    x_seq = torch.randn(B, T, H, device=device)
    x_proj_seq = torch.randn(B, T, H, device=device)

    # precompute R for all timesteps
    theta_flat = rotor.mlp(x_proj_seq.reshape(B * T, H))
    A = torch.zeros(B * T, H, H, device=device)
    A[:, rotor.tril_i, rotor.tril_j] = theta_flat
    A = A - A.transpose(-2, -1)
    R_seq = torch.linalg.matrix_exp(A).reshape(B, T, H, H)

    # initial guess: h0 = f(0, x_t) for all t
    h0 = torch.zeros(B, T, H, device=device)
    for t in range(T):
        h0[:, t, :] = recurrence_step_full(
            x_seq[:, t, :], x_proj_seq[:, t, :],
            torch.zeros(B, H, device=device),
            R_seq[:, t, :, :], gate, use_gate
        )

    sol = h0.detach().clone()

    print("Newton convergence (residual norm per iteration):")
    print(f"{'Iter':>6} | {'max_norm':>12} | {'mean_norm':>12}")
    print("-" * 38)

    for it in range(n_iters):
        res = newton_residual(sol, x_seq, x_proj_seq, R_seq, gate, use_gate)
        max_norm = res.abs().max().item()
        mean_norm = res.norm(dim=-1).mean().item()
        print(f"{it:>6} | {max_norm:>12.6f} | {mean_norm:>12.6f}")

        if max_norm < 1e-6:
            print(f"\nConverged at iteration {it}")
            break

        # compute Jacobians for all timesteps
        jac = torch.zeros(B, T, H, H, device=device)
        h_prev = torch.roll(sol, shifts=1, dims=1)
        h_prev[:, 0, :] = 0.0
        for t in range(T):
            jac[:, t, :, :] = compute_jacobian_analytic(
                x_seq[:, t, :], x_proj_seq[:, t, :],
                h_prev[:, t, :], R_seq[:, t, :, :],
                gate, use_gate
            )

        delta = parallel_reduce_dense(-jac, res)
        sol = sol + delta

    print()

    # verify analytic vs autograd Jacobian
    print("Jacobian verification (analytic vs autograd):")
    t = 0
    jac_auto = compute_jacobian_autograd(
        x_seq[:, t, :], x_proj_seq[:, t, :],
        sol[:, t, :].detach(), R_seq[:, t, :, :],
        gate, use_gate
    )
    jac_anal = compute_jacobian_analytic(
        x_seq[:, t, :], x_proj_seq[:, t, :],
        sol[:, t, :].detach(), R_seq[:, t, :, :],
        gate, use_gate
    )
    diff = (jac_auto - jac_anal).abs()
    print(f"  max diff: {diff.max().item():.2e}")
    print(f"  mean diff: {diff.mean().item():.2e}")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    test_newton_convergence()
