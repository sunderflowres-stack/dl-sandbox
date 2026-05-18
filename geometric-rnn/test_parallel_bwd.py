import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

sys.path.insert(0, '.')

def single_step(x, x_proj, h_prev, R, gw, gb, h_scale):
    alpha = torch.sigmoid(F.linear(torch.cat([x, h_prev], dim=-1), gw, gb))
    Rh = (R @ h_prev.unsqueeze(-1)).squeeze(-1)
    pre = Rh * (1.0 - alpha) + x_proj * alpha
    return F.normalize(pre, dim=-1) * h_scale

def sequential_bwd(x_seq, x_proj_seq, R_seq, h_seq, gw, gb, h_scale, grad_out):
    """
    True sequential backward:
    dl/dh_{t-1} = J_t^T @ dl/dh_t
    dl/dx_t = d(h_t)/dx_t^T @ dl/dh_t  (local grad)
    """
    B, T, H = x_seq.shape
    grad_x = torch.zeros_like(x_seq)
    grad_xp = torch.zeros_like(x_proj_seq)

    h_prev_seq = torch.zeros_like(h_seq)
    h_prev_seq[:, 1:] = h_seq[:, :-1]

    dl_dh = torch.zeros(B, H, device=x_seq.device)

    for t in reversed(range(T)):
        dl_dh_t = dl_dh + grad_out[:, t]

        x_t = x_seq[:, t].clone().requires_grad_(True)
        xp_t = x_proj_seq[:, t].clone().requires_grad_(True)
        h_prev_t = h_prev_seq[:, t].detach().requires_grad_(True)

        with torch.enable_grad():
            h_t = single_step(x_t, xp_t, h_prev_t, R_seq[:, t], gw, gb, h_scale)

        grads = torch.autograd.grad(
            h_t, [x_t, xp_t, h_prev_t], grad_outputs=dl_dh_t, allow_unused=True
        )
        grad_x[:, t] = grads[0].detach()
        grad_xp[:, t] = grads[1].detach()
        dl_dh = grads[2].detach() if grads[2] is not None else torch.zeros(B, H, device=x_seq.device)

    return grad_x, grad_xp

def test(B=2, T=4, H=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Device: {device}, B={B}, T={T}, H={H}\n")
    torch.manual_seed(0)

    from grnn.rotor import RotorLayer
    from grnn.parallel import GeometricSequentialParallelBwd, _compute_jacobian_cell, _parallel_reduce_dense

    gate = spectral_norm(nn.Linear(H + H, H, bias=True)).to(device)
    rotor = RotorLayer(H, triton=False).to(device)
    h_scale = math.sqrt(H)
    gw = gate.weight.detach().clone().to(device)
    gb = gate.bias.detach().clone().to(device)

    x_seq = torch.randn(B, T, H, device=device)
    x_proj_seq = torch.randn(B, T, H, device=device)

    theta_flat = rotor.mlp(x_proj_seq.reshape(B * T, H))
    A = torch.zeros(B * T, H, H, device=device)
    A[:, rotor.tril_i, rotor.tril_j] = theta_flat
    A = A - A.transpose(-2, -1)
    R_seq = torch.linalg.matrix_exp(A).reshape(B, T, H, H).detach()

    h_init = torch.zeros(B, H, device=device)
    grad_out = torch.randn(B, T, H, device=device)

    # run forward to get h_seq
    h_seq = torch.zeros(B, T, H, device=device)
    h = h_init.clone()
    with torch.no_grad():
        for t in range(T):
            h = single_step(x_seq[:, t], x_proj_seq[:, t], h, R_seq[:, t], gw, gb, h_scale)
            h_seq[:, t] = h

    # reference: true sequential backward
    grad_x_ref, grad_xp_ref = sequential_bwd(
        x_seq, x_proj_seq, R_seq, h_seq, gw, gb, h_scale, grad_out
    )

    # parallel backward via our Function
    x2 = x_seq.clone().requires_grad_(True)
    xp2 = x_proj_seq.clone().requires_grad_(True)
    h_par = GeometricSequentialParallelBwd.apply(
        x2, xp2, R_seq, h_init, gw, gb, h_scale
    )
    h_par.backward(grad_out)

    dx_diff = (grad_x_ref - x2.grad).abs()
    dxp_diff = (grad_xp_ref - xp2.grad).abs()

    print(f"grad_x diff:   max={dx_diff.max():.2e}  mean={dx_diff.mean():.2e}")
    print(f"grad_xp diff:  max={dxp_diff.max():.2e}  mean={dxp_diff.mean():.2e}")

    # also check parallel_reduce directly
    print("\nChecking parallel_reduce vs sequential accumulation:")
    h_prev_seq = torch.zeros_like(h_seq)
    h_prev_seq[:, 1:] = h_seq[:, :-1]

    # build backward jacobians
    jac_bwd = torch.zeros(B, T, H, H, device=device)
    for t in range(1, T):
        t_fwd = T - t
        J_t = _compute_jacobian_cell(
            x_seq[:, t_fwd], x_proj_seq[:, t_fwd],
            h_prev_seq[:, t_fwd], R_seq[:, t_fwd],
            gw, gb, h_scale
        )
        jac_bwd[:, t] = J_t.transpose(-1, -2)

    grad_flipped = torch.flip(grad_out, dims=[1])
    dl_dh_par = _parallel_reduce_dense(jac_bwd, grad_flipped)
    dl_dh_par = torch.flip(dl_dh_par, dims=[1])

    # sequential accumulation of dl/dh
    dl_dh_seq = torch.zeros(B, T, H, device=device)
    acc = torch.zeros(B, H, device=device)
    for t in reversed(range(T)):
        acc = acc + grad_out[:, t]
        dl_dh_seq[:, t] = acc
        if t > 0:
            J_fwd = _compute_jacobian_cell(
                x_seq[:, t], x_proj_seq[:, t],
                h_prev_seq[:, t], R_seq[:, t],
                gw, gb, h_scale
            )
            acc = torch.einsum('bij,bj->bi', J_fwd.transpose(-1, -2), acc)

    diff = (dl_dh_seq - dl_dh_par).abs()
    print(f"  dl_dh diff: max={diff.max():.2e}  mean={diff.mean():.2e}")

if __name__ == "__main__":
    test()
