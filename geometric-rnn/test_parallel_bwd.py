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

def test(B=4, H=16, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Device: {device}, B={B}, H={H}\n")
    torch.manual_seed(0)

    from grnn.rotor import RotorLayer
    from grnn.parallel import GeometricSequentialParallelBwd

    gate = spectral_norm(nn.Linear(H + H, H, bias=True)).to(device)
    rotor = RotorLayer(H, triton=False).to(device)
    h_scale = math.sqrt(H)

    gw = gate.weight.detach().clone().to(device)
    gb = gate.bias.detach().clone().to(device)

    x = torch.randn(B, H, device=device)
    xp = torch.randn(B, H, device=device)
    h_prev = torch.zeros(B, H, device=device)

    theta = rotor.mlp(xp)
    A = torch.zeros(B, H, H, device=device)
    A[:, rotor.tril_i, rotor.tril_j] = theta
    A = A - A.transpose(-2, -1)
    R = torch.linalg.matrix_exp(A).detach()

    grad_out = torch.randn(B, H, device=device)

    # reference gradient
    x_r = x.clone().requires_grad_(True)
    xp_r = xp.clone().requires_grad_(True)
    h_r = single_step(x_r, xp_r, h_prev, R, gw, gb, h_scale)
    h_r.backward(grad_out)
    print("Reference grad_x:    ", x_r.grad[0, :4])
    print("Reference grad_xp:   ", xp_r.grad[0, :4])

    # what our backward computes for t=0 (h_prev=0, dl_dh=0)
    x_t = x.clone().requires_grad_(True)
    xp_t = xp.clone().requires_grad_(True)
    with torch.enable_grad():
        alpha = torch.sigmoid(F.linear(torch.cat([x_t, h_prev], dim=-1), gw, gb))
        Rh = (R @ h_prev.unsqueeze(-1)).squeeze(-1)
        pre = Rh * (1.0 - alpha) + xp_t * alpha
        h_t = F.normalize(pre, dim=-1) * h_scale

    # g = dl_dh[:, t] + grad_output[:, t] — for T=1, dl_dh=0
    g = grad_out
    grads = torch.autograd.grad(h_t, [x_t, xp_t], grad_outputs=g, allow_unused=True)
    print("\nParallel bwd grad_x: ", grads[0][0, :4])
    print("Parallel bwd grad_xp:", grads[1][0, :4])

    dx_diff = (x_r.grad - grads[0]).abs()
    dxp_diff = (xp_r.grad - grads[1]).abs()
    print(f"\ngrad_x diff:  max={dx_diff.max():.2e}")
    print(f"grad_xp diff: max={dxp_diff.max():.2e}")


if __name__ == "__main__":
    test()
