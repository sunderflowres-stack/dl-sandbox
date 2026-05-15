import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

sys.path.insert(0, '.')

def sequential_forward_backward(x_seq, x_proj_seq, R_seq, gw, gb, h_scale):
    """Reference: standard autograd through sequential loop."""
    B, T, H = x_seq.shape
    h = torch.zeros(B, H, device=x_seq.device)
    h_seq = []
    for t in range(T):
        alpha = torch.sigmoid(F.linear(torch.cat([x_seq[:, t], h], dim=-1), gw, gb))
        Rh = (R_seq[:, t] @ h.unsqueeze(-1)).squeeze(-1)
        pre = Rh * (1.0 - alpha) + x_proj_seq[:, t] * alpha
        h = F.normalize(pre, dim=-1) * h_scale
        h_seq.append(h)
    return torch.stack(h_seq, dim=1)

def test_gradient_correctness(B=4, T=16, H=16, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Device: {device}, B={B}, T={T}, H={H}\n")
    torch.manual_seed(0)

    from grnn.rotor import RotorLayer
    from grnn.parallel import GeometricSequentialParallelBwd

    gate = spectral_norm(nn.Linear(H + H, H, bias=True)).to(device)
    rotor = RotorLayer(H, triton=False).to(device)
    h_scale = math.sqrt(H)

    x_seq = torch.randn(B, T, H, device=device, requires_grad=True)
    x_proj_seq = torch.randn(B, T, H, device=device, requires_grad=True)

    theta_flat = rotor.mlp(x_proj_seq.detach().reshape(B * T, H))
    A = torch.zeros(B * T, H, H, device=device)
    A[:, rotor.tril_i, rotor.tril_j] = theta_flat
    A = A - A.transpose(-2, -1)
    R_seq = torch.linalg.matrix_exp(A).reshape(B, T, H, H).detach()

    h_init = torch.zeros(B, H, device=device)
    grad_out = torch.randn(B, T, H, device=device)

    gw = gate.weight.detach().clone().requires_grad_(True)
    gb = gate.bias.detach().clone().requires_grad_(True)

    # reference: standard autograd
    x1 = x_seq.detach().requires_grad_(True)
    xp1 = x_proj_seq.detach().requires_grad_(True)
    h_ref = sequential_forward_backward(x1, xp1, R_seq, gw, gb, h_scale)
    h_ref.backward(grad_out)
    grad_x_ref = x1.grad.clone()
    grad_xp_ref = xp1.grad.clone()

    # parallel backward
    x2 = x_seq.detach().requires_grad_(True)
    xp2 = x_proj_seq.detach().requires_grad_(True)
    h_par = GeometricSequentialParallelBwd.apply(
        x2, xp2, R_seq, h_init,
        gw.detach(), gb.detach(), h_scale
    )
    h_par.backward(grad_out)
    grad_x_par = x2.grad.clone()
    grad_xp_par = xp2.grad.clone()

    # compare forward outputs
    fwd_diff = (h_ref.detach() - h_par.detach()).abs()
    print(f"Forward output diff:  max={fwd_diff.max():.2e}  mean={fwd_diff.mean():.2e}")

    # compare gradients
    dx_diff = (grad_x_ref - grad_x_par).abs()
    dxp_diff = (grad_xp_ref - grad_xp_par).abs()
    print(f"grad_x diff:          max={dx_diff.max():.2e}  mean={dx_diff.mean():.2e}")
    print(f"grad_x_proj diff:     max={dxp_diff.max():.2e}  mean={dxp_diff.mean():.2e}")

    ok = dx_diff.max() < 1e-2 and dxp_diff.max() < 1e-2
    print(f"\n{'PASSED' if ok else 'FAILED'}")


if __name__ == "__main__":
    test_gradient_correctness()
