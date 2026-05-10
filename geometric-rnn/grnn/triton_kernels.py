import torch

class _RotorApplyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, h, tril_i, tril_j, order):
        B, N = theta.shape
        H = h.shape[1]

        theta32 = theta.contiguous().float()
        h32 = h.contiguous().float()

        A = torch.zeros(B, H, H, device=theta.device, dtype=torch.float32)

        _scatter_antisym_kernel[(B, triton.next_power_of_2(N))](
            theta32,
            A,
            tril_i.contiguous(),
            tril_j.contiguous(),
            N,
            H,
        )

        out = torch.empty_like(h32)

        _matexp_matvec_kernel[(B,)](
            A,
            h32,
            out,
            H=H,
            H_POW2=triton.next_power_of_2(H),
            ORDER=order,
        )

        ctx.save_for_backward(theta, h, tril_i, tril_j)
        ctx.order = order

        return out.to(h.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        theta, h, tril_i, tril_j = ctx.saved_tensors
        order = ctx.order

        with torch.enable_grad():
            theta_ = theta.detach().requires_grad_(theta.requires_grad)
            h_ = h.detach().requires_grad_(h.requires_grad)

            out = _RotorApplyFn.forward.__func__(ctx, theta_, h_, tril_i, tril_j, order)
            grads = torch.autograd.grad(
                out,
                (theta_, h_),
                grad_out,
                allow_unused=True,
            )

        grad_theta, grad_h = grads
        return grad_theta, grad_h, None, None, None


def rotor_apply(
    theta: torch.Tensor,
    h: torch.Tensor,
    tril_i: torch.Tensor,
    tril_j: torch.Tensor,
    order: int = 6,
    track_norm: bool = False,
    module=None,
) -> torch.Tensor:
    if track_norm and module is not None:
        with torch.no_grad():
            batch = theta.shape[0]
            H = h.shape[1]
            A = torch.zeros(batch, H, H, device=theta.device, dtype=theta.dtype)
            A[:, tril_i, tril_j] = theta
            A = A - A.transpose(-2, -1)
            module.last_A_norm = A.norm(dim=(-2, -1)).mean().item()

    return _RotorApplyFn.apply(theta, h, tril_i, tril_j, order)
