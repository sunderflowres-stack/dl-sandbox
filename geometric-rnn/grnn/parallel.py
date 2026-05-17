import torch
import torch.nn.functional as F


def _parallel_reduce_dense(jacobians: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    J = jacobians.clone()
    r = rhs.clone()
    num_steps = (rhs.shape[1] - 1).bit_length()
    for step in range(num_steps):
        idx = 1 << step
        T = r.shape[1]
        r[:, idx:] += torch.einsum('btij,btj->bti', J[:, idx:], r[:, :T - idx])
        J[:, idx:] = torch.einsum('btij,btjk->btik', J[:, idx:], J[:, :T - idx])
    return r

def _compute_jacobian_cell(x_t, x_proj_t, h_prev, R_t, gw, gb, h_scale):
    H = h_prev.shape[-1]
    W_gate_h = gw[:, x_t.shape[-1]:]

    alpha = torch.sigmoid(F.linear(torch.cat([x_t, h_prev], dim=-1), gw, gb))
    da = alpha * (1.0 - alpha)
    Rh = (R_t @ h_prev.unsqueeze(-1)).squeeze(-1)
    u = Rh * (1.0 - alpha) + x_proj_t * alpha
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u_hat = u / u_norm
    residual = x_proj_t - Rh

    J_u = R_t * (1.0 - alpha).unsqueeze(-1) + \
          (residual * da).unsqueeze(-1) * W_gate_h.unsqueeze(0)

    I = torch.eye(H, device=h_prev.device).unsqueeze(0)
    P = I - u_hat.unsqueeze(-1) * u_hat.unsqueeze(-2)
    J_norm = P / u_norm.unsqueeze(-1)

    return h_scale * torch.bmm(J_norm, J_u)

class GeometricSequentialParallelBwd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x_seq, x_proj_seq, R_seq, h_init, gw, gb, h_scale):
        B, T, H = x_seq.shape
        device = x_seq.device

        h_seq = torch.zeros(B, T, H, device=device, dtype=x_seq.dtype)
        h = h_init.detach().clone()

        # forward is always no_grad
        with torch.no_grad():
            for t in range(T):
                alpha = torch.sigmoid(
                    F.linear(torch.cat([x_seq[:, t], h], dim=-1), gw, gb)
                )
                Rh = (R_seq[:, t] @ h.unsqueeze(-1)).squeeze(-1)
                pre = Rh * (1.0 - alpha) + x_proj_seq[:, t] * alpha
                h = F.normalize(pre, dim=-1) * h_scale
                h_seq[:, t] = h

        ctx.save_for_backward(
            x_seq.detach(), x_proj_seq.detach(), R_seq.detach(),
            h_seq.detach(), gw.detach(), gb.detach()
        )
        ctx.h_scale = h_scale
        return h_seq

    @staticmethod
    def backward(ctx, grad_output):
        x_seq, x_proj_seq, R_seq, h_seq, gw, gb = ctx.saved_tensors
        h_scale = ctx.h_scale
        B, T, H = x_seq.shape
        device = x_seq.device

        h_prev_seq = torch.zeros_like(h_seq)
        h_prev_seq[:, 1:] = h_seq[:, :-1]

        with torch.no_grad():
            jac_scan = torch.zeros(B, T, H, H, device=device)
            for t in range(T - 1):
                J_next = _compute_jacobian_cell(
                    x_seq[:, t + 1], x_proj_seq[:, t + 1],
                    h_prev_seq[:, t + 1], R_seq[:, t + 1],
                    gw, gb, h_scale
                )
                jac_scan[:, t] = J_next.transpose(-1, -2)

        jac_flipped = torch.flip(jac_scan, dims=[1])
        grad_flipped = torch.flip(grad_output.detach(), dims=[1])
        dl_dh_flipped = _parallel_reduce_dense(jac_flipped, grad_flipped)
        dl_dh = torch.flip(dl_dh_flipped, dims=[1])

        grad_x = torch.zeros_like(x_seq)
        grad_x_proj = torch.zeros_like(x_proj_seq)

        for t in range(T):
            x_t = x_seq[:, t].clone().requires_grad_(True)
            xp_t = x_proj_seq[:, t].clone().requires_grad_(True)
            h_prev_t = h_prev_seq[:, t].detach()

            with torch.enable_grad():
                alpha = torch.sigmoid(F.linear(torch.cat([x_t, h_prev_t], dim=-1), gw, gb))
                Rh = (R_seq[:, t] @ h_prev_t.unsqueeze(-1)).squeeze(-1)
                pre = Rh * (1.0 - alpha) + xp_t * alpha
                h_t = F.normalize(pre, dim=-1) * h_scale

            # dl_dh already contains grad_output from parallel reduce
            g = dl_dh[:, t].detach()
            grads = torch.autograd.grad(h_t, [x_t, xp_t], grad_outputs=g, allow_unused=True)
            if grads[0] is not None:
                grad_x[:, t] = grads[0].detach()
            if grads[1] is not None:
                grad_x_proj[:, t] = grads[1].detach()

        return grad_x, grad_x_proj, None, None, None, None, None
