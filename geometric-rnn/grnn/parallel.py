import math
import torch
import torch.nn.functional as F


def _parallel_reduce_dense(jacobians: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """
    Parallel prefix reduction for dense Jacobians.
    jacobians: (B, T, H, H)
    rhs:       (B, T, H)
    Returns:   (B, T, H)
    """
    J = jacobians.clone()
    r = rhs.clone()
    num_steps = (rhs.shape[1] - 1).bit_length()
    for step in range(num_steps):
        idx = 1 << step
        T = r.shape[1]
        r[:, idx:] -= torch.einsum('btij,btj->bti', J[:, idx:], r[:, :T - idx])
        J[:, idx:] = torch.einsum('btij,btjk->btik', -J[:, idx:], J[:, :T - idx])
        J[:, :idx] = 0.0
    return r


def _compute_jacobian_cell(x_t, x_proj_t, h_prev, R_t, gate, h_scale):
    """
    Analytic Jacobian df/dh_{t-1} for GeometricRNNCell with normalized recurrence.
    h_new = normalize(R@h*(1-a) + x_proj*a) * scale
    """
    H = h_prev.shape[-1]
    device = h_prev.device

    W_gate = gate.weight                        # (H, 2H)
    W_gate_h = W_gate[:, x_t.shape[-1]:]       # (H, H)

    alpha = torch.sigmoid(gate(torch.cat([x_t, h_prev], dim=-1)))
    da = alpha * (1.0 - alpha)

    Rh = (R_t @ h_prev.unsqueeze(-1)).squeeze(-1)
    u = Rh * (1.0 - alpha) + x_proj_t * alpha
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u_hat = u / u_norm

    residual = x_proj_t - Rh
    J_u = R_t * (1.0 - alpha).unsqueeze(-2) + \
          (residual * da).unsqueeze(-1) * W_gate_h.unsqueeze(0)

    I = torch.eye(H, device=device).unsqueeze(0)
    P = I - u_hat.unsqueeze(-1) * u_hat.unsqueeze(-2)
    J_norm = P / u_norm.unsqueeze(-1)

    return h_scale * torch.bmm(J_norm, J_u)    # (B, H, H)


class GeometricSequentialParallelBwd(torch.autograd.Function):
    """
    Sequential forward, parallel backward via prefix scan on Jacobians.
    """

    @staticmethod
    def forward(ctx, x_seq, x_proj_seq, R_seq, h_init, gate_weight, gate_bias, h_scale, gate_module):
        B, T, H = x_seq.shape
        device = x_seq.device

        h_seq = torch.zeros(B, T, H, device=device, dtype=x_seq.dtype)
        h = h_init.clone()

        for t in range(T):
            alpha = torch.sigmoid(
                F.linear(torch.cat([x_seq[:, t], h], dim=-1), gate_weight, gate_bias)
            )
            Rh = (R_seq[:, t] @ h.unsqueeze(-1)).squeeze(-1)
            pre = Rh * (1.0 - alpha) + x_proj_seq[:, t] * alpha
            h = F.normalize(pre, dim=-1) * h_scale
            h_seq[:, t] = h

        ctx.save_for_backward(x_seq, x_proj_seq, R_seq, h_seq, gate_weight, gate_bias)
        ctx.h_scale = h_scale
        ctx.gate_module = gate_module
        return h_seq

    @staticmethod
    def backward(ctx, grad_output):
        x_seq, x_proj_seq, R_seq, h_seq, gate_weight, gate_bias = ctx.saved_tensors
        h_scale = ctx.h_scale
        gate_module = ctx.gate_module
        B, T, H = x_seq.shape
        device = x_seq.device

        # build h_prev sequence
        h_prev_seq = torch.zeros_like(h_seq)
        h_prev_seq[:, 1:] = h_seq[:, :-1]

        # compute backward Jacobians: J_bwd[t] = J_fwd[T-1-t]^T, J_bwd[0] = 0
        jac_bwd = torch.zeros(B, T, H, H, device=device)
        for t in range(T):
            t_fwd = T - 1 - t
            if t == 0:
                jac_bwd[:, t] = 0.0
                continue
            J_t = _compute_jacobian_cell(
                x_seq[:, t_fwd], x_proj_seq[:, t_fwd],
                h_prev_seq[:, t_fwd], R_seq[:, t_fwd],
                gate_module, h_scale
            )
            jac_bwd[:, t] = J_t.transpose(-1, -2)

        # flip grad_output to match backward time direction
        grad_flipped = torch.flip(grad_output, dims=[1])

        # parallel reduce: dl/dh_{t-1} from dl/dh_t
        dl_dh = _parallel_reduce_dense(jac_bwd, grad_flipped)

        # flip back to forward time order
        dl_dh = torch.flip(dl_dh, dims=[1])

        # grad wrt x_seq and x_proj_seq via autograd through single steps
        grad_x = torch.zeros_like(x_seq)
        grad_x_proj = torch.zeros_like(x_proj_seq)

        for t in range(T):
            x_t = x_seq[:, t].detach().requires_grad_(True)
            x_proj_t = x_proj_seq[:, t].detach().requires_grad_(True)
            h_prev_t = h_prev_seq[:, t].detach()

            with torch.enable_grad():
                alpha = torch.sigmoid(
                    F.linear(torch.cat([x_t, h_prev_t], dim=-1), gate_weight, gate_bias)
                )
                Rh = (R_seq[:, t] @ h_prev_t.unsqueeze(-1)).squeeze(-1)
                pre = Rh * (1.0 - alpha) + x_proj_t * alpha
                h_t = F.normalize(pre, dim=-1) * h_scale

            g = dl_dh[:, t] + grad_output[:, t]
            grads = torch.autograd.grad(h_t, [x_t, x_proj_t], grad_outputs=g, allow_unused=True)
            grad_x[:, t] = grads[0] if grads[0] is not None else 0.0
            grad_x_proj[:, t] = grads[1] if grads[1] is not None else 0.0

        return grad_x, grad_x_proj, None, None, None, None, None, None
