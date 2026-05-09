import torch
import triton
import triton.language as tl


@triton.jit
def _rotor_forward_kernel(
    theta_ptr,
    h_ptr,
    out_ptr,
    tril_i_ptr,
    tril_j_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    ORDER: tl.constexpr,
):
    bid = tl.program_id(0)

    theta_offsets = bid * N + tl.arange(0, N)
    theta = tl.load(theta_ptr + theta_offsets)

    A = tl.zeros((H, H), dtype=tl.float32)

    for k in tl.static_range(N):
        i = tl.load(tril_i_ptr + k)
        j = tl.load(tril_j_ptr + k)
        val = tl.load(theta_ptr + bid * N + k)
        A[i, j] += val
        A[j, i] -= val

    R = tl.eye(H, dtype=tl.float32)
    term = tl.eye(H, dtype=tl.float32)

    for order in tl.static_range(1, ORDER + 1):
        term = tl.dot(term, A) / order
        R = R + term

    h_offsets = bid * H + tl.arange(0, H)
    h = tl.load(h_ptr + h_offsets)

    h_out = tl.zeros((H,), dtype=tl.float32)
    for row in tl.static_range(H):
        acc = tl.zeros((1,), dtype=tl.float32)
        for col in tl.static_range(H):
            acc += R[row, col] * h[col]
        h_out[row] = acc[0]

    out_offsets = bid * H + tl.arange(0, H)
    tl.store(out_ptr + out_offsets, h_out)


def rotor_forward_triton(
    theta: torch.Tensor,
    h: torch.Tensor,
    tril_i: torch.Tensor,
    tril_j: torch.Tensor,
    order: int = 6,
) -> torch.Tensor:
    B, N = theta.shape
    H = h.shape[1]

    out = torch.empty_like(h)

    _rotor_forward_kernel[(B,)](
        theta, h, out,
        tril_i, tril_j,
        B=B, H=H, N=N,
        ORDER=order,
    )

    return out
