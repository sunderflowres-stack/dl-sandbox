import torch
import triton
import triton.language as tl


@triton.jit
def _rotor_forward_kernel(
    theta_ptr,
    h_ptr,
    out_ptr,
    B,
    H: tl.constexpr,
    N,
    ORDER: tl.constexpr,
    H_POW2: tl.constexpr,
):
    bid = tl.program_id(0)

    rows = tl.arange(0, H_POW2)
    cols = tl.arange(0, H_POW2)

    row_mask = rows < H
    col_mask = cols < H

    h = tl.load(h_ptr + bid * H + rows, mask=row_mask, other=0.0)

    A_flat = tl.zeros((H_POW2 * H_POW2,), dtype=tl.float32)

    n_params = H * (H - 1) // 2
    param_range = tl.arange(0, 1)

    R_rows = tl.zeros((H_POW2, H_POW2), dtype=tl.float32)
    for diag in tl.static_range(H_POW2):
        for row in tl.static_range(H_POW2):
            R_rows[row, row] = 1.0

    A = tl.zeros((H_POW2, H_POW2), dtype=tl.float32)

    k = 0
    for i in tl.static_range(H_POW2):
        for j in tl.static_range(H_POW2):
            if i > j and i < H and j < H:
                val = tl.load(theta_ptr + bid * N + k)
                A[i, j] = val
                A[j, i] = -val
                k += 1

    R = tl.zeros((H_POW2, H_POW2), dtype=tl.float32)
    for i in tl.static_range(H_POW2):
        R[i, i] = 1.0

    term = tl.zeros((H_POW2, H_POW2), dtype=tl.float32)
    for i in tl.static_range(H_POW2):
        term[i, i] = 1.0

    for _ in tl.static_range(ORDER):
        term = tl.dot(term, A)
        R = R + term

    h_col = tl.reshape(h, (H_POW2, 1))
    out_mat = tl.dot(R, h_col)
    out = tl.reshape(out_mat, (H_POW2,))

    tl.store(out_ptr + bid * H + rows, out, mask=row_mask)


def rotor_forward_triton(
    theta: torch.Tensor,
    h: torch.Tensor,
    tril_i: torch.Tensor,
    tril_j: torch.Tensor,
    order: int = 6,
) -> torch.Tensor:
    B, N = theta.shape
    H = h.shape[1]

    H_POW2 = triton.next_power_of_2(H)

    theta_c = theta.contiguous()
    h_c = h.contiguous().float()
    out = torch.empty_like(h_c)

    _rotor_forward_kernel[(B,)](
        theta_c, h_c, out,
        B, H, N,
        ORDER=order,
        H_POW2=H_POW2,
    )

    return out.to(h.dtype)
