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

    # load h with mask
    h = tl.load(h_ptr + bid * H + rows, mask=rows < H, other=0.0)

    # build A via outer-product style masks
    # for each pair (i,j) with i>j, we need theta[k] at A[i,j] and -theta[k] at A[j,i]
    # we do this by accumulating over all k params
    ri = tl.reshape(rows, (H_POW2, 1))  # (H, 1)
    ci = tl.reshape(cols, (1, H_POW2))  # (1, H)

    # identity matrix via mask
    I = (ri == ci).to(tl.float32)

    # build A: iterate over lower-triangular entries
    A = tl.zeros((H_POW2, H_POW2), dtype=tl.float32)
    k = 0
    for i in tl.static_range(1, H):
        for j in tl.static_range(0, i):
            val = tl.load(theta_ptr + bid * N + k)
            mask_ij = (ri == i) & (ci == j)
            mask_ji = (ri == j) & (ci == i)
            A = A + val * mask_ij.to(tl.float32) - val * mask_ji.to(tl.float32)
            k += 1

    # matrix exp via Taylor series: R = I + A + A^2/2! + ... + A^ORDER/ORDER!
    R = I
    term = I
    for o in tl.static_range(1, ORDER + 1):
        term = tl.dot(term, A) * (1.0 / o)
        R = R + term

    # matvec R @ h
    h_col = tl.reshape(h, (H_POW2, 1))
    out = tl.reshape(tl.dot(R, h_col), (H_POW2,))

    tl.store(out_ptr + bid * H + rows, out, mask=rows < H)


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

    theta_c = theta.contiguous().float()
    h_c = h.contiguous().float()
    out = torch.empty_like(h_c)

    _rotor_forward_kernel[(B,)](
        theta_c, h_c, out,
        B, H, N,
        ORDER=order,
        H_POW2=H_POW2,
    )

    return out.to(h.dtype)
