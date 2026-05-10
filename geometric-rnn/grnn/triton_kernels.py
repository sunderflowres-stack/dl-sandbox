import torch
import triton
import triton.language as tl

@triton.jit
def _scatter_antisym_kernel(
    theta_ptr,
    A_ptr,
    tril_i_ptr,
    tril_j_ptr,
    N,
    H,
):
    bid = tl.program_id(0)
    kid = tl.program_id(1)

    if kid >= N:
        return

    val = tl.load(theta_ptr + bid * N + kid)
    row = tl.load(tril_i_ptr + kid)
    col = tl.load(tril_j_ptr + kid)

    tl.store(A_ptr + bid * H * H + row * H + col, val)
    tl.store(A_ptr + bid * H * H + col * H + row, -val)


@triton.jit
def _matexp_matvec_kernel(
    A_ptr,
    h_ptr,
    out_ptr,
    H: tl.constexpr,
    H_POW2: tl.constexpr,
    ORDER: tl.constexpr,
):
    bid = tl.program_id(0)

    rows = tl.arange(0, H_POW2)
    cols = tl.arange(0, H_POW2)
    mask = (rows[:, None] < H) & (cols[None, :] < H)

    A = tl.load(
        A_ptr + bid * H * H + rows[:, None] * H + cols[None, :],
        mask=mask,
        other=0.0,
    )

    I = (rows[:, None] == cols[None, :]).to(tl.float32)

    R = I
    term = I

    for k in tl.static_range(1, ORDER + 1):
        term = tl.dot(term, A) * (1.0 / k)
        R = R + term

    h = tl.load(h_ptr + bid * H + rows, mask=rows < H, other=0.0)
    h = tl.reshape(h, (H_POW2, 1))

    out = tl.dot(R, h)
    out = tl.reshape(out, (H_POW2,))

    tl.store(out_ptr + bid * H + rows, out, mask=rows < H)


def _rotor_apply_forward(theta, h, tril_i, tril_j, order):
    B, N = theta.shape
    H = h.shape[1]

    theta32 = theta.contiguous().float()
    h32 = h.contiguous().float()
    return _RotorApplyFn.apply(theta, h, tril_i, tril_j, order)
