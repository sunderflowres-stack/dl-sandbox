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
    mask2d = (rows[:, None] < H) & (cols[None, :] < H)

    A = tl.load(
        A_ptr + bid * H * H + rows[:, None] * H + cols[None, :],
        mask=mask2d,
        other=0.0,
    )

    I = (rows[:, None] == cols[None, :]).to(tl.float32)

    R = I
    term = I

    for k in tl.static_range(1, ORDER + 1):
        term = tl.dot(term, A) * (1.0 / k)
        R = R + term

    h = tl.load(h_ptr + bid * H + rows, mask=rows < H, other=0.0)
    h_col = tl.reshape(h, (H_POW2, 1))
    out = tl.reshape(tl.dot(R, h_col), (H_POW2,))

    tl.store(out_ptr + bid * H + rows, out, mask=rows < H)

def rotor_apply(
    theta: torch.Tensor,
    h: torch.Tensor,
    tril_i: torch.Tensor,
    tril_j: torch.Tensor,
    order: int = 6,
    track_norm: bool = False,
    module=None,
) -> torch.Tensor:
    B, N = theta.shape
    H = h.shape[1]
    H_POW2 = triton.next_power_of_2(H)
    N_POW2 = triton.next_power_of_2(N)

    theta_c = theta.contiguous().float()
    h_c = h.contiguous().float()

    A = torch.zeros(B, H, H, device=h.device, dtype=torch.float32)

    _scatter_antisym_kernel[(B, N_POW2)](
        theta_c, A,
        tril_i.contiguous(), tril_j.contiguous(),
        N, H,
    )

    if track_norm and module is not None:
        module.last_A_norm = A.norm(dim=(-2, -1)).mean().detach()

    out = torch.empty_like(h_c)
    _matexp_matvec_kernel[(B,)](
        A, h_c, out,
        H=H,
        H_POW2=H_POW2,
        ORDER=order,
    )

    return out.to(h.dtype)

@triton.jit
def _matexp_matrix_kernel(
    A_ptr,
    R_ptr,
    H: tl.constexpr,
    H_POW2: tl.constexpr,
    ORDER: tl.constexpr,
):
    bid = tl.program_id(0)

    rows = tl.arange(0, H_POW2)
    cols = tl.arange(0, H_POW2)
    mask2d = (rows[:, None] < H) & (cols[None, :] < H)

    A = tl.load(
        A_ptr + bid * H * H + rows[:, None] * H + cols[None, :],
        mask=mask2d,
        other=0.0,
    )

    I = (rows[:, None] == cols[None, :]).to(tl.float32)

    R = I
    term = I

    for k in tl.static_range(1, ORDER + 1):
        term = tl.dot(term, A) * (1.0 / k)
        R = R + term

    tl.store(R_ptr + bid * H * H + rows[:, None] * H + cols[None, :], R, mask=mask2d)

def triton_matrix_exp(A: torch.Tensor, order: int = 6) -> torch.Tensor:
    B, H, _ = A.shape
    H_POW2 = triton.next_power_of_2(H)
    
    A_c = A.contiguous().float()
    R = torch.empty_like(A_c)
    
    _matexp_matrix_kernel[(B,)](
        A_c, R,
        H=H,
        H_POW2=H_POW2,
        ORDER=order,
    )
    
    return R.to(A.dtype)
