import torch
import torch.nn as nn

try:
    from .triton_kernels import rotor_forward_triton
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


class RotorLayer(nn.Module):
    def __init__(self, hidden_size: int, triton: bool = True, order: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_params = (hidden_size * (hidden_size - 1)) // 2
        self.use_triton = triton and _TRITON_AVAILABLE
        self.order = order

        self.W_rot = nn.Linear(hidden_size, self.n_params, bias=True)

        i, j = torch.tril_indices(hidden_size, hidden_size, offset=-1)
        self.register_buffer("tril_i", i)
        self.register_buffer("tril_j", j)

        self.last_A_norm: float = 0.0

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.W_rot.weight, -0.01, 0.01)
        nn.init.zeros_(self.W_rot.bias)

    def _forward_torch(self, h: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        batch_size = h.size(0)

        A = torch.zeros(
            batch_size, self.hidden_size, self.hidden_size,
            device=h.device, dtype=h.dtype,
        )
        A[:, self.tril_i, self.tril_j] = theta
        A = A - A.transpose(1, 2)

        if self.training:
            self.last_A_norm = A.norm(dim=(-2, -1)).mean().item()

        R = torch.linalg.matrix_exp(A)

        return (R @ h.unsqueeze(-1)).squeeze(-1)

    def _forward_triton(self, h: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return rotor_forward_triton(theta, h, self.tril_i, self.tril_j, self.order)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        theta = self.W_rot(h)

        if self.use_triton and h.is_cuda:
            try:
                return self._forward_triton(h, theta)
            except Exception as e:
                if self.training:
                    print(f"[RotorLayer] Triton failed, falling back to torch: {e}")
                self.use_triton = False

        return self._forward_torch(h, theta)
