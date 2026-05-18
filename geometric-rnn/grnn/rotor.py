import torch
import torch.nn as nn

try:
    from .triton_kernels import rotor_apply
    _TRITON_AVAILABLE = True
except Exception:
    rotor_apply = None
    _TRITON_AVAILABLE = False

class RotorLayer(nn.Module):
    def __init__(self, hidden_size: int, triton: bool = True, order: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_params = hidden_size * (hidden_size - 1) // 2
        self.order = order
        self.use_triton = triton and _TRITON_AVAILABLE

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, self.n_params, bias=True),
        )

        i, j = torch.tril_indices(hidden_size, hidden_size, offset=-1)
        self.register_buffer("tril_i", i)
        self.register_buffer("tril_j", j)

        self.last_A_norm = None

        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.zeros_(layer.bias)

    def theta(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def _build_A(self, theta: torch.Tensor) -> torch.Tensor:
        batch = theta.shape[0]
        A = torch.zeros(
            batch, self.hidden_size, self.hidden_size,
            device=theta.device, dtype=theta.dtype,
        )
        A[:, self.tril_i, self.tril_j] = theta
        A = A - A.transpose(-2, -1)
        return A

    def _apply_torch(self, theta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        A = self._build_A(theta)

        if self.training:
            self.last_A_norm = A.norm(dim=(-2, -1)).mean().detach()

        R = torch.linalg.matrix_exp(A)
        return torch.matmul(R, h.unsqueeze(-1)).squeeze(-1)

    def apply(self, theta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.use_triton and theta.is_cuda and h.is_cuda:
            try:
                return rotor_apply(
                    theta, h,
                    self.tril_i, self.tril_j,
                    order=self.order,
                    track_norm=self.training,
                    module=self,
                )
            except Exception as e:
                if self.training:
                    print(f"[RotorLayer] Triton failed, falling back: {e}")
                self.use_triton = False

        return self._apply_torch(theta, h)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        theta = self.theta(x)
        return self.apply(theta, h)
