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

        # MLP: x -> theta (rotation parameters)
        # input is x_proj which has shape (B, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, self.n_params, bias=True),
        )

        i, j = torch.tril_indices(hidden_size, hidden_size, offset=-1)
        self.register_buffer("tril_i", i)
        self.register_buffer("tril_j", j)

        self.last_A_norm: float = 0.0

        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.01, 0.01)
                nn.init.zeros_(layer.bias)

    def _build_A(self, theta: torch.Tensor) -> torch.Tensor:
        batch_size = theta.size(0)
        A = torch.zeros(
            batch_size, self.hidden_size, self.hidden_size,
            device=theta.device, dtype=theta.dtype,
        )
        A[:, self.tril_i, self.tril_j] = theta
        A = A - A.transpose(1, 2)
        return A

    def _forward_torch(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A = self._build_A(theta)

        if self.training:
            self.last_A_norm = A.norm(dim=(-2, -1)).mean().item()

        R = torch.linalg.matrix_exp(A)
        return R

    def _forward_triton(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # triton returns R @ h directly, but now we return R for use in cell
        # so we use torch path for R construction, triton for matvec is in cell
        return self._forward_torch(x, theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.mlp(x)

        if self.training:
            A = self._build_A(theta)
            self.last_A_norm = A.norm(dim=(-2, -1)).mean().item()
            R = torch.linalg.matrix_exp(A)
            return R

        A = self._build_A(theta)
        R = torch.linalg.matrix_exp(A)
        return R
