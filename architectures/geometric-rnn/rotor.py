import torch
import torch.nn as nn


class RotorLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_params = (hidden_size * (hidden_size - 1)) // 2

        self.W_rot = nn.Linear(hidden_size, self.n_params, bias=True)

        i, j = torch.tril_indices(hidden_size, hidden_size, offset=-1)
        self.register_buffer("tril_i", i)
        self.register_buffer("tril_j", j)
        self.register_buffer("I", torch.eye(hidden_size))

        self.last_A_norm: float = 0.0

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.W_rot.weight, -0.01, 0.01)
        nn.init.zeros_(self.W_rot.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size = h.size(0)
        theta = self.W_rot(h)

        A = torch.zeros(
            batch_size, self.hidden_size, self.hidden_size,
            device=h.device, dtype=h.dtype,
        )
        A[:, self.tril_i, self.tril_j] = theta
        A = A - A.transpose(1, 2)

        if self.training:
            self.last_A_norm = A.norm(dim=(-2, -1)).mean().item()

        I = self.I.unsqueeze(0)
        R = torch.linalg.solve(I + A, I - A)

        return (R @ h.unsqueeze(-1)).squeeze(-1)