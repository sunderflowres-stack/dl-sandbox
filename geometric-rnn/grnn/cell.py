import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .rotor import RotorLayer

class GeometricRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, use_gate: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_gate = use_gate

        self.rotor = RotorLayer(hidden_size)
        self.W_x = nn.Linear(input_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

        if use_gate:
            # spectral_norm constrains ||W||_2 <= 1, keeping spec_rad(Jacobian) < 1
            self.gate = spectral_norm(
                nn.Linear(input_size + hidden_size, hidden_size, bias=True)
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        if self.use_gate:
            nn.init.xavier_uniform_(self.gate.weight_orig)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_proj = self.W_x(x)
        theta = self.rotor.theta(x_proj)

        if self.use_gate:
            alpha = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
            rotated = self.rotor.apply(theta, h)
            h_new = rotated * (1.0 - alpha) + x_proj * alpha
        else:
            h_new = self.rotor.apply(theta, h) + x_proj

        out = self.norm(torch.arcsinh(h_new))
        return h_new, out

    def init_hidden(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
