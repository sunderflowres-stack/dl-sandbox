import torch
import torch.nn as nn
import torch.nn.functional as F

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
            self.gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        if self.use_gate:
            nn.init.xavier_uniform_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_proj = self.W_x(x)

        # R is computed from x, not h — parallel across time steps
        R = self.rotor(x_proj)

        if self.use_gate:
            alpha = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
            h_new = (R @ h.unsqueeze(-1)).squeeze(-1) * (1.0 - alpha) + x_proj * alpha
        else:
            h_new = (R @ h.unsqueeze(-1)).squeeze(-1) + x_proj

        # nonlinearity applied to output, not to recurrent state
        out = self.norm(torch.arcsinh(h_new))

        return h_new, out

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)
