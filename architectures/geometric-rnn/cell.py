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

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_rot = self.rotor(h)
        x_proj = self.W_x(x)

        if self.use_gate:
            alpha = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
            pre = h_rot * (1.0 - alpha) + x_proj * alpha
        else:
            pre = h_rot + x_proj

        return self.norm(torch.arcsinh(pre))

    def init_hidden(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)