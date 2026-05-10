import torch
import torch.nn as nn
from .cell import GeometricRNNCell

class GeometricRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        output_size: int = 0,
        use_gate: bool = False,
        return_sequences: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences

        self.cells = nn.ModuleList([
            GeometricRNNCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                use_gate=use_gate,
            )
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.readout = nn.Linear(hidden_size, output_size) if output_size > 0 else None

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h_current = [
                torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h_current = [h0[i] for i in range(self.num_layers)]

        outputs = [] if self.return_sequences else None

        for t in range(seq_len):
            x_t = x[:, t, :]
            next_hidden = []

            for layer_idx, cell in enumerate(self.cells):
                h_new, out = cell(x_t, h_current[layer_idx])
                next_hidden.append(h_new)

                x_t = out
                if layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            h_current = next_hidden

            if self.return_sequences:
                outputs.append(x_t)

        h_last = torch.stack(h_current, dim=0)

        if self.return_sequences:
            out = torch.stack(outputs, dim=1)
        else:
            out = x_t

        if self.readout is not None:
            out = self.readout(out)

        return out, h_last
