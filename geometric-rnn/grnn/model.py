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
            assert h0.shape == (self.num_layers, batch, self.hidden_size)
            h_current = [h0[i] for i in range(self.num_layers)]

        if self.return_sequences:
            hiddens = torch.zeros(batch, seq_len, self.hidden_size, device=device, dtype=dtype)

        h_last_step = None

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_next_list = []

            for layer_idx, cell in enumerate(self.cells):
                h_new, out = cell(x_t, h_current[layer_idx])
                h_next_list.append(h_new)
                x_t = out
                if layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)

            h_current = h_next_list

            if self.return_sequences:
                hiddens[:, t, :] = x_t
            else:
                h_last_step = x_t

        h_last = torch.stack(h_current, dim=0)

        if self.return_sequences:
            out_seq = hiddens
        else:
            out_seq = h_last_step

        if self.readout is not None:
            out_seq = self.readout(out_seq)

        return out_seq, h_last
