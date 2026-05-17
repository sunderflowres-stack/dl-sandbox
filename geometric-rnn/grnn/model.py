import torch
import torch.nn as nn

from .cell import GeometricRNNCell
from .parallel import GeometricSequentialParallelBwd

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
        use_parallel: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.use_gate = use_gate
        self.use_parallel = use_parallel and use_gate

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
        if self.use_parallel:
            return self._forward_parallel(x, h0)
        return self._forward_sequential(x, h0)

    def _forward_parallel(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        batch, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        if h0 is None:
            h_current = [torch.zeros(batch, self.hidden_size, device=device, dtype=dtype) for _ in range(self.num_layers)]
        else:
            h_current = [h0[i] for i in range(self.num_layers)]

        out_seq = x
        h_last_list = []

        for layer_idx, cell in enumerate(self.cells):
            x_proj_seq = cell.W_x(out_seq)
            
            theta_flat = cell.rotor.theta(x_proj_seq).reshape(batch * seq_len, -1)
            A = torch.zeros(batch * seq_len, self.hidden_size, self.hidden_size, device=device, dtype=dtype)
            A[:, cell.rotor.tril_i, cell.rotor.tril_j] = theta_flat
            A = A - A.transpose(-2, -1)
            R_seq = torch.linalg.matrix_exp(A).view(batch, seq_len, self.hidden_size, self.hidden_size)

            h_seq = GeometricSequentialParallelBwd.apply(
                out_seq, x_proj_seq, R_seq, h_current[layer_idx], 
                cell.gate.weight, cell.gate.bias, cell.h_scale
            )
            
            h_last_list.append(h_seq[:, -1, :])
            out_seq = cell.norm(torch.arcsinh(h_seq))
            
            if layer_idx < self.num_layers - 1:
                out_seq = self.dropout(out_seq)

        h_last = torch.stack(h_last_list, dim=0)
        
        if not self.return_sequences:
            out_seq = out_seq[:, -1, :]

        if self.readout is not None:
            out_seq = self.readout(out_seq)

        return out_seq, h_last

    def _forward_sequential(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        batch, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        if h0 is None:
            h_current = [torch.zeros(batch, self.hidden_size, device=device, dtype=dtype) for _ in range(self.num_layers)]
        else:
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
        out_seq = hiddens if self.return_sequences else h_last_step

        if self.readout is not None:
            out_seq = self.readout(out_seq)

        return out_seq, h_last
