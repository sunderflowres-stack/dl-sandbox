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

        if h0 is None:
            h_current = [
                torch.zeros(batch, self.hidden_size, device=device)
                for _ in range(self.num_layers)
            ]
        else:
            assert h0.shape == (self.num_layers, batch, self.hidden_size)
            h_current = [h0[i] for i in range(self.num_layers)]

        if self.return_sequences:
            hiddens = torch.zeros(batch, seq_len, self.hidden_size, device=device)

        h_last_step = None

        # pre-compute all projections and rotors for every layer in parallel
        # x_projs[layer]: (B, T, H)
        # R_all[layer]:   (B, T, H, H)
        # alphas[layer]:  (B, T, H) or None
        layer_cache = []
        x_in = x
        for layer_idx, cell in enumerate(self.cells):
            x_proj = cell.W_x(x_in)                          # (B, T, H)

            B, T, H = x_proj.shape
            x_proj_flat = x_proj.reshape(B * T, H)
            theta_flat = cell.rotor.mlp(x_proj_flat)         # (B*T, n_params)
            theta = theta_flat.reshape(B, T, -1)

            # build A and compute matrix_exp for all timesteps at once
            n = cell.rotor.n_params
            A = torch.zeros(B, T, H, H, device=device, dtype=x.dtype)
            A[:, :, cell.rotor.tril_i, cell.rotor.tril_j] = theta
            A = A - A.transpose(-2, -1)
            R_all = torch.linalg.matrix_exp(A)               # (B, T, H, H)

            alpha_all = None
            if cell.use_gate:
                # gate needs h which we don't have yet — compute during loop
                alpha_all = None

            layer_cache.append({
                "x_proj": x_proj,
                "R_all": R_all,
            })

            # for next layer pre-computation we need out, but out depends on h
            # so we can only pre-compute layer 0 fully; deeper layers computed in loop
            if layer_idx == 0:
                x_in = None  # signal to stop pre-computing deeper layers
                break

        # recurrent loop — R matrices already computed for layer 0
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_next_list = []

            for layer_idx, cell in enumerate(self.cells):
                h_prev = h_current[layer_idx]

                if layer_idx < len(layer_cache):
                    cache = layer_cache[layer_idx]
                    x_proj_t = cache["x_proj"][:, t, :]
                    R_t = cache["R_all"][:, t, :, :]
                else:
                    x_proj_t = cell.W_x(x_t)
                    theta_t = cell.rotor.mlp(x_proj_t)
                    A_t = torch.zeros(batch, self.hidden_size, self.hidden_size,
                                      device=device, dtype=x.dtype)
                    A_t[:, cell.rotor.tril_i, cell.rotor.tril_j] = theta_t
                    A_t = A_t - A_t.transpose(-2, -1)
                    R_t = torch.linalg.matrix_exp(A_t)

                if cell.use_gate:
                    alpha = torch.sigmoid(
                        cell.gate(torch.cat([x_t, h_prev], dim=-1))
                    )
                    h_new = (R_t @ h_prev.unsqueeze(-1)).squeeze(-1) * (1.0 - alpha) + x_proj_t * alpha
                else:
                    h_new = (R_t @ h_prev.unsqueeze(-1)).squeeze(-1) + x_proj_t

                out = cell.norm(torch.arcsinh(h_new))
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
