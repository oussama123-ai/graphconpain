"""
models/temporal_model.py
-------------------------
Bidirectional GRU for temporal modeling of pain episode sequences.

Justification (Table 5 in paper):
  BiGRU achieves 88.5% accuracy with 2.3h training / 10.1GB memory,
  outperforming BiLSTM (87.4%, 2.8h, 11.8GB) and Transformer (87.1%, 3.4h, 14.2GB).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """
    2-layer Bidirectional GRU.

    Parameters
    ----------
    input_dim   : dimension of each graph-level embedding z_t (default 64)
    hidden_dim  : per-direction hidden size (paper: 512)
    n_layers    : number of GRU layers (paper: 2)
    dropout     : inter-layer dropout (paper: 0.2)

    Input  : (B, T, input_dim)
    Output : (B, 2 * hidden_dim)  — concatenated final forward/backward states
    """

    def __init__(
        self,
        input_dim:  int   = 64,
        hidden_dim: int   = 512,
        n_layers:   int   = 2,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.out_dim = 2 * hidden_dim   # 1024 for hidden=512

    def forward(
        self,
        x:      torch.Tensor,              # (B, T, input_dim)
        lengths: torch.Tensor | None = None,  # (B,) — for packed sequences
    ) -> torch.Tensor:
        """
        Returns the concatenated final hidden states from both directions.

        Output : (B, 2 * hidden_dim)
        """
        if lengths is not None:
            # Pack padded sequence for efficiency
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(x_packed)
        else:
            _, h_n = self.gru(x)

        # h_n : (n_layers * 2, B, hidden_dim)
        # Take last layer's forward and backward states
        fwd = h_n[-2]   # last layer, forward  (B, hidden_dim)
        bwd = h_n[-1]   # last layer, backward (B, hidden_dim)
        return torch.cat([fwd, bwd], dim=-1)   # (B, 2 * hidden_dim)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return full output sequence (B, T, 2*hidden_dim)."""
        out, _ = self.gru(x)
        return out
