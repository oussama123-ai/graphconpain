"""
models/graph_attention.py
--------------------------
Dynamic Graph Attention Network (GAT) for multimodal neonatal pain assessment.

Architecture (per the paper):
  - 4 modality nodes: facial (17-d), body (102-d), audio (128-d), physio (64-d)
  - Fully connected graph (6 edges + self-loops)
  - 2 GAT layers, 4 attention heads each
  - ELU activation, dropout 0.1
  - Attention pattern adapts based on pain type (vocal vs silent)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Single GAT layer
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """
    One layer of multi-head Graph Attention.

    Parameters
    ----------
    in_dim   : input feature dimension per node
    out_dim  : output dimension per head
    n_heads  : number of attention heads
    concat   : if True concatenate heads; else average (use False for last layer)
    dropout  : dropout on attention weights
    alpha    : LeakyReLU negative slope (paper: 0.2)
    """

    def __init__(
        self,
        in_dim:  int,
        out_dim: int,
        n_heads: int   = 4,
        concat:  bool  = True,
        dropout: float = 0.1,
        alpha:   float = 0.2,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat  = concat

        # Linear transformation per head: W_h ∈ R^{in_dim × out_dim}
        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        # Attention vector per head: a_h ∈ R^{2*out_dim}
        self.a = nn.Parameter(torch.empty(n_heads, 2 * out_dim))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout    = nn.Dropout(dropout)
        self.elu        = nn.ELU()

    def forward(
        self,
        H:   torch.Tensor,   # (B, N, in_dim)  N = 4 modality nodes
        adj: torch.Tensor,   # (N, N) adjacency matrix (binary or weighted)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        H_new  : (B, N, n_heads*out_dim) if concat, else (B, N, out_dim)
        alphas : (B, n_heads, N, N) attention weights
        """
        B, N, _ = H.shape

        # 1. Linear transform: (B, N, in) → (B, n_heads, N, out_dim)
        H_t = torch.einsum("bni,hio->bhno", H, self.W)  # (B, H, N, F')

        # 2. Attention coefficients e_ij = LeakyReLU(a^T [Whi || Whj])
        #    Concat: (B, H, N, 1, F') with (B, H, 1, N, F') → (B, H, N, N, 2F')
        H_i = H_t.unsqueeze(3).expand(-1, -1, -1, N, -1)   # (B,H,N,N,F')
        H_j = H_t.unsqueeze(2).expand(-1, -1, N, -1, -1)   # (B,H,N,N,F')
        concat_ij = torch.cat([H_i, H_j], dim=-1)           # (B,H,N,N,2F')

        # a^T · concat: (B, H, N, N)
        e = self.leaky_relu(
            (concat_ij * self.a[None, :, None, None, :]).sum(-1)
        )

        # 3. Mask with adjacency (+ self-loops)
        adj_mask = (adj + torch.eye(N, device=H.device)).bool()
        e = e.masked_fill(~adj_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # 4. Softmax normalization → attention weights α
        alpha = F.softmax(e, dim=-1)                    # (B, H, N, N)
        alpha = self.dropout(alpha)

        # 5. Weighted aggregation: Σ_j α_ij W h_j
        H_new = torch.einsum("bhnj,bhjf->bhnf", alpha, H_t)  # (B,H,N,F')
        H_new = self.elu(H_new)

        if self.concat:
            H_new = H_new.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B,N,H*F')
        else:
            H_new = H_new.mean(dim=1)                              # (B,N,F')

        return H_new, alpha


# ---------------------------------------------------------------------------
# Full GAT (2 layers + global pooling)
# ---------------------------------------------------------------------------

class GraphAttentionNetwork(nn.Module):
    """
    2-layer GAT for multimodal pain graph.

    Input node features must first be aligned to the same dimension via
    linear projection (handled in GraphConPain wrapper).

    Parameters
    ----------
    node_dim   : unified node feature dimension after projection
    hidden_dim : per-head dimension (paper: 64)
    n_heads    : number of attention heads (paper: 4)
    n_nodes    : number of modality nodes (paper: 4)
    dropout    : dropout on attention weights
    """

    def __init__(
        self,
        node_dim:   int = 64,
        hidden_dim: int = 64,
        n_heads:    int = 4,
        n_nodes:    int = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes

        # Layer 1: concat → output dim = n_heads * hidden_dim
        self.layer1 = GATLayer(node_dim, hidden_dim, n_heads,
                               concat=True,  dropout=dropout)
        # Layer 2: average → output dim = hidden_dim
        self.layer2 = GATLayer(n_heads * hidden_dim, hidden_dim, n_heads,
                               concat=False, dropout=dropout)

        self.layer_norm1 = nn.LayerNorm(n_heads * hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Attention-weighted global pooling
        self.pool_attn = nn.Linear(hidden_dim, 1)

        self.out_dim = hidden_dim

    def build_adjacency(self, n_nodes: int, device) -> torch.Tensor:
        """Fully connected graph (all pairs connected, no self-loop here —
        self-loops are added inside GATLayer)."""
        return torch.ones(n_nodes, n_nodes, device=device) - torch.eye(n_nodes, device=device)

    def forward(
        self,
        H: torch.Tensor,    # (B, N, node_dim)
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Returns
        -------
        z       : (B, hidden_dim)  — graph-level embedding via attention pooling
        attns   : list of alpha tensors from each layer
        """
        B, N, _ = H.shape
        adj = self.build_adjacency(N, H.device)

        # Layer 1
        H1, alpha1 = self.layer1(H, adj)
        H1 = self.layer_norm1(H1)
        H1 = F.dropout(H1, p=0.1, training=self.training)

        # Layer 2
        H2, alpha2 = self.layer2(H1, adj)
        H2 = self.layer_norm2(H2)           # (B, N, hidden_dim)

        # Attention-weighted global pooling  z = Σ_i α_i h_i
        pool_w = F.softmax(self.pool_attn(H2), dim=1)  # (B, N, 1)
        z = (pool_w * H2).sum(dim=1)                   # (B, hidden_dim)

        return z, [alpha1, alpha2]
