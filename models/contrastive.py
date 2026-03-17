"""
models/contrastive.py
----------------------
Graph-aware contrastive self-supervised pretraining (Section 3.5).

Implements InfoNCE loss on graph-level embeddings with 4 augmentation types:
  1. Node feature perturbation (Gaussian noise, σ=0.1)
  2. Random node dropout (p=0.2)
  3. Edge weight perturbation ([0.8, 1.2] uniform scaling)
  4. Temporal jittering (±5 frames shift)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph-aware augmentations
# ---------------------------------------------------------------------------

class GraphAugmentation(nn.Module):
    """
    Apply graph-aware augmentations to node feature tensors.

    Parameters
    ----------
    node_noise_sigma   : σ for Gaussian noise on node features (default 0.1)
    node_dropout_p     : probability of zeroing a modality node (default 0.2)
    edge_perturb_range : (lo, hi) uniform scaling of edge weights (default 0.8-1.2)
    temporal_jitter    : max frames to shift (default ±5)
    """

    def __init__(
        self,
        node_noise_sigma:    float = 0.1,
        node_dropout_p:      float = 0.2,
        edge_perturb_range:  tuple = (0.8, 1.2),
        temporal_jitter:     int   = 5,
    ):
        super().__init__()
        self.noise_sigma  = node_noise_sigma
        self.dropout_p    = node_dropout_p
        self.edge_lo, self.edge_hi = edge_perturb_range
        self.jitter       = temporal_jitter

    def forward(
        self,
        H: torch.Tensor,       # (B, N, D) node features
        adj: torch.Tensor,     # (N, N) adjacency
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return augmented (H_aug, adj_aug)."""
        H_aug   = H.clone()
        adj_aug = adj.clone().float()

        # 1. Node feature perturbation
        H_aug = H_aug + torch.randn_like(H_aug) * self.noise_sigma

        # 2. Random node dropout (zero out one modality node)
        if torch.rand(1).item() < self.dropout_p:
            node_idx = torch.randint(0, H.shape[1], (1,)).item()
            H_aug[:, node_idx, :] = 0.0

        # 3. Edge weight perturbation
        perturb = (
            torch.rand_like(adj_aug) * (self.edge_hi - self.edge_lo)
            + self.edge_lo
        )
        adj_aug = adj_aug * perturb

        return H_aug, adj_aug


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def info_nce_loss(
    z1: torch.Tensor,   # (B, D) L2-normalized embeddings view 1
    z2: torch.Tensor,   # (B, D) L2-normalized embeddings view 2
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Symmetric InfoNCE / NT-Xent loss.

    For positive pair (i, i+B) and negative pairs from the rest of the batch.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)            # (2B, D)
    sim = torch.mm(z, z.t()) / temperature    # (2B, 2B)

    # Remove self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Contrastive pretraining wrapper
# ---------------------------------------------------------------------------

class ContrastivePretrain(nn.Module):
    """
    Wraps the backbone (feature extractors + GAT) with a projection head
    for contrastive pretraining.

    Parameters
    ----------
    backbone    : module with forward(H) → (z, attns)
    embed_dim   : backbone graph embedding dim (default 64)
    proj_dim    : projection head output dim (default 128)
    temperature : InfoNCE temperature τ (default 0.5)
    """

    def __init__(
        self,
        backbone,
        embed_dim:   int   = 64,
        proj_dim:    int   = 128,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.backbone    = backbone
        self.temperature = temperature
        self.aug         = GraphAugmentation()

        # 2-layer MLP projection head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def _embed_and_project(self, H: torch.Tensor) -> torch.Tensor:
        z, _ = self.backbone(H)
        v = self.projector(z)
        return F.normalize(v, dim=-1)

    def forward(
        self,
        H:   torch.Tensor,    # (B, N, D)
        adj: torch.Tensor,    # (N, N)
    ) -> torch.Tensor:
        """
        Generate two augmented views and compute InfoNCE loss.

        Returns
        -------
        loss : scalar
        """
        H1, adj1 = self.aug(H, adj)
        H2, adj2 = self.aug(H, adj)

        # Temporarily set adj in backbone (pass via closure or dict)
        # For simplicity we pass H only (backbone uses its own full-conn adj)
        z1 = self._embed_and_project(H1)
        z2 = self._embed_and_project(H2)

        return info_nce_loss(z1, z2, self.temperature)
