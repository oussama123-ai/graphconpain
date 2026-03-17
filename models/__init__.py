"""
models/__init__.py
-------------------
Full GraphConPain model: feature projection → GAT → BiGRU → multi-task heads.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractors import FacialExtractor, BodyExtractor, AudioExtractor, PhysioExtractor
from .graph_attention     import GraphAttentionNetwork
from .temporal_model      import TemporalModel
from .multitask_head      import MultiTaskHead
from .contrastive         import ContrastivePretrain, GraphAugmentation


class GraphConPain(nn.Module):
    """
    End-to-end GraphConPain model.

    Pipeline
    --------
    [facial(T,17), body(T,102), audio(T,65), physio(T,3,250)]
        ↓  feature extractors
    node features (T, N=4, varying_dim)
        ↓  linear projections → unified node_dim
    (T, 4, node_dim)
        ↓  GAT (per timestep)
    graph embeddings (T, graph_embed_dim)
        ↓  BiGRU
    temporal embedding (2*gru_hidden)
        ↓  MultiTaskHead
    {continuous_score, class_logits, silent_logit}

    Parameters
    ----------
    node_dim        : unified projection dimension per node (default 64)
    gat_hidden      : per-head hidden dim in GAT (default 64)
    gat_heads       : number of attention heads (default 4)
    gru_hidden      : per-direction GRU hidden size (default 512)
    gru_layers      : number of GRU layers (default 2)
    n_classes       : pain levels (default 4)
    dropout         : global dropout rate (default 0.1)
    silent_pos_weight : class weight for imbalanced silent pain BCE
    """

    # raw feature dims from each extractor
    FEAT_DIMS = {"facial": 17, "body": 102, "audio": 128, "physio": 64}

    def __init__(
        self,
        node_dim:          int   = 64,
        gat_hidden:        int   = 64,
        gat_heads:         int   = 4,
        gru_hidden:        int   = 512,
        gru_layers:        int   = 2,
        n_classes:         int   = 4,
        dropout:           float = 0.1,
        silent_pos_weight: float | None = None,
    ):
        super().__init__()

        # Feature extractors
        self.facial_ext  = FacialExtractor()
        self.body_ext    = BodyExtractor(out_dim=102)
        self.audio_ext   = AudioExtractor(in_dim=65, out_dim=128, dropout=dropout)
        self.physio_ext  = PhysioExtractor(n_signals=3, out_dim=64, dropout=dropout)

        # Node projection: align each modality to node_dim
        self.proj = nn.ModuleDict({
            "facial":  nn.Linear(17,  node_dim),
            "body":    nn.Linear(102, node_dim),
            "audio":   nn.Linear(128, node_dim),
            "physio":  nn.Linear(64,  node_dim),
        })

        # GAT
        self.gat = GraphAttentionNetwork(
            node_dim=node_dim,
            hidden_dim=gat_hidden,
            n_heads=gat_heads,
            n_nodes=4,
            dropout=dropout,
        )

        # BiGRU
        self.gru = TemporalModel(
            input_dim=gat_hidden,
            hidden_dim=gru_hidden,
            n_layers=gru_layers,
            dropout=dropout,
        )

        # Multi-task heads
        self.heads = MultiTaskHead(
            in_dim=self.gru.out_dim,
            n_classes=n_classes,
            dropout=dropout,
            silent_pos_weight=silent_pos_weight,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode_nodes(
        self,
        facial:  torch.Tensor,   # (B, T, 17)
        body:    torch.Tensor,   # (B, T, 51)
        audio:   torch.Tensor,   # (B, T, 65)
        physio:  torch.Tensor,   # (B, T, 3, 250)
    ) -> torch.Tensor:
        """
        Extract and project features to unified node embeddings.

        Returns : (B, T, 4, node_dim)
        """
        f = F.relu(self.proj["facial"](self.facial_ext(facial)))       # (B,T,nd)
        b = F.relu(self.proj["body"](self.body_ext(body)))
        a = F.relu(self.proj["audio"](self.audio_ext(audio)))
        p = F.relu(self.proj["physio"](self.physio_ext(physio)))

        return torch.stack([f, b, a, p], dim=2)   # (B, T, 4, node_dim)

    def forward(
        self,
        facial:  torch.Tensor,
        body:    torch.Tensor,
        audio:   torch.Tensor,
        physio:  torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns
        -------
        dict with keys: continuous, class_logits, silent_logit, attentions
        """
        B, T, _, _ = physio.shape   # use physio shape for B, T

        # 1. Node embeddings per timestep
        nodes = self.encode_nodes(facial, body, audio, physio)  # (B, T, 4, nd)

        # 2. GAT per timestep (batch over B*T)
        nodes_flat = nodes.view(B * T, 4, -1)
        z_flat, attns = self.gat(nodes_flat)      # (B*T, gat_hidden), [...]
        z_seq = z_flat.view(B, T, -1)             # (B, T, gat_hidden)

        # 3. BiGRU
        h = self.gru(z_seq, lengths)              # (B, 2*gru_hidden)

        # 4. Task heads
        preds = self.heads(h)
        preds["attentions"] = attns

        return preds

    def compute_loss(self, preds, y_cont, y_class, y_silent):
        return self.heads.compute_loss(preds, y_cont, y_class, y_silent)


__all__ = ["GraphConPain", "ContrastivePretrain", "GraphAugmentation"]
