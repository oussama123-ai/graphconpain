"""
models/multitask_head.py
-------------------------
Multi-task prediction heads for GraphConPain:
  1. ContinuousScoringHead  — regression, pain intensity [0, 10]
  2. PainClassificationHead — 4-class softmax (none/mild/moderate/severe)
  3. SilentPainHead         — binary classifier for cry-absent pain

Also implements the uncertainty-weighted combined loss (Kendall et al. 2018,
Eq. 17 in paper):
    L = (1/2σ1²)·L_cont + (1/σ2²)·L_class + (1/σ3²)·L_silent + log(σ1·σ2·σ3)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual heads
# ---------------------------------------------------------------------------

class ContinuousScoringHead(nn.Module):
    """Linear regression → scalar pain score in [0, 10]."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_dim) → (B,)"""
        return self.fc(x).squeeze(-1)


class PainClassificationHead(nn.Module):
    """
    4-class pain level classifier:
      0 = None (0–2), 1 = Mild (3–4), 2 = Moderate (5–7), 3 = Severe (8–10)
    """

    def __init__(self, in_dim: int, hidden: list[int] = None,
                 n_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or [512, 256]
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_dim) → (B, 4)  log-softmax"""
        return F.log_softmax(self.net(x), dim=-1)


class SilentPainHead(nn.Module):
    """
    Binary classifier for silent (cry-absent) pain episodes.
    Returns logit (before sigmoid); use BCEWithLogitsLoss.
    """

    def __init__(self, in_dim: int, hidden: list[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden = hidden or [512, 256]
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, in_dim) → (B,)  logit"""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Combined multi-task head with uncertainty weighting
# ---------------------------------------------------------------------------

class MultiTaskHead(nn.Module):
    """
    Combines all three task heads with learnable homoscedastic uncertainty
    weights (σ1, σ2, σ3).

    The combined loss (Eq. 17):
        L = (1/2σ1²)·MSE + (1/σ2²)·CE + (1/σ3²)·BCE + log(σ1·σ2·σ3)
    """

    def __init__(
        self,
        in_dim:         int,
        n_classes:      int   = 4,
        head_hidden:    list  = None,
        dropout:        float = 0.1,
        silent_pos_weight: float | None = None,
    ):
        super().__init__()
        head_hidden = head_hidden or [512, 256]

        self.continuous  = ContinuousScoringHead(in_dim)
        self.classifier  = PainClassificationHead(in_dim, head_hidden,
                                                  n_classes, dropout)
        self.silent_head = SilentPainHead(in_dim, head_hidden, dropout)

        # Learnable log(σ²) for each task — initialized to 0 (σ=1)
        self.log_sigma2 = nn.Parameter(torch.zeros(3))

        # Optional class weight for silent pain imbalance
        self._silent_pos_weight = (
            torch.tensor([silent_pos_weight]) if silent_pos_weight else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "continuous":  self.continuous(x),
            "class_logits": self.classifier(x),
            "silent_logit": self.silent_head(x),
        }

    def compute_loss(
        self,
        preds:       dict[str, torch.Tensor],
        y_cont:      torch.Tensor,   # (B,) float
        y_class:     torch.Tensor,   # (B,) long  0-3
        y_silent:    torch.Tensor,   # (B,) float 0/1
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute uncertainty-weighted multi-task loss.

        Returns
        -------
        total_loss : scalar
        losses     : dict with individual task losses
        """
        sigma2 = torch.exp(self.log_sigma2)   # (3,) positive via exp

        # Task 1: continuous MSE
        L_cont = F.mse_loss(preds["continuous"], y_cont)

        # Task 2: classification cross-entropy
        L_class = F.nll_loss(preds["class_logits"], y_class)

        # Task 3: silent pain BCE (with optional pos weight)
        pw = self._silent_pos_weight
        if pw is not None:
            pw = pw.to(y_silent.device)
        L_silent = F.binary_cross_entropy_with_logits(
            preds["silent_logit"], y_silent, pos_weight=pw
        )

        # Uncertainty-weighted sum (Kendall et al.)
        # L = (1/2σ1²)·L1 + (1/σ2²)·L2 + (1/σ3²)·L3 + log(σ1·σ2·σ3)
        total = (
            L_cont   / (2.0 * sigma2[0]) +
            L_class  /        sigma2[1]  +
            L_silent /        sigma2[2]  +
            0.5 * self.log_sigma2.sum()   # log(σ1·σ2·σ3) = 0.5·Σ log(σ²)
        )

        return total, {
            "loss_cont":   L_cont.item(),
            "loss_class":  L_class.item(),
            "loss_silent": L_silent.item(),
            "sigma_cont":  sigma2[0].item() ** 0.5,
            "sigma_class": sigma2[1].item() ** 0.5,
            "sigma_silent":sigma2[2].item() ** 0.5,
        }
