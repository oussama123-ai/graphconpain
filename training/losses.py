"""
training/losses.py
-------------------
Loss functions used during training:
  - InfoNCE for contrastive pretraining
  - Uncertainty-weighted multi-task loss (re-exported from multitask_head)
  - Focal loss variant for class imbalance
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.contrastive import info_nce_loss


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification with class imbalance.
    FL(pt) = -αt(1 - pt)^γ log(pt)
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, log_probs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        log_probs : (B, C) — log-softmax output
        targets   : (B,)  — integer class labels
        """
        probs = log_probs.exp()
        pt    = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss  = -(1.0 - pt) ** self.gamma * log_probs.gather(
            1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            loss = loss * self.alpha

        return loss.mean() if self.reduction == "mean" else loss.sum()


__all__ = ["info_nce_loss", "FocalLoss"]
