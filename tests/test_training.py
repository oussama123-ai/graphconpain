"""
tests/test_training.py
-----------------------
Integration tests for training loop components.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from models import GraphConPain
from models.contrastive import ContrastivePretrain
from training.losses import FocalLoss


class TestFocalLoss:
    def test_forward(self):
        loss_fn = FocalLoss(gamma=2.0)
        log_probs = torch.log_softmax(torch.randn(8, 4), dim=-1)
        targets   = torch.randint(0, 4, (8,))
        loss = loss_fn(log_probs, targets)
        assert loss.item() >= 0

    def test_easy_samples_downweighted(self):
        """Easy predictions (high confidence correct) → lower focal loss."""
        loss_fn = FocalLoss(gamma=2.0)
        # Confident correct prediction
        easy_lp  = torch.zeros(2, 4)
        easy_lp[:, 0] = 10.0
        easy_lp  = torch.log_softmax(easy_lp, dim=-1)
        easy_tgt = torch.zeros(2, dtype=torch.long)

        # Hard prediction (uniform)
        hard_lp  = torch.log_softmax(torch.zeros(2, 4), dim=-1)
        hard_tgt = torch.zeros(2, dtype=torch.long)

        loss_easy = FocalLoss(gamma=2.0)(easy_lp, easy_tgt)
        loss_hard = FocalLoss(gamma=2.0)(hard_lp, hard_tgt)
        assert loss_easy < loss_hard


class TestContrastivePretrain:
    def test_loss_shape(self):
        model    = GraphConPain(node_dim=32, gat_hidden=32, gru_hidden=64)
        contrast = ContrastivePretrain(
            backbone=model.gat, embed_dim=32, temperature=0.5
        )
        H   = torch.randn(4, 4, 32)
        adj = torch.ones(4, 4) - torch.eye(4)
        loss = contrast(H, adj)
        assert loss.item() >= 0

    def test_gradient_flows(self):
        model    = GraphConPain(node_dim=32, gat_hidden=32, gru_hidden=64)
        contrast = ContrastivePretrain(backbone=model.gat, embed_dim=32)
        H        = torch.randn(4, 4, 32)
        adj      = torch.ones(4, 4) - torch.eye(4)
        loss     = contrast(H, adj)
        loss.backward()
        for p in contrast.parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any()


class TestModelPersistence:
    def test_save_load(self, tmp_path):
        model    = GraphConPain(node_dim=32, gat_hidden=32, gru_hidden=64)
        ckpt_path = tmp_path / "model.pth"
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        model2 = GraphConPain(node_dim=32, gat_hidden=32, gru_hidden=64)
        ckpt   = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(ckpt["model_state_dict"])

        # Verify weights are identical
        for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                       model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"
