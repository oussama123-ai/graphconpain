"""
tests/test_models.py
---------------------
Unit tests for model forward passes.  Run with: pytest tests/test_models.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from models import GraphConPain
from models.graph_attention import GraphAttentionNetwork, GATLayer
from models.temporal_model import TemporalModel
from models.multitask_head import MultiTaskHead, ContinuousScoringHead
from models.contrastive import ContrastivePretrain, info_nce_loss


B, T = 2, 10   # batch size, sequence length


@pytest.fixture
def model():
    return GraphConPain(node_dim=32, gat_hidden=32, gru_hidden=64)


def dummy_batch(B=2, T=10, device="cpu"):
    return {
        "facial":  torch.randn(B, T, 17),
        "body":    torch.randn(B, T, 51),
        "audio":   torch.randn(B, T, 65),
        "physio":  torch.randn(B, T, 3, 250),
        "y_cont":  torch.rand(B),
        "y_class": torch.randint(0, 4, (B,)),
        "y_silent":torch.randint(0, 2, (B,)).float(),
    }


# ---------------------------------------------------------------------------
# GAT tests
# ---------------------------------------------------------------------------

class TestGATLayer:
    def test_output_shape_concat(self):
        layer = GATLayer(in_dim=64, out_dim=32, n_heads=4, concat=True)
        H     = torch.randn(3, 4, 64)
        adj   = torch.ones(4, 4) - torch.eye(4)
        H_new, alpha = layer(H, adj)
        assert H_new.shape  == (3, 4, 128), f"Expected (3,4,128) got {H_new.shape}"
        assert alpha.shape  == (3, 4, 4, 4)

    def test_output_shape_average(self):
        layer = GATLayer(in_dim=64, out_dim=32, n_heads=4, concat=False)
        H     = torch.randn(3, 4, 64)
        adj   = torch.ones(4, 4) - torch.eye(4)
        H_new, _ = layer(H, adj)
        assert H_new.shape == (3, 4, 32)

    def test_attention_sums_to_one(self):
        layer = GATLayer(in_dim=32, out_dim=16, n_heads=2, concat=True)
        H     = torch.randn(2, 4, 32)
        adj   = torch.ones(4, 4) - torch.eye(4)
        _, alpha = layer(H, adj)
        # Each row should sum to 1 (softmax over neighbors including self)
        row_sums = alpha.sum(dim=-1)   # (B, H, N)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestGAT:
    def test_output_shape(self):
        gat = GraphAttentionNetwork(node_dim=32, hidden_dim=32, n_heads=4)
        H   = torch.randn(B * T, 4, 32)
        z, attns = gat(H)
        assert z.shape    == (B * T, 32)
        assert len(attns) == 2   # 2 layers


# ---------------------------------------------------------------------------
# Temporal model tests
# ---------------------------------------------------------------------------

class TestTemporalModel:
    def test_output_shape(self):
        gru = TemporalModel(input_dim=32, hidden_dim=64, n_layers=2)
        x   = torch.randn(B, T, 32)
        out = gru(x)
        assert out.shape == (B, 128)   # 2*64

    def test_sequence_output(self):
        gru = TemporalModel(input_dim=32, hidden_dim=64)
        x   = torch.randn(B, T, 32)
        seq = gru.forward_sequence(x)
        assert seq.shape == (B, T, 128)


# ---------------------------------------------------------------------------
# Multi-task head tests
# ---------------------------------------------------------------------------

class TestMultiTaskHead:
    def test_forward(self):
        head  = MultiTaskHead(in_dim=256, n_classes=4)
        x     = torch.randn(B, 256)
        preds = head(x)
        assert preds["continuous"].shape    == (B,)
        assert preds["class_logits"].shape  == (B, 4)
        assert preds["silent_logit"].shape  == (B,)

    def test_loss(self):
        head   = MultiTaskHead(in_dim=64, n_classes=4)
        x      = torch.randn(B, 64)
        preds  = head(x)
        y_cont = torch.rand(B)
        y_cls  = torch.randint(0, 4, (B,))
        y_sil  = torch.randint(0, 2, (B,)).float()
        loss, info = head.compute_loss(preds, y_cont, y_cls, y_sil)
        assert loss.item() >= 0
        assert "loss_cont" in info


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

class TestGraphConPain:
    def test_forward_pass(self, model):
        batch = dummy_batch()
        preds = model(**{k: v for k, v in batch.items()
                         if k in ["facial","body","audio","physio"]})
        assert "continuous"   in preds
        assert "class_logits" in preds
        assert "silent_logit" in preds
        assert "attentions"   in preds
        assert preds["continuous"].shape   == (B,)
        assert preds["class_logits"].shape == (B, 4)

    def test_loss_computation(self, model):
        batch  = dummy_batch()
        preds  = model(**{k: v for k, v in batch.items()
                          if k in ["facial","body","audio","physio"]})
        loss, info = model.compute_loss(
            preds, batch["y_cont"], batch["y_class"], batch["y_silent"]
        )
        assert loss.item() >= 0

    def test_gradient_flow(self, model):
        batch  = dummy_batch()
        preds  = model(**{k: v for k, v in batch.items()
                          if k in ["facial","body","audio","physio"]})
        loss, _ = model.compute_loss(
            preds, batch["y_cont"], batch["y_class"], batch["y_silent"]
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


# ---------------------------------------------------------------------------
# Contrastive loss test
# ---------------------------------------------------------------------------

class TestContrastiveLoss:
    def test_info_nce(self):
        B = 4
        z1 = torch.randn(B, 128); z1 = z1 / z1.norm(dim=-1, keepdim=True)
        z2 = torch.randn(B, 128); z2 = z2 / z2.norm(dim=-1, keepdim=True)
        loss = info_nce_loss(z1, z2, temperature=0.5)
        assert loss.item() >= 0

    def test_identical_views(self):
        B = 4
        z = torch.randn(B, 128); z = z / z.norm(dim=-1, keepdim=True)
        loss = info_nce_loss(z, z.clone(), temperature=0.5)
        # Identical views → very low loss
        assert loss.item() < 2.0
