#!/usr/bin/env python3
"""
training/pretrain.py
---------------------
Phase 1: Graph-aware contrastive self-supervised pretraining (Algorithm 8-9).

Usage
-----
    python training/pretrain.py \
        --config config/pretrain.yaml \
        --data_dir data/ \
        --output_dir checkpoints/ \
        --epochs 50 \
        --batch_size 32 \
        --temperature 0.5 \
        --gpus 1
"""

from __future__ import annotations
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import GraphConPain
from models.contrastive import ContrastivePretrain
from utils.data_loader import UnlabeledMultimodalDataset
from utils.augmentation import MultimodalAugmentation


def parse_args():
    p = argparse.ArgumentParser(description="GraphConPain SSL Pretraining")
    p.add_argument("--config",      type=str,  default="config/pretrain.yaml")
    p.add_argument("--data_dir",    type=str,  default="data/")
    p.add_argument("--output_dir",  type=str,  default="checkpoints/")
    p.add_argument("--epochs",      type=int,  default=50)
    p.add_argument("--batch_size",  type=int,  default=32)
    p.add_argument("--lr",          type=float,default=1e-3)
    p.add_argument("--temperature", type=float,default=0.5)
    p.add_argument("--node_dim",    type=int,  default=64)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--num_workers", type=int,  default=4)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args   = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[Pretrain] device={device}  epochs={args.epochs}  tau={args.temperature}")

    # ------------------------------------------------------------------ #
    # Dataset (unlabeled multimodal recordings)
    # ------------------------------------------------------------------ #
    dataset = UnlabeledMultimodalDataset(args.data_dir)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[Pretrain] Dataset size: {len(dataset)}")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    backbone = GraphConPain(node_dim=args.node_dim).to(device)
    model    = ContrastivePretrain(
        backbone=backbone.gat,
        embed_dim=backbone.gat.out_dim,
        temperature=args.temperature,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    # Cosine LR with no warm-up (SSL phase)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Optional W&B
    if args.wandb:
        import wandb
        wandb.init(project="graphconpain", name="pretrain", config=vars(args))

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            H   = batch["node_features"].to(device)   # (B, N, D)
            adj = batch.get("adj", None)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                loss = model(H, adj)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(loader), 1)
        elapsed  = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s")

        if args.wandb:
            wandb.log({"pretrain/loss": avg_loss}, step=epoch)

        # Save checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "backbone_state":   backbone.state_dict(),
            "loss": avg_loss,
        }
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, output_dir / "pretrained_ssl.pth")
            print(f"  ✓ Best checkpoint saved (loss={best_loss:.4f})")

        if epoch % 10 == 0:
            torch.save(ckpt, output_dir / f"pretrain_epoch{epoch:03d}.pth")

    print(f"[Pretrain] Complete. Best loss: {best_loss:.4f}")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
