#!/usr/bin/env python3
"""
training/finetune.py
---------------------
Phase 2: Supervised multi-task fine-tuning with uncertainty-weighted loss.

Usage
-----
    python training/finetune.py \
        --config config/finetune.yaml \
        --pretrained_weights checkpoints/pretrained_ssl.pth \
        --data_dir data/ \
        --output_dir checkpoints/ \
        --epochs 100 \
        --batch_size 32 \
        --lr 5e-4
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
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import GraphConPain
from utils.data_loader import NeonatalPainDataset
from utils.augmentation import MultimodalAugmentation


def parse_args():
    p = argparse.ArgumentParser(description="GraphConPain Fine-Tuning")
    p.add_argument("--config",             type=str, default="config/finetune.yaml")
    p.add_argument("--pretrained_weights", type=str, default=None)
    p.add_argument("--data_dir",           type=str, default="data/")
    p.add_argument("--output_dir",         type=str, default="checkpoints/")
    p.add_argument("--epochs",             type=int, default=100)
    p.add_argument("--batch_size",         type=int, default=32)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--weight_decay",       type=float, default=1e-4)
    p.add_argument("--patience",           type=int, default=10)
    p.add_argument("--node_dim",           type=int, default=64)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--wandb",              action="store_true")
    p.add_argument("--num_workers",        type=int, default=4)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    correct, total = 0, 0
    silent_correct, silent_total = 0, 0

    with torch.no_grad():
        for batch in loader:
            facial  = batch["facial"].to(device)
            body    = batch["body"].to(device)
            audio   = batch["audio"].to(device)
            physio  = batch["physio"].to(device)
            y_cont  = batch["y_cont"].to(device)
            y_class = batch["y_class"].to(device)
            y_silent= batch["y_silent"].to(device)

            preds = model(facial, body, audio, physio)
            loss, _ = model.compute_loss(preds, y_cont, y_class, y_silent)

            total_loss += loss.item() * facial.shape[0]
            n          += facial.shape[0]

            # Classification accuracy
            pred_class = preds["class_logits"].argmax(dim=-1)
            correct   += (pred_class == y_class).sum().item()
            total     += y_class.shape[0]

            # Silent pain accuracy
            pred_silent = (preds["silent_logit"].sigmoid() > 0.5).long()
            silent_correct += (pred_silent == y_silent.long()).sum().item()
            silent_total   += y_silent.shape[0]

    return {
        "loss":          total_loss / n,
        "accuracy":      correct / total,
        "silent_recall": silent_correct / silent_total,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[Finetune] device={device}  epochs={args.epochs}  lr={args.lr}")

    # ------------------------------------------------------------------ #
    # Dataset  (70/15/15 split)
    # ------------------------------------------------------------------ #
    full_ds = NeonatalPainDataset(args.data_dir, split="train")
    n       = len(full_ds)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val
    train_ds, val_ds, _ = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    aug = MultimodalAugmentation()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    print(f"[Finetune] Train={len(train_ds)}  Val={len(val_ds)}  Test={n_test}")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = GraphConPain(node_dim=args.node_dim).to(device)

    if args.pretrained_weights and Path(args.pretrained_weights).exists():
        ckpt  = torch.load(args.pretrained_weights, map_location=device)
        state = ckpt.get("backbone_state", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded pretrained weights. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("  No pretrained weights. Training from scratch.")

    # ------------------------------------------------------------------ #
    # Optimizer & scheduler
    # ------------------------------------------------------------------ #
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    if args.wandb:
        import wandb
        wandb.init(project="graphconpain", name="finetune", config=vars(args))

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            facial  = batch["facial"].to(device)
            body    = batch["body"].to(device)
            audio   = batch["audio"].to(device)
            physio  = batch["physio"].to(device)
            y_cont  = batch["y_cont"].to(device)
            y_class = batch["y_class"].to(device)
            y_silent= batch["y_silent"].to(device)

            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                preds = model(facial, body, audio, physio)
                loss, task_losses = model.compute_loss(
                    preds, y_cont, y_class, y_silent
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={epoch_loss/len(train_loader):.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']*100:.2f}% | "
            f"silent_recall={val_metrics['silent_recall']*100:.1f}% | "
            f"{elapsed:.1f}s"
        )

        if args.wandb:
            wandb.log({
                "train/loss":      epoch_loss / len(train_loader),
                "val/loss":        val_metrics["loss"],
                "val/accuracy":    val_metrics["accuracy"],
                "val/silent_recall": val_metrics["silent_recall"],
            }, step=epoch)

        # Save best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            no_improve   = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
            }, output_dir / "finetuned_full.pth")
            print(f"  ✓ Best checkpoint (val_acc={best_val_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict()},
                output_dir / f"finetune_epoch{epoch:03d}.pth"
            )

    print(f"[Finetune] Complete. Best val accuracy: {best_val_acc*100:.2f}%")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
