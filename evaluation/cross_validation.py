#!/usr/bin/env python3
"""
evaluation/cross_validation.py
--------------------------------
5-fold stratified cross-validation reproducing Table 6 main results.

Usage
-----
    python evaluation/cross_validation.py \
        --config config/finetune.yaml \
        --data_dir data/ \
        --folds 5 \
        --output results/cross_validation.csv
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import GraphConPain
from utils.data_loader import NeonatalPainDataset
from evaluation.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      type=str, default="config/finetune.yaml")
    p.add_argument("--checkpoint",  type=str, default="checkpoints/finetuned_full.pth")
    p.add_argument("--data_dir",    type=str, default="data/")
    p.add_argument("--folds",       type=int, default=5)
    p.add_argument("--output",      type=str, default="results/cross_validation.json")
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def evaluate_fold(model, loader, device) -> dict:
    model.eval()
    all_y_class, all_p_class, all_prob_class = [], [], []
    all_y_cont,  all_p_cont  = [], []
    all_y_silent, all_p_silent, all_prob_silent = [], [], []

    for batch in loader:
        facial  = batch["facial"].to(device)
        body    = batch["body"].to(device)
        audio   = batch["audio"].to(device)
        physio  = batch["physio"].to(device)

        preds = model(facial, body, audio, physio)

        all_y_class.extend(batch["y_class"].numpy())
        all_p_class.extend(preds["class_logits"].argmax(-1).cpu().numpy())
        all_prob_class.extend(preds["class_logits"].exp().cpu().numpy())

        all_y_cont.extend(batch["y_cont"].numpy())
        all_p_cont.extend(preds["continuous"].cpu().numpy())

        all_y_silent.extend(batch["y_silent"].numpy())
        prob_s = preds["silent_logit"].sigmoid().cpu().numpy()
        all_p_silent.extend((prob_s > 0.5).astype(int))
        all_prob_silent.extend(prob_s)

    return compute_all_metrics(
        np.array(all_y_class),  np.array(all_p_class),  np.array(all_prob_class),
        np.array(all_y_cont),   np.array(all_p_cont),
        np.array(all_y_silent), np.array(all_p_silent),  np.array(all_prob_silent),
    )


def main():
    args   = parse_args()
    device = torch.device(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = NeonatalPainDataset(args.data_dir)
    labels  = np.array([dataset.records[i]["y_class"] for i in range(len(dataset))])

    skf     = StratifiedKFold(n_splits=args.folds, shuffle=True,
                               random_state=args.seed)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(labels, labels)):
        print(f"\n=== Fold {fold+1}/{args.folds} ===")

        # Build model and load checkpoint
        model = GraphConPain()
        if Path(args.checkpoint).exists():
            ckpt  = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        model = model.to(device).eval()

        test_ds = Subset(dataset, test_idx.tolist())
        loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

        metrics = evaluate_fold(model, loader, device)
        fold_results.append(metrics)

        print(f"  Accuracy:       {metrics['accuracy']*100:.2f}%")
        print(f"  F1 (macro):     {metrics['f1_macro']:.3f}")
        print(f"  Silent Recall:  {metrics['silent_sensitivity']*100:.1f}%")
        print(f"  Cont. MSE:      {metrics['reg_mse']:.4f}")

    # Aggregate
    keys = fold_results[0].keys()
    summary = {
        k: {
            "mean": float(np.mean([r[k] for r in fold_results])),
            "std":  float(np.std( [r[k] for r in fold_results])),
        }
        for k in keys
    }
    summary["n_folds"] = args.folds

    with open(output, "w") as f:
        json.dump({"per_fold": fold_results, "summary": summary}, f, indent=2)

    print(f"\n=== {args.folds}-Fold CV Summary ===")
    print(f"  Accuracy:       {summary['accuracy']['mean']*100:.2f} ± {summary['accuracy']['std']*100:.2f}%")
    print(f"  F1 (macro):     {summary['f1_macro']['mean']:.3f} ± {summary['f1_macro']['std']:.3f}")
    print(f"  Silent Recall:  {summary.get('silent_sensitivity',{}).get('mean',0)*100:.1f}%")
    print(f"Results saved → {output}")


if __name__ == "__main__":
    main()
