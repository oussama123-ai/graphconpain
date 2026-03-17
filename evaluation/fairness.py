#!/usr/bin/env python3
"""
evaluation/fairness.py
-----------------------
Demographic stratification and fairness analysis (Section 5.3).

Reproduces Table 8: performance by skin tone, gestational age, sex.
Computes: demographic parity ratio (0.97), equalized odds ratio (0.94).

Usage
-----
    python evaluation/fairness.py \
        --checkpoint checkpoints/finetuned_full.pth \
        --data_dir data/ \
        --stratify skin_tone,gestational_age,sex \
        --output results/fairness_metrics.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import kruskal, mannwhitneyu

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import GraphConPain
from utils.data_loader import NeonatalPainDataset
from evaluation.metrics import (
    compute_classification_metrics, compute_silent_pain_metrics,
    demographic_parity_ratio, equalized_odds_ratio,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/finetuned_full.pth")
    p.add_argument("--data_dir",   type=str, default="data/")
    p.add_argument("--stratify",   type=str, default="skin_tone,gestational_age,sex")
    p.add_argument("--output",     type=str, default="results/fairness_metrics.json")
    p.add_argument("--device",     type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Group definitions matching Table 8
# ---------------------------------------------------------------------------

def skin_tone_group(tone: int) -> str:
    if tone in (1, 2):  return "I-II (lightest)"
    if tone in (3, 4):  return "III-IV (medium)"
    return "V-VI (darkest)"


def ga_group(ga: int) -> str:
    if ga < 28:   return "<28w (extremely preterm)"
    if ga < 32:   return "28-32w (very preterm)"
    if ga < 37:   return "32-37w (moderate preterm)"
    return ">=37w (term)"


def sex_group(sex: str) -> str:
    return sex.upper() if sex else "Unknown"


GROUP_FUNS = {
    "skin_tone":      skin_tone_group,
    "gestational_age": ga_group,
    "sex":            sex_group,
}


# ---------------------------------------------------------------------------
# Collect predictions
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    records = []
    for batch in loader:
        facial  = batch["facial"].to(device)
        body    = batch["body"].to(device)
        audio   = batch["audio"].to(device)
        physio  = batch["physio"].to(device)

        preds   = model(facial, body, audio, physio)
        pred_class  = preds["class_logits"].argmax(-1).cpu().numpy()
        prob_silent = preds["silent_logit"].sigmoid().cpu().numpy()
        pred_silent = (prob_silent > 0.5).astype(int)

        B = facial.shape[0]
        for i in range(B):
            idx = batch.get("_idx", list(range(B)))[i] if "_idx" in batch else i
            records.append({
                "y_class":     int(batch["y_class"][i]),
                "pred_class":  int(pred_class[i]),
                "y_silent":    int(batch["y_silent"][i]),
                "pred_silent": int(pred_silent[i]),
                "prob_silent": float(prob_silent[i]),
                "skin_tone":   int(batch.get("skin_tone", [-1]*B)[i]),
                "ga_weeks":    int(batch.get("ga_weeks",  [-1]*B)[i]),
                "sex":         batch.get("sex", [""] * B)[i],
            })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Model
    model = GraphConPain()
    if Path(args.checkpoint).exists():
        ckpt  = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model = model.to(device).eval()

    # Data
    dataset = NeonatalPainDataset(args.data_dir, split="test")
    records = collect_predictions(model, dataset, device)

    stratify_keys = [s.strip() for s in args.stratify.split(",")]
    results = {}

    for strat_key in stratify_keys:
        if strat_key not in GROUP_FUNS:
            continue

        group_fn = GROUP_FUNS[strat_key]
        groups   = defaultdict(list)
        for r in records:
            g = group_fn(r.get(strat_key, -1))
            groups[g].append(r)

        group_metrics = {}
        print(f"\n=== Stratification: {strat_key} ===")
        print(f"{'Group':<25} {'N':>4} {'Acc':>7} {'F1':>6} {'SilentRecall':>14}")

        group_accs = {}
        for g_name, grp in sorted(groups.items()):
            yt_c  = np.array([r["y_class"]     for r in grp])
            yp_c  = np.array([r["pred_class"]  for r in grp])
            yt_s  = np.array([r["y_silent"]    for r in grp])
            yp_s  = np.array([r["pred_silent"] for r in grp])

            cls_m = compute_classification_metrics(yt_c, yp_c)
            sil_m = compute_silent_pain_metrics(yt_s, yp_s)
            m = {**cls_m, **{f"silent_{k}": v for k, v in sil_m.items()},
                 "n": len(grp)}
            group_metrics[g_name] = m
            group_accs[g_name]    = cls_m["accuracy"]

            print(f"  {g_name:<23} {len(grp):>4} "
                  f"{cls_m['accuracy']*100:>6.1f}% "
                  f"{cls_m['f1_macro']:>6.3f} "
                  f"{sil_m['sensitivity']*100:>13.1f}%")

        # Overall fairness metrics
        all_yt  = np.array([r["y_class"]    for r in records])
        all_yp  = np.array([r["pred_class"] for r in records])
        all_grp = np.array([GROUP_FUNS[strat_key](r.get(strat_key, -1))
                             for r in records])

        dp_ratio = demographic_parity_ratio(all_yt, all_yp, all_grp)
        eo_ratio = equalized_odds_ratio(all_yt, all_yp, all_grp)

        # Statistical test
        group_acc_arrays = [
            np.array([int(r["pred_class"] == r["y_class"]) for r in grp])
            for grp in groups.values() if len(grp) >= 5
        ]
        if len(group_acc_arrays) >= 3:
            stat, pval = kruskal(*group_acc_arrays)
            stat_test = {"name": "Kruskal-Wallis", "statistic": float(stat),
                         "p_value": float(pval)}
        elif len(group_acc_arrays) == 2:
            stat, pval = mannwhitneyu(*group_acc_arrays, alternative="two-sided")
            stat_test = {"name": "Mann-Whitney U", "statistic": float(stat),
                         "p_value": float(pval)}
        else:
            stat_test = {}

        results[strat_key] = {
            "groups":         group_metrics,
            "fairness": {
                "demographic_parity_ratio": float(dp_ratio),
                "equalized_odds_ratio":     float(eo_ratio),
            },
            "statistical_test": stat_test,
        }
        print(f"\n  Demographic parity ratio : {dp_ratio:.3f}  "
              f"{'✓ FAIR' if dp_ratio >= 0.90 else '✗ CHECK'}")
        print(f"  Equalized odds ratio     : {eo_ratio:.3f}")
        if stat_test:
            sig = "n.s." if stat_test["p_value"] > 0.05 else "*"
            print(f"  {stat_test['name']}: H/U = {stat_test['statistic']:.2f}, "
                  f"p = {stat_test['p_value']:.3f} ({sig})")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFairness results saved → {output}")


if __name__ == "__main__":
    main()
