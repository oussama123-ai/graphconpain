"""
utils/visualization.py
-----------------------
Plotting utilities for training curves, confusion matrices,
ROC curves, and cross-validation results.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, confusion_matrix


PAIN_LABELS = ["None\n(0-2)", "Mild\n(3-4)", "Moderate\n(5-7)", "Severe\n(8-10)"]


def plot_training_curves(
    train_losses: list, val_losses: list,
    train_accs:   list, val_accs:   list,
    save_path: str,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "#2196F3", label="Train", lw=2)
    ax1.plot(epochs, val_losses,   "#E53935", label="Val",   lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in train_accs], "#2196F3", label="Train", lw=2)
    ax2.plot(epochs, [a * 100 for a in val_accs],   "#E53935", label="Val",   lw=2)
    ax2.axhline(88.5, color="gray", ls="--", lw=1, label="Target 88.5%")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    save_path: str, title: str = "Confusion Matrix",
):
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm_pct[i,j]:.1f}%", ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black", fontsize=9)

    ax.set_xticks(range(4)); ax.set_xticklabels(PAIN_LABELS, fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels(PAIN_LABELS, fontsize=8)
    ax.set_xlabel("Predicted Pain Level"); ax.set_ylabel("True Pain Level")
    ax.set_title(f"{title}\n(Overall Acc: {(y_true == y_pred).mean()*100:.1f}%)")
    plt.colorbar(im, ax=ax, label="% within true class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(
    y_true:  np.ndarray,     # (N,) integer class labels
    y_probs: np.ndarray,     # (N, C) class probabilities
    y_true_silent:  np.ndarray,
    y_prob_silent:  np.ndarray,
    save_path: str,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#E53935", "#FB8C00", "#43A047", "#1E88E5"]
    for c, color in zip(range(4), colors):
        binary = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(binary, y_probs[:, c])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, lw=2,
                 label=f"{PAIN_LABELS[c].strip()} (AUC={roc_auc:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC — Multiclass Pain Classification (One-vs-Rest)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    fpr_s, tpr_s, _ = roc_curve(y_true_silent, y_prob_silent)
    auc_s = auc(fpr_s, tpr_s)
    ax2.plot(fpr_s, tpr_s, "#9C27B0", lw=2.5,
             label=f"Silent Pain (AUC={auc_s:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("Sensitivity")
    ax2.set_title("ROC — Silent Pain Detection")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cv_fold_results(
    fold_accs:     list,
    fold_f1s:      list,
    fold_silent:   list,
    fold_mses:     list,
    save_path:     str,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    folds = range(1, len(fold_accs) + 1)
    mean_acc = np.mean(fold_accs)
    mean_f1  = np.mean(fold_f1s)
    mean_s   = np.mean(fold_silent)
    mean_mse = np.mean(fold_mses)

    for ax, vals, mean, label, color in zip(
        axes.flatten(),
        [fold_accs, fold_f1s, fold_silent, fold_mses],
        [mean_acc, mean_f1, mean_s, mean_mse],
        ["Accuracy (%)", "F1-Score (×100)", "Silent Recall (%)", "MSE"],
        ["#2196F3", "#43A047", "#E53935", "#FB8C00"],
    ):
        scale = 100 if "%" in label or "F1" in label else 1
        ax.bar(folds, [v * scale for v in vals], color=color, alpha=0.8, width=0.6)
        ax.axhline(mean * scale, color="navy", ls="--", lw=1.5,
                   label=f"Mean = {mean*scale:.2f}")
        ax.set_xlabel("Fold"); ax.set_ylabel(label)
        ax.set_title(f"5-Fold CV — {label}"); ax.legend(); ax.grid(alpha=0.2)

    plt.suptitle("Cross-Validation Performance (GraphConPain)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
