"""
evaluation/metrics.py
----------------------
All evaluation metrics from the paper:
  accuracy, F1, MSE, MAE, Pearson, AUC-ROC,
  silent pain sensitivity/specificity/AUC,
  demographic parity, equalized odds.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, mean_squared_error, mean_absolute_error,
)
from scipy.stats import pearsonr


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """
    y_true, y_pred : (N,) integer class labels
    y_prob         : (N, C) predicted probabilities (optional)
    """
    results = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro",
                                            zero_division=0)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average="weighted",
                                            zero_division=0)),
    }
    if y_prob is not None:
        try:
            results["auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except Exception:
            results["auc_ovr"] = float("nan")
    return results


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    try:
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = float("nan")
    return {"mse": mse, "mae": mae, "pearson_r": float(r)}


def compute_silent_pain_metrics(
    y_true:   np.ndarray,   # binary 0/1
    y_pred:   np.ndarray,   # binary 0/1
    y_prob:   np.ndarray | None = None,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision   = tp / (tp + fp + 1e-8)
    results = {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision":   float(precision),
        "f1":          float(2 * precision * sensitivity /
                              (precision + sensitivity + 1e-8)),
    }
    if y_prob is not None:
        try:
            results["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            results["auc_roc"] = float("nan")
    return results


def compute_all_metrics(
    y_true_class:   np.ndarray,
    y_pred_class:   np.ndarray,
    y_prob_class:   np.ndarray | None,
    y_true_cont:    np.ndarray,
    y_pred_cont:    np.ndarray,
    y_true_silent:  np.ndarray,
    y_pred_silent:  np.ndarray,
    y_prob_silent:  np.ndarray | None = None,
) -> dict[str, float]:
    m = {}
    m.update(compute_classification_metrics(y_true_class, y_pred_class, y_prob_class))
    m.update({f"reg_{k}": v
              for k, v in compute_regression_metrics(y_true_cont, y_pred_cont).items()})
    m.update({f"silent_{k}": v
              for k, v in compute_silent_pain_metrics(y_true_silent, y_pred_silent,
                                                       y_prob_silent).items()})
    return m


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------

def demographic_parity_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    min_i(PPV_i) / max_i(PPV_i)   where PPV = positive predictive value.
    Ratio > 0.90 indicates fairness.
    """
    unique = np.unique(groups)
    ppvs = []
    for g in unique:
        mask   = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        if yp.sum() == 0:
            continue
        tp = ((yt == 1) & (yp == 1)).sum()
        ppvs.append(tp / yp.sum())
    if len(ppvs) < 2:
        return 1.0
    return float(min(ppvs) / (max(ppvs) + 1e-8))


def equalized_odds_ratio(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    groups:  np.ndarray,
) -> float:
    """
    min(TPR_i) / max(TPR_i) across demographic groups.
    """
    unique = np.unique(groups)
    tprs = []
    for g in unique:
        mask   = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        if tp + fn == 0:
            continue
        tprs.append(tp / (tp + fn))
    if len(tprs) < 2:
        return 1.0
    return float(min(tprs) / (max(tprs) + 1e-8))
