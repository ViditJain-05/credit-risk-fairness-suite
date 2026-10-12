"""Credit-risk evaluation metrics, calibration curves, and cost-optimal thresholding.

Implements standard ML metrics alongside retail banking / credit risk benchmarks:
- ROC-AUC & PR-AUC (Average Precision)
- Gini Coefficient: 2 * ROC_AUC - 1
- Kolmogorov-Smirnov (KS) Statistic: max |TPR(t) - FPR(t)| across thresholds
- Brier Score & Reliability (Calibration) Curve
- Business Cost-Matrix Threshold Optimization (evaluated on Validation Set only)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from creditfair import config


def compute_ks_statistic(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float]:
    """Compute the Kolmogorov-Smirnov (KS) statistic and the threshold where max separation occurs.

    In credit risk modeling, KS measures the maximum vertical separation between the
    cumulative distribution of defaulters (TPR) and non-defaulters (FPR).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)
    diff = np.abs(tpr - fpr)
    max_idx = int(np.argmax(diff))
    ks_value = float(diff[max_idx])
    ks_threshold = float(thresholds[max_idx]) if max_idx < len(thresholds) else 0.5
    return ks_value, ks_threshold


def compute_gini(roc_auc: float) -> float:
    """Compute the Gini coefficient from ROC-AUC: Gini = 2 * AUC - 1."""
    return float(2.0 * roc_auc - 1.0)


def find_cost_optimal_threshold(
    y_val: np.ndarray,
    y_val_prob: np.ndarray,
    cost_fn: float = config.DEFAULT_COST_FN,
    cost_fp: float = config.DEFAULT_COST_FP,
) -> Dict[str, Any]:
    """Select the decision threshold on the VALIDATION set that minimizes business cost.

    Decision rule: Deny credit (predict 1 = default risk) if P(default) >= threshold.
    - False Negative (FN): Approving an applicant who defaults (cost = `cost_fn`, e.g., 5.0).
    - False Positive (FP): Denying a creditworthy applicant (cost = `cost_fp`, e.g., 1.0).
    """
    y_val_arr = np.asarray(y_val, dtype=int)
    y_prob_arr = np.asarray(y_val_prob, dtype=float)
    n_samples = len(y_val_arr)

    candidate_thresholds = np.linspace(0.05, 0.90, 171)
    costs = []

    for t in candidate_thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_arr, y_pred, labels=[0, 1]).ravel()
        avg_cost = float((cost_fn * fn + cost_fp * fp) / n_samples)
        costs.append(avg_cost)

    best_idx = int(np.argmin(costs))
    best_t = float(round(candidate_thresholds[best_idx], 4))

    # Also compute cost at default 0.50 threshold for comparison
    y_pred_50 = (y_prob_arr >= 0.50).astype(int)
    _, fp_50, fn_50, _ = confusion_matrix(y_val_arr, y_pred_50, labels=[0, 1]).ravel()
    cost_at_50 = float((cost_fn * fn_50 + cost_fp * fp_50) / n_samples)

    return {
        "optimal_threshold": best_t,
        "min_expected_cost_per_applicant": round(float(costs[best_idx]), 4),
        "cost_at_0_50_threshold": round(cost_at_50, 4),
        "cost_fn": float(cost_fn),
        "cost_fp": float(cost_fp),
        "threshold_grid": [round(float(x), 4) for x in candidate_thresholds[::2]],
        "cost_curve": [round(float(c), 4) for c in costs[::2]],
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute comprehensive credit-risk and classification metrics at a given threshold."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    gini = compute_gini(roc_auc)
    ks_stat, ks_thresh = compute_ks_statistic(y_true_arr, y_prob_arr)
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    tpr = float(tp / max(tp + fn, 1))
    fpr = float(fp / max(fp + tn, 1))

    prob_true, prob_pred = calibration_curve(
        y_true_arr, y_prob_arr, n_bins=n_bins, strategy="uniform"
    )

    # Downsample ROC curve to ~100 points for compact JSON storage
    fpr_curve, tpr_curve, _ = roc_curve(y_true_arr, y_prob_arr)
    step = max(1, len(fpr_curve) // 100)

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "gini": round(gini, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_threshold": round(ks_thresh, 4),
        "brier_score": round(brier, 4),
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 4),
        "precision": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
        "recall_tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "f1_score": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
        "selection_rate_denial": round(float(y_pred.mean()), 4),
        "approval_rate": round(float(1.0 - y_pred.mean()), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "calibration_curve": {
            "prob_true": [round(float(x), 4) for x in prob_true],
            "prob_pred": [round(float(x), 4) for x in prob_pred],
        },
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr_curve[::step]],
            "tpr": [round(float(x), 4) for x in tpr_curve[::step]],
        },
    }
