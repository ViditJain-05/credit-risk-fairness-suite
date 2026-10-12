"""Fairness auditing with Fairlearn MetricFrame and bootstrap confidence intervals.

Audits model predictions across protected attributes (`SEX`, `AGE_GROUP`, `MARRIAGE`,
and intersectional `SEX x AGE_GROUP`) using `fairlearn.metrics.MetricFrame`:
- Selection Rate (predicted denial rate: P(Y_hat = 1))
- Approval Rate (P(Y_hat = 0))
- Demographic Parity Difference (max - min selection rate across groups)
- Disparate Impact Ratio (min / max selection rate & approval rate; Four-Fifths Rule >= 0.80)
- Equal Opportunity Difference (TPR gap across groups)
- False Positive Rate (FPR) Gap (wrongful credit denial rate gap across groups)
- Equalized Odds Difference (max of TPR gap and FPR gap)
- Bootstrap 95% Confidence Intervals on group gaps to separate statistically meaningful
  disparities from small-sample noise.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score, precision_score

from creditfair import config
from creditfair.data import DatasetSplits, load_saved_splits


def _safe_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def _approval_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Credit approval rate: fraction of applicants predicted NOT to default (y_pred == 0)."""
    return float(np.mean(np.asarray(y_pred) == 0))


def compute_bootstrap_gap_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
    n_boot: int = 200,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """Compute 95% bootstrap confidence intervals for fairness summary metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    s_a = np.asarray(sensitive_attr)
    y_approve = (y_p == 0).astype(int)

    dp_diffs = []
    eo_diffs = []
    eod_diffs = []
    di_approval_ratios = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt_b, yp_b, ya_b, sa_b = y_t[idx], y_p[idx], y_approve[idx], s_a[idx]
        try:
            dp_diffs.append(float(demographic_parity_difference(yt_b, yp_b, sensitive_features=sa_b)))
            eo_diffs.append(float(equal_opportunity_difference(yt_b, yp_b, sensitive_features=sa_b)))
            eod_diffs.append(float(equalized_odds_difference(yt_b, yp_b, sensitive_features=sa_b)))
            di_approval_ratios.append(
                float(demographic_parity_ratio(yt_b, ya_b, sensitive_features=sa_b))
            )
        except Exception:
            continue

    def _ci_summary(values: list[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "ci_lower_95": 0.0, "ci_upper_95": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "ci_lower_95": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(arr, 97.5)), 4),
        }

    return {
        "demographic_parity_difference": _ci_summary(dp_diffs),
        "equal_opportunity_difference": _ci_summary(eo_diffs),
        "equalized_odds_difference": _ci_summary(eod_diffs),
        "disparate_impact_approval_ratio": _ci_summary(di_approval_ratios),
    }


def audit_single_attribute(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_series: pd.Series | np.ndarray,
    attr_name: str,
    n_boot: int = 200,
) -> Dict[str, Any]:
    """Run Fairlearn MetricFrame audit + bootstrap CIs for a single sensitive attribute."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    s_ser = pd.Series(sensitive_series).astype(str).reset_index(drop=True)

    metrics_dict = {
        "count": lambda yt, yp: len(yt),
        "base_default_rate": lambda yt, yp: float(np.mean(yt)),
        "accuracy": accuracy_score,
        "selection_rate_denial": selection_rate,
        "approval_rate": _approval_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
        "precision": _safe_precision,
    }

    mf = MetricFrame(
        metrics=metrics_dict,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=s_ser,
    )

    by_group_df = mf.by_group.copy()
    by_group_records: Dict[str, Dict[str, float | int]] = {}
    for grp_name, row in by_group_df.iterrows():
        by_group_records[str(grp_name)] = {
            "count": int(row["count"]),
            "base_default_rate": round(float(row["base_default_rate"]), 4),
            "accuracy": round(float(row["accuracy"]), 4),
            "selection_rate_denial": round(float(row["selection_rate_denial"]), 4),
            "approval_rate": round(float(row["approval_rate"]), 4),
            "tpr": round(float(row["tpr"]), 4),
            "fpr": round(float(row["fpr"]), 4),
            "precision": round(float(row["precision"]), 4),
        }

    # Summary disparities
    y_approve = (y_p == 0).astype(int)
    dp_diff = float(demographic_parity_difference(y_t, y_p, sensitive_features=s_ser))
    dp_denial_ratio = float(demographic_parity_ratio(y_t, y_p, sensitive_features=s_ser))
    di_approval_ratio = float(demographic_parity_ratio(y_t, y_approve, sensitive_features=s_ser))
    eo_diff = float(equal_opportunity_difference(y_t, y_p, sensitive_features=s_ser))
    eod_diff = float(equalized_odds_difference(y_t, y_p, sensitive_features=s_ser))
    fpr_gap = float(mf.difference(method="between_groups")["fpr"])

    boot_cis = compute_bootstrap_gap_cis(y_t, y_p, s_ser.to_numpy(), n_boot=n_boot)

    return {
        "attribute": attr_name,
        "overall": {
            k: (int(v) if k == "count" else round(float(v), 4))
            for k, v in mf.overall.to_dict().items()
        },
        "by_group": by_group_records,
        "summary_metrics": {
            "demographic_parity_difference": round(dp_diff, 4),
            "disparate_impact_denial_ratio": round(dp_denial_ratio, 4),
            "disparate_impact_approval_ratio": round(di_approval_ratio, 4),
            "passes_four_fifths_approval_rule": bool(di_approval_ratio >= config.FOUR_FIFTHS_THRESHOLD),
            "equal_opportunity_difference_tpr_gap": round(eo_diff, 4),
            "fpr_gap": round(fpr_gap, 4),
            "equalized_odds_difference": round(eod_diff, 4),
        },
        "bootstrap_95_ci": boot_cis,
    }


def run_fairness_audit(
    splits: DatasetSplits | None = None,
    model_bundle: Dict[str, Any] | None = None,
    n_boot: int = 200,
) -> Dict[str, Any]:
    """Run complete Phase 4 Fairlearn audit across SEX, AGE_GROUP, MARRIAGE, and SEX x AGE_GROUP."""
    if splits is None:
        splits = load_saved_splits()
    if model_bundle is None:
        model_bundle = joblib.load(config.get_artifact_path("model.joblib"))

    primary_model = model_bundle["primary_model"]
    threshold = float(model_bundle["optimal_threshold"])

    X_test, y_test, S_test = splits.X_test, splits.y_test, splits.S_test
    test_probs = primary_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    audits: Dict[str, Any] = {}
    for attr in ["SEX", "AGE_GROUP", "MARRIAGE", "SEX_x_AGE_GROUP"]:
        audits[attr] = audit_single_attribute(
            y_true=y_test,
            y_pred=test_preds,
            sensitive_series=S_test[attr],
            attr_name=attr,
            n_boot=n_boot,
        )

    payload = {
        "model_name": "XGBoost (Calibrated, Unmitigated Baseline)",
        "decision_threshold": round(threshold, 4),
        "test_set_size": int(len(y_test)),
        "audits": audits,
    }

    with open(config.get_artifact_path("fairness_before.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
