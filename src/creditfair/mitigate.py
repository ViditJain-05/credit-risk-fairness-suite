"""Bias mitigation techniques and fairness-accuracy trade-off quantification.

Implements and compares 3 distinct families of algorithmic fairness interventions
against the unmitigated baseline across both `SEX` and `AGE_GROUP`:
1. Pre-processing: Sample Reweighing (Kamiran & Calders, 2012) — re-weights training
   examples so sensitive group membership S and target label Y are statistically independent:
       w(s, y) = [P(S = s) * P(Y = y)] / P(S = s, Y = y)
2. In-processing: `fairlearn.reductions.ExponentiatedGradient` with
   `DemographicParity` and `EqualizedOdds` constraints.
3. Post-processing: `fairlearn.postprocessing.ThresholdOptimizer` with
   `demographic_parity` and `equalized_odds` constraints fitted on the validation set.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from creditfair import config
from creditfair.data import DatasetSplits, load_saved_splits
from creditfair.fairness import audit_single_attribute
from creditfair.train import build_xgboost_model


class _Float64ProbaWrapper(BaseEstimator, ClassifierMixin):
    """Wraps a fitted classifier and ensures predict_proba returns float64 arrays."""

    def __init__(self, base_est: Any = None) -> None:
        self.base_est = base_est
        self.classes_ = np.array([0, 1])

    def fit(self, X: Any, y: Any) -> "_Float64ProbaWrapper":
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.base_est.predict_proba(X), dtype=np.float64)


def compute_reweighing_weights(
    y: pd.Series | np.ndarray, sensitive: pd.Series | np.ndarray
) -> np.ndarray:
    """Compute Kamiran & Calders (2012) sample weights on the training set.

    Formula:
        w(s, y) = (P(S = s) * P(Y = y)) / P(S = s, Y = y)
    """
    y_ser = pd.Series(y).reset_index(drop=True)
    s_ser = pd.Series(sensitive).astype(str).reset_index(drop=True)
    n_total = float(len(y_ser))

    p_s = s_ser.value_counts() / n_total
    p_y = y_ser.value_counts() / n_total
    joint_counts = pd.crosstab(s_ser, y_ser) / n_total

    weights = np.ones(len(y_ser), dtype=float)
    for i in range(len(y_ser)):
        s_val = s_ser.iloc[i]
        y_val = y_ser.iloc[i]
        p_joint = float(joint_counts.loc[s_val, y_val])
        if p_joint > 0:
            weights[i] = float((p_s[s_val] * p_y[y_val]) / p_joint)
    return weights


def _summarize_method_performance(
    method_name: str,
    paradigm: str,
    requires_sensitive_at_inference: bool,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    s_test: pd.Series,
    attr_name: str,
) -> Dict[str, Any]:
    """Compute unified performance + fairness metrics for the trade-off comparison table."""
    y_t = np.asarray(y_test, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    s_ser = pd.Series(s_test).astype(str).reset_index(drop=True)
    y_approve = (y_p == 0).astype(int)

    acc = float(accuracy_score(y_t, y_p))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    auc = float(roc_auc_score(y_t, y_prob)) if y_prob is not None else None

    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    expected_cost = float(
        (config.DEFAULT_COST_FN * fn + config.DEFAULT_COST_FP * fp) / max(len(y_t), 1)
    )

    dp_diff = float(demographic_parity_difference(y_t, y_p, sensitive_features=s_ser))
    eo_diff = float(equal_opportunity_difference(y_t, y_p, sensitive_features=s_ser))
    eod_diff = float(equalized_odds_difference(y_t, y_p, sensitive_features=s_ser))
    di_approval = float(demographic_parity_ratio(y_t, y_approve, sensitive_features=s_ser))

    group_audit = audit_single_attribute(
        y_true=y_t, y_pred=y_p, sensitive_series=s_ser, attr_name=attr_name, n_boot=100
    )

    return {
        "method": method_name,
        "paradigm": paradigm,
        "requires_sensitive_at_inference": requires_sensitive_at_inference,
        "roc_auc": round(auc, 4) if auc is not None else None,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "selection_rate_denial": round(float(y_p.mean()), 4),
        "approval_rate": round(float(y_approve.mean()), 4),
        "expected_cost_per_applicant": round(expected_cost, 4),
        "demographic_parity_difference": round(dp_diff, 4),
        "equal_opportunity_difference": round(eo_diff, 4),
        "equalized_odds_difference": round(eod_diff, 4),
        "disparate_impact_approval_ratio": round(di_approval, 4),
        "passes_four_fifths_rule": bool(di_approval >= config.FOUR_FIFTHS_THRESHOLD),
        "by_group": group_audit["by_group"],
    }


def run_mitigation_for_attribute(
    splits: DatasetSplits,
    model_bundle: Dict[str, Any],
    attr_name: str = "SEX",
) -> list[Dict[str, Any]]:
    """Run all mitigation strategies for a given protected attribute (`SEX` or `AGE_GROUP`)."""
    X_train, y_train, S_train = splits.X_train, splits.y_train, splits.S_train
    X_val, y_val, S_val = splits.X_val, splits.y_val, splits.S_val
    X_test, y_test, S_test = splits.X_test, splits.y_test, splits.S_test

    primary_model = model_bundle["primary_model"]
    xgb_base = model_bundle["xgb_base"]
    threshold = float(model_bundle["optimal_threshold"])

    s_train_col = S_train[attr_name].astype(str)
    s_val_col = S_val[attr_name].astype(str)
    s_test_col = S_test[attr_name].astype(str)

    results: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 0. Unmitigated Baseline (Calibrated XGBoost at cost-optimal threshold)
    # ------------------------------------------------------------------
    base_probs = primary_model.predict_proba(X_test)[:, 1]
    base_preds = (base_probs >= threshold).astype(int)
    results.append(
        _summarize_method_performance(
            method_name="Unmitigated Baseline (XGBoost)",
            paradigm="Baseline",
            requires_sensitive_at_inference=False,
            y_test=y_test.to_numpy(),
            y_pred=base_preds,
            y_prob=base_probs,
            s_test=s_test_col,
            attr_name=attr_name,
        )
    )

    # ------------------------------------------------------------------
    # 1. Pre-Processing: Sample Reweighing (Kamiran & Calders, 2012)
    # ------------------------------------------------------------------
    sample_weights = compute_reweighing_weights(y_train, s_train_col)
    xgb_reweighed = build_xgboost_model()
    xgb_reweighed.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    rw_probs = xgb_reweighed.predict_proba(X_test)[:, 1]
    rw_preds = (rw_probs >= threshold).astype(int)
    results.append(
        _summarize_method_performance(
            method_name="Pre-Processing: Sample Reweighing",
            paradigm="Pre-processing",
            requires_sensitive_at_inference=False,
            y_test=y_test.to_numpy(),
            y_pred=rw_preds,
            y_prob=rw_probs,
            s_test=s_test_col,
            attr_name=attr_name,
        )
    )

    # ------------------------------------------------------------------
    # 2A. In-Processing: ExponentiatedGradient (Demographic Parity)
    # ------------------------------------------------------------------
    # Use scaled features with a fast regularized estimator inside ExponentiatedGradient
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    eg_dp = ExponentiatedGradient(
        estimator=LogisticRegression(C=1.0, max_iter=500, random_state=config.RANDOM_SEED),
        constraints=DemographicParity(),
        eps=0.01,
        max_iter=30,
    )
    eg_dp.fit(X_tr_sc, y_train.to_numpy(), sensitive_features=s_train_col.to_numpy())
    np.random.seed(config.RANDOM_SEED)
    eg_dp_preds = eg_dp.predict(X_te_sc, random_state=config.RANDOM_SEED)
    eg_dp_pmf = eg_dp._pmf_predict(X_te_sc)[:, 1]
    results.append(
        _summarize_method_performance(
            method_name="In-Processing: ExponentiatedGradient (Demographic Parity)",
            paradigm="In-processing",
            requires_sensitive_at_inference=False,
            y_test=y_test.to_numpy(),
            y_pred=eg_dp_preds,
            y_prob=eg_dp_pmf,
            s_test=s_test_col,
            attr_name=attr_name,
        )
    )

    # ------------------------------------------------------------------
    # 2B. In-Processing: ExponentiatedGradient (Equalized Odds)
    # ------------------------------------------------------------------
    eg_eo = ExponentiatedGradient(
        estimator=LogisticRegression(C=1.0, max_iter=500, random_state=config.RANDOM_SEED),
        constraints=EqualizedOdds(),
        eps=0.01,
        max_iter=30,
    )
    eg_eo.fit(X_tr_sc, y_train.to_numpy(), sensitive_features=s_train_col.to_numpy())
    eg_eo_preds = eg_eo.predict(X_te_sc, random_state=config.RANDOM_SEED)
    eg_eo_pmf = eg_eo._pmf_predict(X_te_sc)[:, 1]
    results.append(
        _summarize_method_performance(
            method_name="In-Processing: ExponentiatedGradient (Equalized Odds)",
            paradigm="In-processing",
            requires_sensitive_at_inference=False,
            y_test=y_test.to_numpy(),
            y_pred=eg_eo_preds,
            y_prob=eg_eo_pmf,
            s_test=s_test_col,
            attr_name=attr_name,
        )
    )

    # ------------------------------------------------------------------
    # 3A. Post-Processing: ThresholdOptimizer (Demographic Parity)
    # Fitted on Validation set (prefit=True) so no training overfitting occurs.
    # Note: Cast estimator probabilities to float64 to avoid pandas 3.0 float32 LossySetitemError.
    # ------------------------------------------------------------------
    f64_estimator = _Float64ProbaWrapper(primary_model)

    to_dp = ThresholdOptimizer(
        estimator=f64_estimator,
        constraints="demographic_parity",
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    to_dp.fit(X_val, y_val.to_numpy(), sensitive_features=s_val_col.to_numpy())
    to_dp_preds = to_dp.predict(
        X_test, sensitive_features=s_test_col.to_numpy(), random_state=config.RANDOM_SEED
    )
    results.append(
        _summarize_method_performance(
            method_name="Post-Processing: ThresholdOptimizer (Demographic Parity)",
            paradigm="Post-processing",
            requires_sensitive_at_inference=True,
            y_test=y_test.to_numpy(),
            y_pred=to_dp_preds,
            y_prob=None,  # Randomized binary threshold post-processor
            s_test=s_test_col,
            attr_name=attr_name,
        )
    )

    # ------------------------------------------------------------------
    # 3B. Post-Processing: ThresholdOptimizer (Equalized Odds)
    # ------------------------------------------------------------------
    to_eo = ThresholdOptimizer(
        estimator=f64_estimator,
        constraints="equalized_odds",
        objective="accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    to_eo.fit(X_val, y_val.to_numpy(), sensitive_features=s_val_col.to_numpy())
    to_eo_preds = to_eo.predict(
        X_test, sensitive_features=s_test_col.to_numpy(), random_state=config.RANDOM_SEED
    )
    results.append(
        _summarize_method_performance(
            method_name="Post-Processing: ThresholdOptimizer (Equalized Odds)",
            paradigm="Post-processing",
            requires_sensitive_at_inference=True,
            y_test=y_test.to_numpy(),
            y_pred=to_eo_preds,
            y_prob=None,
            s_test=s_test_col,
            attr_name=attr_name,
        )
    )

    return results


def run_mitigation_pipeline(
    splits: DatasetSplits | None = None,
    model_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run Phase 5 bias mitigation experiments across SEX and AGE_GROUP and save artifacts."""
    if splits is None:
        splits = load_saved_splits()
    if model_bundle is None:
        model_bundle = joblib.load(config.get_artifact_path("model.joblib"))

    sex_comparison = run_mitigation_for_attribute(splits, model_bundle, attr_name="SEX")
    age_comparison = run_mitigation_for_attribute(splits, model_bundle, attr_name="AGE_GROUP")

    payload = {
        "comparisons": {
            "SEX": sex_comparison,
            "AGE_GROUP": age_comparison,
        },
        "regulatory_and_legal_discussion": {
            "disparate_treatment_vs_disparate_impact": (
                "Post-processing via ThresholdOptimizer achieves the lowest Demographic Parity "
                "and Equalized Odds gaps, but it requires passing the applicant's protected "
                "attribute (SEX or AGE_GROUP) at prediction time to apply group-specific cutoffs. "
                "Under fair-lending frameworks such as the US Equal Credit Opportunity Act (ECOA) "
                "and EU non-discrimination directives, explicitly applying different credit score "
                "cutoffs based on sex or age constitutes Disparate Treatment, which is generally "
                "prohibited even when motivated by reducing Disparate Impact. Therefore, in regulated "
                "lending, Pre-processing (Sample Reweighing) and In-processing (ExponentiatedGradient) "
                "are strongly preferred because they do not use protected attributes at inference time."
            )
        },
    }

    with open(config.get_artifact_path("fairness_after.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
