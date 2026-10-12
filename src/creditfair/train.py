"""Model training, probability calibration, and baseline model comparison.

Trains:
1. Scaled Logistic Regression (`StandardScaler` + `LogisticRegression`)
2. HistGradientBoostingClassifier (scikit-learn native GBDT comparison)
3. XGBoost Classifier (`XGBClassifier`) with regularization & probability calibration

Crucial Anti-Leakage Rules Enforced:
- All scalers and model parameters are fit ONLY on `(X_train, y_train)`.
- Probability calibration (Isotonic / Sigmoid) and cost-optimal thresholding
  are fit/selected on `(X_val, y_val)`.
- The held-out test set `(X_test, y_test)` is evaluated ONCE at the end.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from creditfair import config
from creditfair.data import DatasetSplits, load_saved_splits
from creditfair.evaluate import evaluate_predictions, find_cost_optimal_threshold


class CalibratedXGBModel:
    """Wrapper pairing a trained XGBClassifier with validation-fitted Isotonic calibration.

    Exposes the standard scikit-learn estimator interface (`predict_proba`, `predict`,
    `classes_`) while preserving direct access to `.base_estimator` for fast
    `shap.TreeExplainer` extraction.
    """

    def __init__(self, base_estimator: XGBClassifier, calibrator: IsotonicRegression):
        self.base_estimator = base_estimator
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw_prob = self.base_estimator.predict_proba(X)[:, 1].astype(np.float64)
        cal_prob = np.clip(
            np.asarray(self.calibrator.predict(raw_prob), dtype=np.float64),
            1e-6,
            1.0 - 1e-6,
        )
        return np.column_stack([1.0 - cal_prob, cal_prob]).astype(np.float64)

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def build_logistic_baseline() -> Pipeline:
    """Build a leak-safe Pipeline with StandardScaler + L2 Logistic Regression."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_hist_gbdt_baseline() -> HistGradientBoostingClassifier:
    """Build a HistGradientBoostingClassifier for benchmark comparison."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=30,
        random_state=config.RANDOM_SEED,
    )


def build_xgboost_model() -> XGBClassifier:
    """Build a well-regularized XGBClassifier for credit default probability estimation."""
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
        eval_metric="logloss",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )


def train_and_evaluate_all(splits: DatasetSplits | None = None) -> Dict[str, Any]:
    """Train baseline models, calibrate XGBoost probabilities, select threshold on Val,
    and evaluate once on the Test set. Saves `model.joblib` and `metrics.json` to `artifacts/`.
    """
    if splits is None:
        splits = load_saved_splits()

    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    X_test, y_test = splits.X_test, splits.y_test

    # 1. Train Logistic Regression Baseline
    lr_model = build_logistic_baseline()
    lr_model.fit(X_train, y_train)

    # 2. Train HistGradientBoosting Baseline
    hgb_model = build_hist_gbdt_baseline()
    hgb_model.fit(X_train, y_train)

    # 3. Train XGBoost Model (Unweighted + Regularized for well-behaved probabilities)
    xgb_raw = build_xgboost_model()
    xgb_raw.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Also train a class-weighted XGBoost to demonstrate why scale_pos_weight distorts
    # raw probabilities without post-hoc calibration
    neg_pos_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb_weighted = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
        scale_pos_weight=neg_pos_ratio,
        eval_metric="logloss",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    xgb_weighted.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 4. Calibrate XGBoost probabilities on Validation Set using Isotonic Regression
    val_probs_uncal = xgb_weighted.predict_proba(X_val)[:, 1]
    iso_calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso_calibrator.fit(val_probs_uncal, y_val.to_numpy())

    calibrated_xgb = CalibratedXGBModel(
        base_estimator=xgb_raw, calibrator=IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    )
    # Fit calibrator on raw XGBoost validation probabilities
    val_probs_raw_xgb = xgb_raw.predict_proba(X_val)[:, 1]
    calibrated_xgb.calibrator.fit(val_probs_raw_xgb, y_val.to_numpy())

    # 5. Select Cost-Optimal Decision Threshold on VALIDATION Set
    val_probs_cal = calibrated_xgb.predict_proba(X_val)[:, 1]
    threshold_info = find_cost_optimal_threshold(
        y_val=y_val.to_numpy(),
        y_val_prob=val_probs_cal,
        cost_fn=config.DEFAULT_COST_FN,
        cost_fp=config.DEFAULT_COST_FP,
    )
    optimal_threshold = threshold_info["optimal_threshold"]

    # 6. Final One-Time Evaluation on Held-Out TEST Set
    lr_test_probs = lr_model.predict_proba(X_test)[:, 1]
    hgb_test_probs = hgb_model.predict_proba(X_test)[:, 1]
    xgb_weighted_test_probs = xgb_weighted.predict_proba(X_test)[:, 1]
    xgb_raw_test_probs = xgb_raw.predict_proba(X_test)[:, 1]
    xgb_cal_test_probs = calibrated_xgb.predict_proba(X_test)[:, 1]

    metrics_payload: Dict[str, Any] = {
        "primary_model": "XGBoost (Calibrated)",
        "optimal_threshold": optimal_threshold,
        "threshold_selection": threshold_info,
        "models_test_set": {
            "Logistic Regression (Scaled)": evaluate_predictions(
                y_test.to_numpy(), lr_test_probs, threshold=optimal_threshold
            ),
            "HistGradientBoosting": evaluate_predictions(
                y_test.to_numpy(), hgb_test_probs, threshold=optimal_threshold
            ),
            "XGBoost (Weighted, Uncalibrated)": evaluate_predictions(
                y_test.to_numpy(), xgb_weighted_test_probs, threshold=0.50
            ),
            "XGBoost (Raw Uncalibrated)": evaluate_predictions(
                y_test.to_numpy(), xgb_raw_test_probs, threshold=optimal_threshold
            ),
            "XGBoost (Calibrated)": evaluate_predictions(
                y_test.to_numpy(), xgb_cal_test_probs, threshold=optimal_threshold
            ),
        },
        "models_test_at_0_50_threshold": {
            "Logistic Regression (Scaled)": evaluate_predictions(
                y_test.to_numpy(), lr_test_probs, threshold=0.50
            ),
            "HistGradientBoosting": evaluate_predictions(
                y_test.to_numpy(), hgb_test_probs, threshold=0.50
            ),
            "XGBoost (Calibrated)": evaluate_predictions(
                y_test.to_numpy(), xgb_cal_test_probs, threshold=0.50
            ),
        },
    }

    # Save model bundle and metrics.json to artifacts/
    model_bundle = {
        "primary_model": calibrated_xgb,
        "xgb_base": xgb_raw,
        "logistic_regression": lr_model,
        "hist_gbdt": hgb_model,
        "optimal_threshold": optimal_threshold,
        "feature_names": list(X_train.columns),
    }
    joblib.dump(model_bundle, config.get_artifact_path("model.joblib"))

    metrics_path = config.get_artifact_path("metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return metrics_payload
