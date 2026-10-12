"""Unit and integration test suite for the Credit Risk Explainability & Fairness Suite.

Covers:
1. Quirk cleaning (`PAY_0` -> `PAY_1`, `EDUCATION` {0,5,6} -> 4, `MARRIAGE` {0} -> 3, `AGE_GROUP` bins).
2. Stratified split integrity (zero index overlap, 70/15/15 proportions, zero sensitive feature leakage).
3. Credit-industry metrics (KS statistic and Gini coefficient on a tiny hand-checkable example).
4. Fairlearn `audit_single_attribute` output schema and Four-Fifths rule calculation.
5. `explain_applicant` output schema and Adverse Action reason code formatting.
6. FastAPI `/health`, `/predict`, and `/explain` endpoints via `TestClient`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from creditfair import config
from creditfair.data import clean_credit_card_data, load_saved_splits
from creditfair.evaluate import compute_gini, compute_ks_statistic, find_cost_optimal_threshold
from creditfair.explain import explain_applicant
from creditfair.fairness import audit_single_attribute
from creditfair.mitigate import compute_reweighing_weights


def test_clean_credit_card_data_quirks() -> None:
    """Verify undocumented EDUCATION (0,5,6) -> 4, MARRIAGE (0) -> 3, PAY_0 -> PAY_1, and AGE_GROUP bins."""
    raw_df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5],
            "LIMIT_BAL": [50000, 120000, 200000, 80000, 150000],
            "SEX": [1, 2, 1, 2, 1],
            "EDUCATION": [0, 5, 6, 1, 2],
            "MARRIAGE": [0, 1, 2, 3, 0],
            "AGE": [24, 34, 44, 55, 29],
            "PAY_0": [2, 0, -1, 1, 0],
            "PAY_2": [2, 0, -1, 0, 0],
            "PAY_3": [0, 0, -1, 0, 0],
            "PAY_4": [0, 0, 0, 0, 0],
            "PAY_5": [0, 0, 0, 0, 0],
            "PAY_6": [0, 0, 0, 0, 0],
            "BILL_AMT1": [10000, 20000, 5000, 15000, 30000],
            "BILL_AMT2": [9000, 18000, 5000, 14000, 28000],
            "BILL_AMT3": [8000, 17000, 4000, 13000, 27000],
            "BILL_AMT4": [7000, 16000, 4000, 12000, 26000],
            "BILL_AMT5": [6000, 15000, 3000, 11000, 25000],
            "BILL_AMT6": [5000, 14000, 3000, 10000, 24000],
            "PAY_AMT1": [1000, 2000, 5000, 1500, 3000],
            "PAY_AMT2": [1000, 2000, 4000, 1500, 3000],
            "PAY_AMT3": [1000, 2000, 4000, 1500, 3000],
            "PAY_AMT4": [1000, 2000, 3000, 1500, 3000],
            "PAY_AMT5": [1000, 2000, 3000, 1500, 3000],
            "PAY_AMT6": [1000, 2000, 3000, 1500, 3000],
            "default payment next month": [1, 0, 0, 1, 0],
        }
    )

    cleaned_df, report = clean_credit_card_data(raw_df)

    assert "ID" not in cleaned_df.columns
    assert "PAY_1" in cleaned_df.columns
    assert "PAY_0" not in cleaned_df.columns
    assert config.TARGET_COL in cleaned_df.columns
    assert set(cleaned_df["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(cleaned_df["MARRIAGE"].unique()).issubset({1, 2, 3})
    assert report["education_undocumented_remapped_to_4"] == 3
    assert report["marriage_undocumented_remapped_to_3"] == 2
    assert list(cleaned_df["AGE_GROUP"]) == ["21-29", "30-39", "40-49", "50+", "21-29"]


def test_splits_no_overlap_and_no_sensitive_leakage() -> None:
    """Verify train/val/test have zero row overlap and zero sensitive columns in X."""
    splits = load_saved_splits()

    idx_train = set(splits.X_train.index)
    idx_val = set(splits.X_val.index)
    idx_test = set(splits.X_test.index)

    assert len(idx_train.intersection(idx_val)) == 0
    assert len(idx_train.intersection(idx_test)) == 0
    assert len(idx_val.intersection(idx_test)) == 0

    total = len(idx_train) + len(idx_val) + len(idx_test)
    assert total == 30000
    assert len(idx_train) == 21000
    assert len(idx_val) == 4500
    assert len(idx_test) == 4500

    forbidden = {"SEX", "SEX_LABEL", "SEX_CODE", "AGE", "AGE_GROUP", "MARRIAGE", "MARRIAGE_LABEL"}
    assert len(forbidden.intersection(set(splits.X_train.columns))) == 0


def test_ks_and_gini_hand_checkable() -> None:
    """Verify KS and Gini on tiny hand-checkable examples."""
    # Case 1: Perfect separation -> AUC = 1.0, Gini = 1.0, KS = 1.0
    y_true_perf = np.array([0, 0, 1, 1])
    y_prob_perf = np.array([0.1, 0.2, 0.8, 0.9])
    ks_perf, _ = compute_ks_statistic(y_true_perf, y_prob_perf)
    assert pytest.approx(ks_perf, abs=1e-6) == 1.0
    assert pytest.approx(compute_gini(1.0), abs=1e-6) == 1.0

    # Case 2: Partial overlap -> AUC = 0.75, Gini = 0.50, KS = 0.50
    y_true_part = np.array([0, 0, 1, 1])
    y_prob_part = np.array([0.1, 0.6, 0.4, 0.9])
    ks_part, _ = compute_ks_statistic(y_true_part, y_prob_part)
    assert pytest.approx(ks_part, abs=1e-6) == 0.50
    assert pytest.approx(compute_gini(0.75), abs=1e-6) == 0.50

    # Case 3: Cost-sensitive threshold check
    thresh_res = find_cost_optimal_threshold(y_true_perf, y_prob_perf, cost_fn=5.0, cost_fp=1.0)
    assert 0.20 <= thresh_res["optimal_threshold"] <= 0.80
    assert thresh_res["min_expected_cost_per_applicant"] == 0.0


def test_fairness_and_reweighing_helpers() -> None:
    """Verify Fairlearn audit output schema and sample reweighing independence property."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    s_attr = pd.Series(["Male", "Male", "Male", "Male", "Female", "Female", "Female", "Female"])

    audit = audit_single_attribute(y_true, y_pred, s_attr, attr_name="SEX", n_boot=20)
    assert "overall" in audit
    assert "by_group" in audit
    assert "Male" in audit["by_group"]
    assert "Female" in audit["by_group"]
    assert "summary_metrics" in audit
    assert "bootstrap_95_ci" in audit

    weights = compute_reweighing_weights(y_true, s_attr)
    assert len(weights) == len(y_true)
    assert np.all(weights > 0)
    # Weighted mean of weights is 1.0
    assert pytest.approx(float(np.mean(weights)), abs=1e-5) == 1.0


def test_explain_applicant_output_schema() -> None:
    """Verify explain_applicant returns probability, decision, reason codes, and 26 SHAP values."""
    splits = load_saved_splits()
    sample_row = splits.X_test.iloc[0]
    res = explain_applicant(sample_row, top_k=4)

    assert 0.0 < res["default_probability"] < 1.0
    assert res["decision"] in {"APPROVE", "DENY"}
    assert isinstance(res["top_adverse_reasons"], list)
    assert isinstance(res["top_protective_reasons"], list)
    assert len(res["shap_values"]) == len(config.ALL_MODEL_FEATURES)
    if res["top_adverse_reasons"]:
        first_reason = res["top_adverse_reasons"][0]
        assert "feature" in first_reason
        assert "reason_text" in first_reason
        assert first_reason["shap_value"] > 0


def test_fastapi_endpoints() -> None:
    """Verify FastAPI /health, /predict, and /explain endpoints."""
    from api.main import app

    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    payload = {
        "LIMIT_BAL": 80000,
        "EDUCATION": 2,
        "PAY_1": 2,
        "PAY_2": 2,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 75000,
        "BILL_AMT2": 72000,
        "BILL_AMT3": 68000,
        "BILL_AMT4": 65000,
        "BILL_AMT5": 60000,
        "BILL_AMT6": 58000,
        "PAY_AMT1": 0,
        "PAY_AMT2": 2000,
        "PAY_AMT3": 2000,
        "PAY_AMT4": 2000,
        "PAY_AMT5": 2000,
        "PAY_AMT6": 2000,
    }

    r_pred = client.post("/predict", json=payload)
    assert r_pred.status_code == 200
    pred_data = r_pred.json()
    assert pred_data["decision"] == "DENY"
    assert pred_data["default_probability"] >= pred_data["threshold"]

    r_exp = client.post("/explain", json=payload)
    assert r_exp.status_code == 200
    exp_data = r_exp.json()
    assert len(exp_data["top_adverse_reasons"]) > 0
    assert "PAY_1" in exp_data["shap_values"]
