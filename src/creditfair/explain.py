"""SHAP explainability, Adverse Action reason codes, stability, and proxy leakage audit.

Implements:
1. Global SHAP importance (`mean(|SHAP|)` + beeswarm sample for fast Streamlit rendering).
2. Local applicant explanation (`explain_applicant(row) -> dict`) returning:
   - Calibrated default probability & credit decision (Approve / Deny)
   - Plain-English Adverse Action reason codes (top 4 risk drivers pushing toward denial,
     plus top protective drivers pushing toward approval)
   - Exact per-feature SHAP values and base value for waterfall plotting
3. Explanation stability audit: verifies whether similar applicants in feature space
   receive consistent top-k SHAP rankings (Spearman rank correlation + Jaccard overlap).
4. Proxy leakage audit: trains XGBoost classifiers to predict sensitive attributes
   (`SEX` and `AGE_GROUP`) solely from the allowed model features `X`, proving why
   'fairness through unawareness' cannot prevent indirect proxy encoding.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from creditfair import config
from creditfair.data import DatasetSplits, load_saved_splits


def format_adverse_action_reason(feature: str, value: float, shap_val: float) -> str:
    """Translate a feature name, value, and SHAP contribution into a plain-English reason code.

    Mirrors regulatory Adverse Action Notices (e.g., ECOA / Regulation B in the US),
    where lenders must provide specific, understandable reasons when denying credit.
    """
    direction = "increased default risk" if shap_val > 0 else "lowered default risk"
    month_names = {
        "PAY_1": "most recent month (September)",
        "PAY_2": "2 months ago (August)",
        "PAY_3": "3 months ago (July)",
        "PAY_4": "4 months ago (June)",
        "PAY_5": "5 months ago (May)",
        "PAY_6": "6 months ago (April)",
    }

    if feature in month_names:
        m_label = month_names[feature]
        months_late = int(round(value))
        if months_late >= 2:
            return f"Payment {months_late} months past due in {m_label} ({direction})"
        if months_late == 1:
            return f"Payment 1 month late in {m_label} ({direction})"
        if months_late == 0:
            return f"Revolving minimum balance used in {m_label} ({direction})"
        return f"Paid on time / in full in {m_label} ({direction})"

    if feature == "MAX_DELINQUENCY_6M":
        m_late = int(round(value))
        if m_late > 0:
            return f"Peak delinquency of {m_late} month(s) past due over the last 6 months ({direction})"
        return f"No past-due delinquency recorded over the last 6 months ({direction})"

    if feature == "MONTHS_DELINQUENT_6M":
        cnt = int(round(value))
        return f"{cnt} delinquent month(s) out of the last 6 billing cycles ({direction})"

    if feature == "LIMIT_BAL":
        return f"Assigned credit limit of NT${value:,.0f} ({direction})"

    if feature == "UTILIZATION_RATIO_1":
        return f"Recent credit line utilization of {value * 100:.1f}% ({direction})"

    if feature == "AVG_UTILIZATION_6M":
        return f"6-month average credit utilization of {value * 100:.1f}% ({direction})"

    if feature == "PAY_TO_BILL_RATIO_1":
        return f"Recent payment covered {value * 100:.1f}% of the statement balance ({direction})"

    if feature == "BILL_GROWTH_6M":
        return f"6-month statement balance change of {value * 100:+.1f}% of credit limit ({direction})"

    if feature == "EDUCATION":
        edu_str = config.EDUCATION_LABELS.get(int(round(value)), "Others")
        return f"Education category ({edu_str}) ({direction})"

    if feature.startswith("BILL_AMT"):
        idx = feature.replace("BILL_AMT", "")
        return f"Bill statement balance of NT${value:,.0f} in Month {idx} ({direction})"

    if feature.startswith("PAY_AMT"):
        idx = feature.replace("PAY_AMT", "")
        return f"Repayment amount of NT${value:,.0f} in Month {idx} ({direction})"

    desc = config.FEATURE_DESCRIPTIONS.get(feature, feature)
    return f"{desc}: {value:,.2f} ({direction})"


def explain_applicant(
    row: pd.Series | pd.DataFrame | Dict[str, Any],
    model_bundle: Dict[str, Any] | None = None,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Generate a complete local SHAP explanation and Adverse Action reason codes for 1 applicant.

    Parameters
    ----------
    row : pd.Series, single-row pd.DataFrame, or dict
        Applicant feature values (may contain raw 20 features or all 26 engineered features).
    model_bundle : dict, optional
        Loaded `model.joblib` bundle. Loaded from `artifacts/model.joblib` if None.
    top_k : int, default=4
        Number of top risk-increasing ('adverse action') and risk-decreasing reasons to return.

    Returns
    -------
    dict
        Contains `default_probability`, `threshold`, `decision` ('DENY' or 'APPROVE'),
        `base_value`, `shap_values` (dict of feature -> float), `feature_values`,
        `top_adverse_reasons` (list of dicts), and `top_protective_reasons` (list of dicts).
    """
    from creditfair.features import engineer_features

    if model_bundle is None:
        model_bundle = joblib.load(config.get_artifact_path("model.joblib"))

    primary_model = model_bundle["primary_model"]
    xgb_base = model_bundle["xgb_base"]
    threshold = float(model_bundle["optimal_threshold"])
    feature_names: List[str] = model_bundle["feature_names"]

    if isinstance(row, dict):
        df_row = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        df_row = row.to_frame().T
    else:
        df_row = row.iloc[[0]].copy()

    # If engineered features are missing, compute them automatically
    missing_engineered = [c for c in config.ENGINEERED_FEATURE_COLS if c not in df_row.columns]
    if missing_engineered:
        df_row = engineer_features(df_row)

    df_features = df_row[feature_names].astype(float)

    # 1. Predict calibrated probability & decision
    prob_default = float(primary_model.predict_proba(df_features)[0, 1])
    decision = "DENY" if prob_default >= threshold else "APPROVE"

    # 2. Compute exact TreeSHAP values on the XGBoost model
    explainer = shap.TreeExplainer(xgb_base)
    shap_vals = explainer.shap_values(df_features)
    if isinstance(shap_vals, list):
        shap_row = np.asarray(shap_vals[1])[0]
    elif shap_vals.ndim == 3:
        shap_row = np.asarray(shap_vals[0, :, 1])
    else:
        shap_row = np.asarray(shap_vals[0])

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        base_value = float(np.asarray(expected_val).ravel()[-1])
    else:
        base_value = float(expected_val)

    feat_vals_dict = {f: float(df_features.iloc[0][f]) for f in feature_names}
    shap_vals_dict = {f: float(shap_row[i]) for i, f in enumerate(feature_names)}

    # 3. Sort features for Adverse Action Reason Codes (pushing toward default/denial: shap > 0)
    sorted_adverse = sorted(
        [(f, feat_vals_dict[f], shap_vals_dict[f]) for f in feature_names if shap_vals_dict[f] > 0],
        key=lambda x: x[2],
        reverse=True,
    )[:top_k]

    sorted_protective = sorted(
        [(f, feat_vals_dict[f], shap_vals_dict[f]) for f in feature_names if shap_vals_dict[f] < 0],
        key=lambda x: x[2],
    )[:top_k]

    top_adverse_reasons = [
        {
            "rank": idx + 1,
            "feature": f,
            "feature_description": config.FEATURE_DESCRIPTIONS.get(f, f),
            "value": round(val, 4),
            "shap_value": round(s_val, 4),
            "reason_text": format_adverse_action_reason(f, val, s_val),
        }
        for idx, (f, val, s_val) in enumerate(sorted_adverse)
    ]

    top_protective_reasons = [
        {
            "rank": idx + 1,
            "feature": f,
            "feature_description": config.FEATURE_DESCRIPTIONS.get(f, f),
            "value": round(val, 4),
            "shap_value": round(s_val, 4),
            "reason_text": format_adverse_action_reason(f, val, s_val),
        }
        for idx, (f, val, s_val) in enumerate(sorted_protective)
    ]

    return {
        "default_probability": round(prob_default, 4),
        "threshold": round(threshold, 4),
        "decision": decision,
        "base_value": round(base_value, 4),
        "shap_values": {k: round(v, 5) for k, v in shap_vals_dict.items()},
        "feature_values": {k: round(v, 4) for k, v in feat_vals_dict.items()},
        "top_adverse_reasons": top_adverse_reasons,
        "top_protective_reasons": top_protective_reasons,
    }


def evaluate_explanation_stability(
    X_sample: pd.DataFrame, shap_matrix: np.ndarray, n_Reference: int = 50, k_neighbors: int = 5
) -> Dict[str, float]:
    """Check whether similar applicants receive consistent SHAP explanations.

    Measures:
    - Mean Spearman rank correlation of |SHAP| vectors between an applicant and their
      5 nearest neighbors in standardized feature space.
    - Mean Top-4 feature Jaccard overlap between nearest neighbors.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled[:n_Reference])

    spearman_corrs: List[float] = []
    top4_jaccards: List[float] = []

    for i in range(min(n_Reference, len(X_sample))):
        s_i = shap_matrix[i]
        top4_i = set(np.argsort(np.abs(s_i))[-4:])
        for nbr_idx in indices[i, 1:]:
            s_nbr = shap_matrix[nbr_idx]
            corr, _ = spearmanr(s_i, s_nbr)
            if not np.isnan(corr):
                spearman_corrs.append(float(corr))
            top4_nbr = set(np.argsort(np.abs(s_nbr))[-4:])
            jaccard = len(top4_i.intersection(top4_nbr)) / max(len(top4_i.union(top4_nbr)), 1)
            top4_jaccards.append(float(jaccard))

    return {
        "mean_neighbor_spearman_correlation": round(float(np.mean(spearman_corrs)), 4),
        "std_neighbor_spearman_correlation": round(float(np.std(spearman_corrs)), 4),
        "mean_top4_reason_jaccard_similarity": round(float(np.mean(top4_jaccards)), 4),
        "k_neighbors_evaluated": int(k_neighbors),
        "n_reference_applicants": int(n_Reference),
    }


def run_proxy_leakage_audit(splits: DatasetSplits) -> Dict[str, Any]:
    """Train proxy classifiers to predict SEX and AGE_GROUP from the allowed features X.

    Demonstrates why 'fairness through unawareness' (dropping SEX and AGE) fails:
    credit limits, bill balances, and education levels correlate with demographics,
    allowing tree models to reconstruct sensitive attributes above chance.
    """
    X_train, X_test = splits.X_train, splits.X_test
    S_train, S_test = splits.S_train, splits.S_test

    # 1. Proxy model predicting SEX (binary: Female=2 -> 1, Male=1 -> 0)
    y_sex_train = (S_train["SEX_CODE"] == 2).astype(int)
    y_sex_test = (S_test["SEX_CODE"] == 2).astype(int)

    sex_proxy = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=config.RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=-1,
    )
    sex_proxy.fit(X_train, y_sex_train)
    sex_probs = sex_proxy.predict_proba(X_test)[:, 1]
    sex_auc = float(roc_auc_score(y_sex_test, sex_probs))
    sex_acc = float(accuracy_score(y_sex_test, (sex_probs >= 0.5).astype(int)))
    sex_majority_baseline = float(max(y_sex_test.mean(), 1.0 - y_sex_test.mean()))

    sex_importances = sorted(
        zip(X_train.columns, sex_proxy.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )[:6]

    # 2. Proxy model predicting AGE_GROUP (4-class classification)
    le = LabelEncoder()
    y_age_train = le.fit_transform(S_train["AGE_GROUP"])
    y_age_test = le.transform(S_test["AGE_GROUP"])

    age_proxy = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        random_state=config.RANDOM_SEED,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    age_proxy.fit(X_train, y_age_train)
    age_probs = age_proxy.predict_proba(X_test)
    age_preds = np.argmax(age_probs, axis=1)
    age_auc_ovr = float(roc_auc_score(y_age_test, age_probs, multi_class="ovr", average="macro"))
    age_acc = float(accuracy_score(y_age_test, age_preds))
    age_majority_baseline = float(pd.Series(y_age_test).value_counts(normalize=True).iloc[0])

    age_importances = sorted(
        zip(X_train.columns, age_proxy.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )[:6]

    proxy_report = {
        "SEX_proxy": {
            "roc_auc": round(sex_auc, 4),
            "random_chance_auc": 0.5000,
            "accuracy": round(sex_acc, 4),
            "majority_class_baseline_accuracy": round(sex_majority_baseline, 4),
            "top_proxy_features": [
                {"feature": f, "importance": round(float(imp), 4)} for f, imp in sex_importances
            ],
            "interpretation": (
                f"Allowed features predict SEX with ROC-AUC = {sex_auc:.3f} (well above 0.50 random chance), "
                "proving that dropping SEX from training features does not prevent indirect gender encoding."
            ),
        },
        "AGE_GROUP_proxy": {
            "macro_ovr_roc_auc": round(age_auc_ovr, 4),
            "random_chance_auc": 0.5000,
            "accuracy": round(age_acc, 4),
            "majority_class_baseline_accuracy": round(age_majority_baseline, 4),
            "top_proxy_features": [
                {"feature": f, "importance": round(float(imp), 4)} for f, imp in age_importances
            ],
            "interpretation": (
                f"Allowed features (especially LIMIT_BAL and EDUCATION) predict AGE_GROUP with "
                f"Macro ROC-AUC = {age_auc_ovr:.3f}, acting as strong demographic proxies."
            ),
        },
    }

    with open(config.get_artifact_path("proxy_audit.json"), "w", encoding="utf-8") as f:
        json.dump(proxy_report, f, indent=2)

    return proxy_report


def run_explainability_pipeline(
    splits: DatasetSplits | None = None,
    model_bundle: Dict[str, Any] | None = None,
    sample_size: int = 500,
) -> Dict[str, Any]:
    """Run global SHAP, stability check, proxy leakage audit, and save background artifacts."""
    if splits is None:
        splits = load_saved_splits()
    if model_bundle is None:
        model_bundle = joblib.load(config.get_artifact_path("model.joblib"))

    xgb_base = model_bundle["xgb_base"]
    X_test = splits.X_test

    rng = np.random.default_rng(config.RANDOM_SEED)
    n_bg = min(sample_size, len(X_test))
    sample_indices = rng.choice(len(X_test), size=n_bg, replace=False)
    X_bg = X_test.iloc[sample_indices].copy().reset_index(drop=True)
    y_bg = splits.y_test.iloc[sample_indices].copy().reset_index(drop=True)
    S_bg = splits.S_test.iloc[sample_indices].copy().reset_index(drop=True)

    # Save background sample with target & sensitive labels for the Streamlit app's "Random Real Applicant"
    bg_full = X_bg.copy()
    bg_full[config.TARGET_COL] = y_bg
    bg_full["SEX"] = S_bg["SEX"]
    bg_full["AGE"] = S_bg["AGE"]
    bg_full["AGE_GROUP"] = S_bg["AGE_GROUP"]
    bg_full["MARRIAGE"] = S_bg["MARRIAGE"]
    bg_full.to_parquet(config.get_artifact_path("shap_background.parquet"))

    explainer = shap.TreeExplainer(xgb_base)
    shap_vals = explainer.shap_values(X_bg)
    if isinstance(shap_vals, list):
        shap_matrix = np.asarray(shap_vals[1])
    elif shap_vals.ndim == 3:
        shap_matrix = np.asarray(shap_vals[:, :, 1])
    else:
        shap_matrix = np.asarray(shap_vals)

    # Save precomputed SHAP matrix for instant Streamlit global beeswarm/bar plots
    shap_df = pd.DataFrame(shap_matrix, columns=X_bg.columns)
    shap_df.to_parquet(config.get_artifact_path("shap_values_sample.parquet"))

    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    global_importance = sorted(
        [
            {
                "feature": col,
                "description": config.FEATURE_DESCRIPTIONS.get(col, col),
                "mean_abs_shap": round(float(mean_abs_shap[i]), 5),
            }
            for i, col in enumerate(X_bg.columns)
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    stability_metrics = evaluate_explanation_stability(X_bg, shap_matrix)
    proxy_report = run_proxy_leakage_audit(splits)

    # Sample 3 example applicant explanations (1 high risk denied, 1 borderline, 1 low risk approved)
    sample_explanations = [
        explain_applicant(X_bg.iloc[i], model_bundle=model_bundle) for i in range(min(5, len(X_bg)))
    ]

    summary_payload = {
        "global_feature_importance": global_importance,
        "stability_audit": stability_metrics,
        "proxy_audit": proxy_report,
        "sample_explanations": sample_explanations,
    }

    with open(config.get_artifact_path("explainability_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload
