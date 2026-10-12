"""Home page of the Credit Risk Explainability & Fairness Suite Streamlit app.

Loads precomputed JSON/Parquet artifacts using `@st.cache_data` so the app starts
instantly on Streamlit Community Cloud without retraining models on startup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure `creditfair` is importable even if Streamlit Cloud runs without `pip install -e .`
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from creditfair import config

st.set_page_config(
    page_title="Credit Risk Explainability & Fairness Suite",
    page_icon="⚖️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Why @st.cache_data vs @st.cache_resource?
# - @st.cache_data is used for serializable data objects (dicts, DataFrames, JSONs).
#   Streamlit serializes/copies the return value so mutations in one user session
#   cannot corrupt cached data for other users.
# - @st.cache_resource (used in Underwriter View) is for non-serializable singleton
#   resources like fitted ML models or database connections shared across sessions.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json_artifact(filename: str) -> Dict[str, Any]:
    path = config.ARTIFACTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.title("⚖️ Credit Risk Explainability & Fairness Auditing Suite")
    st.caption(
        "Regulator-Facing Explainable AI (SHAP Adverse Action Notices) & Algorithmic Fairness "
        "(Fairlearn MetricFrame + Bias Mitigation Trade-Offs)"
    )

    st.warning(
        "**⚠️ Educational & Regulatory Disclaimer:** This application is an educational portfolio "
        "demonstration built on the historical 2005 UCI Taiwan Credit Card Default dataset "
        "($N = 30,000$). Its fairness metrics and predictions reflect historical patterns in that "
        "dataset and **must never be used for real-world credit underwriting or lending decisions**."
    )

    eda = load_json_artifact("eda_summary.json")
    metrics = load_json_artifact("metrics.json")
    fairness_before = load_json_artifact("fairness_before.json")

    primary_name = metrics["primary_model"]
    primary_test = metrics["models_test_set"][primary_name]
    opt_thresh = metrics["optimal_threshold"]

    # Top KPI Row (100% read from generated artifacts)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Dataset Rows (UCI)", f"{eda['total_rows']:,}", f"{eda['default_rate']:.2%} Default Rate")
    col2.metric("Test ROC-AUC", f"{primary_test['roc_auc']:.4f}", f"Gini: {primary_test['gini']:.4f}")
    col3.metric("KS Statistic", f"{primary_test['ks_statistic']:.4f}", f"PR-AUC: {primary_test['pr_auc']:.4f}")
    col4.metric("Brier Score (Calibrated)", f"{primary_test['brier_score']:.4f}", "Isotonic Calibration")
    col5.metric("Cost-Optimal Cutoff", f"{opt_thresh:.2f}", "5:1 FN:FP Cost Ratio")

    st.markdown("---")

    # Navigation Overview Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🔍 1. Underwriter View")
        st.write(
            "Inspect individual credit applications with calibrated default probabilities, "
            "**interactive SHAP waterfall plots**, and plain-English **Adverse Action Notice reason codes** "
            "(mirroring US ECOA / Regulation B requirements)."
        )
    with c2:
        st.subheader("📊 2. Compliance & Fairness View")
        st.write(
            "Audit subgroup disparities across `SEX`, `AGE_GROUP`, and `MARRIAGE` using "
            "`fairlearn.metrics.MetricFrame` with **95% bootstrap confidence intervals**, proxy leakage "
            "detection, and **Pre / In / Post-processing bias mitigation**."
        )
    with c3:
        st.subheader("📋 3. Model Card")
        st.write(
            "Read the formal **Mitchell et al. (2019) Model Card** documenting intended use, "
            "dataset quirks, subgroup metrics, fairness-accuracy trade-offs, and ethical limitations."
        )

    st.markdown("---")

    # Section 1: Model Comparison & Probability Calibration
    st.header("📈 Phase 2: Baseline Model Comparison & Probability Calibration")
    left_col, right_col = st.columns([1.25, 1.0])

    with left_col:
        st.subheader("Held-Out Test Set Performance ($N = 4,500$)")
        rows = []
        for model_label, m_dict in metrics["models_test_set"].items():
            rows.append(
                {
                    "Model": model_label,
                    "ROC-AUC": m_dict["roc_auc"],
                    "PR-AUC": m_dict["pr_auc"],
                    "Gini": m_dict["gini"],
                    "KS Stat": m_dict["ks_statistic"],
                    "Brier Score ↓": m_dict["brier_score"],
                    "Accuracy": m_dict["accuracy"],
                    "Recall (TPR)": m_dict["recall_tpr"],
                }
            )
        df_models = pd.DataFrame(rows)
        st.dataframe(df_models, use_container_width=True, hide_index=True)
        st.caption(
            "**Why Brier Score & Calibration Matter:** Class-weighted XGBoost (`scale_pos_weight`) "
            "achieves similar ranking power (ROC-AUC) but severely inflates predicted probabilities, "
            "resulting in a poor Brier score. Isotonic calibration restores true probability estimates."
        )

    with right_col:
        st.subheader("Reliability (Calibration) Curve — Before vs. After")
        fig_cal = go.Figure()
        fig_cal.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfectly Calibrated",
            )
        )
        for label, color in [
            ("XGBoost (Weighted, Uncalibrated)", "#EF4444"),
            ("XGBoost (Raw Uncalibrated)", "#F59E0B"),
            ("XGBoost (Calibrated)", "#10B981"),
        ]:
            curve = metrics["models_test_set"][label]["calibration_curve"]
            fig_cal.add_trace(
                go.Scatter(
                    x=curve["prob_pred"],
                    y=curve["prob_true"],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2.5),
                )
            )
        fig_cal.update_layout(
            xaxis_title="Mean Predicted Default Probability",
            yaxis_title="Observed Default Frequency",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("---")

    # Section 2: Dataset Quirks & Demographic Base Rates
    st.header("🗂️ Phase 1: Dataset Hygiene & Demographic Base Default Rates")
    b_col1, b_col2 = st.columns([1.0, 1.2])

    with b_col1:
        st.subheader("Documented Quirks Cleaned in `data.py`")
        st.markdown(
            f"""
            - **Total Records:** `{eda['total_rows']:,}` (`Train: {eda['split_sizes']['train']:,}`, `Val: {eda['split_sizes']['val']:,}`, `Test: {eda['split_sizes']['test']:,}`)
            - **Column Alignment:** Renamed `PAY_0` $\\rightarrow$ `PAY_1` so repayment history follows `PAY_1`..`PAY_6`.
            - **Undocumented `EDUCATION` Codes (`0, 5, 6`):** Consolidated **{eda['education_undocumented_remapped_to_4']:,} rows** into code `4` (*Others*).
            - **Undocumented `MARRIAGE` Code (`0`):** Consolidated **{eda['marriage_undocumented_remapped_to_3']:,} rows** into code `3` (*Others*).
            - **Leakage Guard:** Protected attributes (`SEX`, `AGE`, `AGE_GROUP`, `MARRIAGE`) are stored in separate audit frames (`S_train`, `S_val`, `S_test`) and **never** passed as model features.
            """
        )

    with b_col2:
        st.subheader("Base Default Rates by Protected Group")
        attr_choice = st.radio(
            "Select Demographic Dimension:",
            options=["SEX", "AGE_GROUP", "MARRIAGE", "SEX_x_AGE_GROUP"],
            horizontal=True,
        )
        group_rates = eda["group_base_rates"][attr_choice]
        df_base = pd.DataFrame(
            [
                {
                    "Group": grp,
                    "Count (N)": vals["n"],
                    "Base Default Rate": vals["default_rate"],
                }
                for grp, vals in group_rates.items()
            ]
        )
        fig_base = go.Figure(
            go.Bar(
                x=df_base["Group"],
                y=df_base["Base Default Rate"],
                text=[f"{r:.1%} (n={n:,})" for r, n in zip(df_base["Base Default Rate"], df_base["Count (N)"])],
                textposition="auto",
                marker_color="#3B82F6",
            )
        )
        fig_base.add_hline(
            y=eda["default_rate"],
            line_dash="dot",
            line_color="#EF4444",
            annotation_text=f"Overall Default Rate ({eda['default_rate']:.1%})",
        )
        fig_base.update_layout(
            yaxis_title="Observed Default Rate",
            yaxis_tickformat=".0%",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_base, use_container_width=True)


if __name__ == "__main__":
    main()
