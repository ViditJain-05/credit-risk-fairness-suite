"""Page 1: Underwriter View — Local SHAP Explanations & Adverse Action Reason Codes.

Allows a credit underwriter or auditor to:
1. Pick a real test-set applicant (or load a random / high-risk / low-risk sample)
   or adjust key financial sliders manually.
2. View the calibrated default probability, cost-sensitive credit decision,
   and regulatory Adverse Action Notice reason codes in plain English.
3. Inspect the interactive local SHAP waterfall chart alongside global SHAP importance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from creditfair import config
from creditfair.explain import explain_applicant

st.set_page_config(
    page_title="Underwriter View | Credit Risk Suite",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> Dict[str, Any]:
    """Cache the fitted model bundle across all sessions as a shared singleton resource."""
    return joblib.load(config.ARTIFACTS_DIR / "model.joblib")


@st.cache_data(show_spinner=False)
def load_background_sample() -> pd.DataFrame:
    """Cache the 500-row test-set sample for fast applicant inspection."""
    return pd.read_parquet(config.ARTIFACTS_DIR / "shap_background.parquet")


@st.cache_data(show_spinner=False)
def load_explainability_summary() -> Dict[str, Any]:
    with open(config.ARTIFACTS_DIR / "explainability_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.title("🔍 Underwriter View — Individual Decision & Adverse Action Reasons")
    st.info(
        "**Regulatory Context (ECOA / Regulation B & GDPR Art. 22):** When an automated model "
        "denies credit, the lender must provide specific, human-understandable **Adverse Action "
        "Reason Codes** explaining the principal factors that drove the denial."
    )

    model_bundle = load_model_bundle()
    bg_df = load_background_sample()
    exp_summary = load_explainability_summary()

    if "selected_applicant_idx" not in st.session_state:
        st.session_state["selected_applicant_idx"] = 0

    # Quick Preset Buttons
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 2])
    with btn_col1:
        if st.button("🎲 Random Real Applicant", use_container_width=True):
            rng = np.random.default_rng()
            st.session_state["selected_applicant_idx"] = int(rng.integers(0, len(bg_df)))
    with btn_col2:
        if st.button("🚨 High-Risk Sample", use_container_width=True):
            high_risk_indices = bg_df.index[bg_df["PAY_1"] >= 2].tolist()
            if high_risk_indices:
                st.session_state["selected_applicant_idx"] = int(high_risk_indices[0])
    with btn_col3:
        if st.button("✅ Prime Low-Risk Sample", use_container_width=True):
            low_risk_indices = bg_df.index[
                (bg_df["PAY_1"] <= 0) & (bg_df["MONTHS_DELINQUENT_6M"] == 0) & (bg_df["LIMIT_BAL"] >= 200000)
            ].tolist()
            if low_risk_indices:
                st.session_state["selected_applicant_idx"] = int(low_risk_indices[0])
    with btn_col4:
        selected_idx = st.number_input(
            "Test Sample Row Index (0 to 499):",
            min_value=0,
            max_value=len(bg_df) - 1,
            value=int(st.session_state["selected_applicant_idx"]),
            step=1,
        )
        st.session_state["selected_applicant_idx"] = int(selected_idx)

    base_row = bg_df.iloc[st.session_state["selected_applicant_idx"]].copy()

    # Sidebar / Expander for interactive what-if overrides
    with st.expander("🛠️ Edit / What-If Applicant Features (Interactive Sliders)", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            limit_bal = st.slider(
                "Credit Limit (LIMIT_BAL, NT$)",
                min_value=10000,
                max_value=800000,
                value=int(base_row["LIMIT_BAL"]),
                step=10000,
            )
            education = st.selectbox(
                "Education Level",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(int(base_row["EDUCATION"]))
                if int(base_row["EDUCATION"]) in [1, 2, 3, 4]
                else 1,
                format_func=lambda x: f"{x} - {config.EDUCATION_LABELS.get(x, 'Others')}",
            )
        with f_col2:
            pay_1 = st.slider(
                "Recent Repayment Status (PAY_1: -1=On Time, 0=Revolving, 1..8=Months Late)",
                min_value=-2,
                max_value=8,
                value=int(base_row["PAY_1"]),
                step=1,
            )
            pay_2 = st.slider(
                "2 Months Ago Status (PAY_2)",
                min_value=-2,
                max_value=8,
                value=int(base_row["PAY_2"]),
                step=1,
            )
        with f_col3:
            bill_amt1 = st.number_input(
                "Recent Statement Balance (BILL_AMT1, NT$)",
                min_value=-50000,
                max_value=1000000,
                value=int(base_row["BILL_AMT1"]),
                step=5000,
            )
            pay_amt1 = st.number_input(
                "Recent Payment Made (PAY_AMT1, NT$)",
                min_value=0,
                max_value=500000,
                value=int(base_row["PAY_AMT1"]),
                step=1000,
            )
        with f_col4:
            st.markdown("**Audit-Only Protected Profile (Never Fed to Model):**")
            st.write(f"- **Sex:** `{base_row.get('SEX', 'N/A')}`")
            st.write(f"- **Age Group:** `{base_row.get('AGE_GROUP', 'N/A')}` (Age: `{base_row.get('AGE', 'N/A')}`)")
            st.write(f"- **Marriage:** `{base_row.get('MARRIAGE', 'N/A')}`")
            actual_label = "Defaulted (1)" if int(base_row.get(config.TARGET_COL, 0)) == 1 else "Paid On Time (0)"
            st.write(f"- **Actual Ground Truth:** `{actual_label}`")

    # Apply interactive overrides and recompute engineered features
    applicant_dict = base_row[config.BASE_FEATURE_COLS].to_dict()
    applicant_dict["LIMIT_BAL"] = float(limit_bal)
    applicant_dict["EDUCATION"] = float(education)
    applicant_dict["PAY_1"] = float(pay_1)
    applicant_dict["PAY_2"] = float(pay_2)
    applicant_dict["BILL_AMT1"] = float(bill_amt1)
    applicant_dict["PAY_AMT1"] = float(pay_amt1)

    explanation = explain_applicant(applicant_dict, model_bundle=model_bundle, top_k=4)
    prob = explanation["default_probability"]
    thresh = explanation["threshold"]
    decision = explanation["decision"]

    st.markdown("---")

    # Decision Banner & Reason Codes
    d_col1, d_col2 = st.columns([1.0, 1.4])
    with d_col1:
        if decision == "DENY":
            st.error(f"### ❌ Decision: DENY CREDIT\n**Predicted Default Risk:** `{prob:.1%}` (Cutoff: `{thresh:.1%}`)")
        else:
            st.success(f"### ✅ Decision: APPROVE CREDIT\n**Predicted Default Risk:** `{prob:.1%}` (Cutoff: `{thresh:.1%}`)")

        st.progress(min(max(prob, 0.0), 1.0), text=f"Calibrated Default Probability: {prob:.1%}")

        st.subheader("📋 Adverse Action Reason Codes (Top Risk Drivers)")
        if explanation["top_adverse_reasons"]:
            for r in explanation["top_adverse_reasons"]:
                st.markdown(
                    f"**{r['rank']}.** {r['reason_text']} — *(`{r['feature']}`, SHAP = `{r['shap_value']:+.3f}`)*"
                )
        else:
            st.write("No positive risk-increasing factors found for this applicant.")

        st.subheader("🛡️ Top Protective / Mitigating Factors")
        for r in explanation["top_protective_reasons"]:
            st.markdown(
                f"**{r['rank']}.** {r['reason_text']} — *(`{r['feature']}`, SHAP = `{r['shap_value']:+.3f}`)*"
            )

    with d_col2:
        st.subheader("📊 Local SHAP Feature Attribution (Top 10 Features)")
        shap_items = sorted(
            explanation["shap_values"].items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:10][::-1]
        feats = [
            f"{k} = {explanation['feature_values'][k]:,.2f}" for k, _ in shap_items
        ]
        vals = [v for _, v in shap_items]
        colors = ["#EF4444" if v > 0 else "#10B981" for v in vals]

        fig_local = go.Figure(
            go.Bar(
                x=vals,
                y=feats,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in vals],
                textposition="auto",
            )
        )
        fig_local.update_layout(
            xaxis_title="SHAP Contribution to Log-Odds of Default (Red = Higher Risk, Green = Lower Risk)",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_local, use_container_width=True)

    st.markdown("---")

    # Global SHAP & Explanation Stability Audit
    g_col1, g_col2 = st.columns([1.3, 1.0])
    with g_col1:
        st.subheader("🌍 Global Feature Importance (Mean |SHAP| on Test Set)")
        global_imp = exp_summary["global_feature_importance"][:12][::-1]
        fig_glob = go.Figure(
            go.Bar(
                x=[x["mean_abs_shap"] for x in global_imp],
                y=[x["feature"] for x in global_imp],
                orientation="h",
                marker_color="#2563EB",
                text=[f"{x['mean_abs_shap']:.3f}" for x in global_imp],
                textposition="auto",
            )
        )
        fig_glob.update_layout(
            xaxis_title="Mean |SHAP Value| (Log-Odds Impact)",
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_glob, use_container_width=True)

    with g_col2:
        st.subheader("🔬 Explanation Stability Audit")
        stab = exp_summary["stability_audit"]
        st.metric(
            "Nearest-Neighbor SHAP Rank Correlation (Spearman)",
            f"{stab['mean_neighbor_spearman_correlation']:.3f}",
            f"± {stab['std_neighbor_spearman_correlation']:.3f} across k={stab['k_neighbors_evaluated']} neighbors",
        )
        st.metric(
            "Top-4 Adverse Reason Code Overlap (Jaccard)",
            f"{stab['mean_top4_reason_jaccard_similarity']:.1%}",
            "Consistent local explanations",
        )
        st.caption(
            "**Why Stability Matters:** If two nearly identical applicants receive completely "
            "different Adverse Action reason codes, the explanation system is unstable. Exact "
            "`TreeExplainer` on regularized shallow trees ($d=4$) produces high local rank stability."
        )


if __name__ == "__main__":
    main()
