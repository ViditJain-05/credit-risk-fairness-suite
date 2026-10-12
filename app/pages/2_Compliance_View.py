"""Page 2: Compliance & Fairness View — Fairlearn MetricFrame, Bootstrap CIs & Mitigation.

Provides the regulator/compliance officer dashboard:
1. Subgroup MetricFrame audit across `SEX`, `AGE_GROUP`, `MARRIAGE`, and `SEX_x_AGE_GROUP`.
2. 95% Bootstrap Confidence Intervals on fairness gaps (separating true disparity from noise).
3. Proxy Leakage Audit proving why 'fairness through unawareness' fails.
4. Bias Mitigation comparison (Sample Reweighing, ExponentiatedGradient, ThresholdOptimizer)
   and Fairness Gap vs. Accuracy/Cost trade-off visualization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from creditfair import config

st.set_page_config(
    page_title="Compliance & Fairness View | Credit Risk Suite",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_fairness_artifacts() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    with open(config.ARTIFACTS_DIR / "fairness_before.json", "r", encoding="utf-8") as f1:
        before = json.load(f1)
    with open(config.ARTIFACTS_DIR / "fairness_after.json", "r", encoding="utf-8") as f2:
        after = json.load(f2)
    with open(config.ARTIFACTS_DIR / "proxy_audit.json", "r", encoding="utf-8") as f3:
        proxy = json.load(f3)
    return before, after, proxy


def main() -> None:
    st.title("📊 Compliance & Fairness Audit Dashboard")
    st.caption(
        "Subgroup Disparity Analysis (Fairlearn MetricFrame + 95% Bootstrap CIs), "
        "Proxy Leakage Detection, and Algorithmic Bias Mitigation Trade-Offs"
    )

    fairness_before, fairness_after, proxy_audit = load_fairness_artifacts()

    # Section 1: Unmitigated Subgroup Audit + Bootstrap CIs
    st.header("1️⃣ Phase 4: Unmitigated Model Subgroup Audit (`MetricFrame`)")
    selected_attr = st.selectbox(
        "Select Protected Attribute to Audit:",
        options=["SEX", "AGE_GROUP", "MARRIAGE", "SEX_x_AGE_GROUP"],
        index=0,
    )

    audit_data = fairness_before["audits"][selected_attr]
    summary = audit_data["summary_metrics"]
    boot_ci = audit_data["bootstrap_95_ci"]

    # Top KPI cards for selected attribute
    k1, k2, k3, k4 = st.columns(4)
    di_app = summary["disparate_impact_approval_ratio"]
    passes_45 = summary["passes_four_fifths_approval_rule"]
    k1.metric(
        "Approval Disparate Impact Ratio",
        f"{di_app:.4f}",
        "✅ Passes 4/5ths Rule (≥0.80)" if passes_45 else "⚠️ Below 0.80 Threshold",
    )
    k2.metric(
        "Demographic Parity Diff (Denial Gap)",
        f"{summary['demographic_parity_difference']:.4f}",
        f"95% CI: [{boot_ci['demographic_parity_difference']['ci_lower_95']:.3f}, {boot_ci['demographic_parity_difference']['ci_upper_95']:.3f}]",
    )
    k3.metric(
        "Equal Opportunity Diff (TPR Gap)",
        f"{summary['equal_opportunity_difference_tpr_gap']:.4f}",
        f"95% CI: [{boot_ci['equal_opportunity_difference']['ci_lower_95']:.3f}, {boot_ci['equal_opportunity_difference']['ci_upper_95']:.3f}]",
    )
    k4.metric(
        "Equalized Odds Diff (Max TPR/FPR Gap)",
        f"{summary['equalized_odds_difference']:.4f}",
        f"FPR Gap: {summary['fpr_gap']:.4f}",
    )

    c_table, c_chart = st.columns([1.25, 1.0])
    with c_table:
        st.subheader(f"Per-Group MetricFrame Breakdown (`{selected_attr}`)")
        rows = []
        for grp, vals in audit_data["by_group"].items():
            rows.append(
                {
                    "Group": grp,
                    "Count (N)": vals["count"],
                    "Base Default Rate": vals["base_default_rate"],
                    "Denial Rate (Selection)": vals["selection_rate_denial"],
                    "Approval Rate": vals["approval_rate"],
                    "TPR (Recall)": vals["tpr"],
                    "FPR (Wrongful Denial)": vals["fpr"],
                    "Precision": vals["precision"],
                    "Accuracy": vals["accuracy"],
                }
            )
        df_grp = pd.DataFrame(rows)
        st.dataframe(df_grp, use_container_width=True, hide_index=True)

    with c_chart:
        st.subheader("95% Bootstrap Confidence Intervals on Fairness Gaps")
        gap_labels = [
            "Demographic Parity Diff",
            "Equal Opportunity Diff (TPR)",
            "Equalized Odds Diff",
        ]
        gap_keys = [
            "demographic_parity_difference",
            "equal_opportunity_difference",
            "equalized_odds_difference",
        ]
        means = [boot_ci[k]["mean"] for k in gap_keys]
        err_upper = [boot_ci[k]["ci_upper_95"] - boot_ci[k]["mean"] for k in gap_keys]
        err_lower = [boot_ci[k]["mean"] - boot_ci[k]["ci_lower_95"] for k in gap_keys]

        fig_ci = go.Figure(
            go.Bar(
                x=gap_labels,
                y=means,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=err_upper,
                    arrayminus=err_lower,
                    thickness=2,
                    width=6,
                ),
                marker_color=["#3B82F6", "#8B5CF6", "#F59E0B"],
                text=[f"{m:.3f}" for m in means],
                textposition="auto",
            )
        )
        fig_ci.update_layout(
            yaxis_title="Between-Group Disparity (Lower = Fairer)",
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_ci, use_container_width=True)

    st.markdown("---")

    # Section 2: Phase 5 Bias Mitigation & Trade-Off Analysis
    st.header("2️⃣ Phase 5: Bias Mitigation & Accuracy/Fairness Trade-Off")
    mit_attr = st.radio(
        "Select Protected Attribute for Mitigation Comparison:",
        options=["SEX", "AGE_GROUP"],
        horizontal=True,
    )

    comp_list = fairness_after["comparisons"][mit_attr]
    df_mit = pd.DataFrame(
        [
            {
                "Method": item["method"],
                "Paradigm": item["paradigm"],
                "Uses Protected Attr at Inference?": "Yes ⚠️"
                if item["requires_sensitive_at_inference"]
                else "No ✅",
                "Accuracy": item["accuracy"],
                "F1 Score": item["f1_score"],
                "Expected Cost / App ↓": item["expected_cost_per_applicant"],
                "DP Diff ↓": item["demographic_parity_difference"],
                "EO Diff (TPR Gap) ↓": item["equal_opportunity_difference"],
                "Equalized Odds Diff ↓": item["equalized_odds_difference"],
                "Approval DI Ratio (≥0.80)": item["disparate_impact_approval_ratio"],
            }
            for item in comp_list
        ]
    )
    st.dataframe(df_mit, use_container_width=True, hide_index=True)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.subheader("Fairness-Accuracy Trade-Off (Equalized Odds Diff vs. Accuracy)")
        fig_trade = go.Figure()
        for item in comp_list:
            fig_trade.add_trace(
                go.Scatter(
                    x=[item["equalized_odds_difference"]],
                    y=[item["accuracy"]],
                    mode="markers+text",
                    name=item["method"],
                    text=[item["paradigm"]],
                    textposition="top center",
                    marker=dict(size=14),
                )
            )
        fig_trade.update_layout(
            xaxis_title="Equalized Odds Difference (Lower = Fairer)",
            yaxis_title="Test Accuracy (Higher = Better)",
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_trade, use_container_width=True)

    with t_col2:
        st.subheader("Demographic Parity Diff Across Mitigation Methods")
        fig_dp = go.Figure(
            go.Bar(
                x=[item["demographic_parity_difference"] for item in comp_list][::-1],
                y=[item["method"] for item in comp_list][::-1],
                orientation="h",
                marker_color="#10B981",
                text=[f"{item['demographic_parity_difference']:.4f}" for item in comp_list][::-1],
                textposition="auto",
            )
        )
        fig_dp.update_layout(
            xaxis_title="Demographic Parity Difference (Lower = More Equal Selection Rates)",
            height=360,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_dp, use_container_width=True)

    st.info(
        f"**⚖️ Legal & Regulatory Trade-Off (Disparate Treatment vs. Disparate Impact):**\n\n"
        f"{fairness_after['regulatory_and_legal_discussion']['disparate_treatment_vs_disparate_impact']}"
    )

    st.markdown("---")

    # Section 3: Proxy Leakage Audit
    st.header("3️⃣ Proxy Leakage Audit — Why 'Fairness Through Unawareness' Fails")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        sex_p = proxy_audit["SEX_proxy"]
        st.subheader(f"Predicting `SEX` from Allowed Features (ROC-AUC = `{sex_p['roc_auc']:.3f}`)")
        st.write(sex_p["interpretation"])
        df_sp = pd.DataFrame(sex_p["top_proxy_features"])
        st.dataframe(df_sp, use_container_width=True, hide_index=True)

    with p_col2:
        age_p = proxy_audit["AGE_GROUP_proxy"]
        st.subheader(
            f"Predicting `AGE_GROUP` from Allowed Features (Macro AUC = `{age_p['macro_ovr_roc_auc']:.3f}`)"
        )
        st.write(age_p["interpretation"])
        df_ap = pd.DataFrame(age_p["top_proxy_features"])
        st.dataframe(df_ap, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
