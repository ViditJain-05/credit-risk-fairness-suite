"""Generate high-resolution visual charts in `assets/` from `artifacts/*.json` for README.md."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from creditfair import config


def _load(name: str) -> dict:
    with open(config.get_artifact_path(name), "r", encoding="utf-8") as f:
        return json.load(f)


def generate_all_charts() -> None:
    assets_dir = config.PROJECT_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
    )

    metrics = _load("metrics.json")
    explain = _load("explainability_summary.json")
    fb = _load("fairness_before.json")
    fa = _load("fairness_after.json")

    # -------------------------------------------------------------------------
    # Chart 1: Calibration Curve & Model Ranking Benchmark
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", lw=1.3, label="Perfectly Calibrated (Diagonal)")
    colors_map = {
        "XGBoost (Weighted, Uncalibrated)": ("#EF4444", "s--"),
        "Logistic Regression (Scaled)": ("#64748B", "^-."),
        "HistGradientBoosting": ("#F59E0B", "d-"),
        "XGBoost (Calibrated)": ("#0284C7", "o-"),
    }
    for name, (col, style) in colors_map.items():
        m = metrics["models_test_set"][name]
        cal = m["calibration_curve"]
        ax1.plot(
            cal["prob_pred"],
            cal["prob_true"],
            style,
            color=col,
            lw=2.2 if "Calibrated" in name else 1.6,
            markersize=6,
            label=f"{name} (Brier={m['brier_score']:.4f})",
        )
    ax1.set_title(
        "Probability Calibration Curve (Reliability Diagram)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Mean Predicted Default Probability", fontsize=10.5)
    ax1.set_ylabel("Observed Empirical Default Rate", fontsize=10.5)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="#F8FAFC")

    ax2 = axes[1]
    model_names = [
        "LogReg\n(Scaled)",
        "HistGBDT",
        "XGBoost\n(Uncalibrated)",
        "XGBoost\n(Calibrated)",
    ]
    model_keys = [
        "Logistic Regression (Scaled)",
        "HistGradientBoosting",
        "XGBoost (Weighted, Uncalibrated)",
        "XGBoost (Calibrated)",
    ]
    x = np.arange(len(model_names))
    width = 0.25
    aucs = [metrics["models_test_set"][k]["roc_auc"] for k in model_keys]
    ginis = [metrics["models_test_set"][k]["gini"] for k in model_keys]
    kss = [metrics["models_test_set"][k]["ks_statistic"] for k in model_keys]

    b1 = ax2.bar(x - width, aucs, width, label="ROC-AUC", color="#0284C7")
    b2 = ax2.bar(x, ginis, width, label="Gini (2*AUC-1)", color="#6366F1")
    b3 = ax2.bar(x + width, kss, width, label="KS Statistic", color="#10B981")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    ax2.set_title(
        "Held-Out Test Set Credit Metrics (N = 4,500)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=9.5)
    ax2.set_ylim(0, 0.92)
    ax2.set_ylabel("Metric Score", fontsize=10.5)
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper left", fontsize=9, frameon=True, facecolor="#F8FAFC")

    fig.tight_layout()
    fig.savefig(assets_dir / "calibration_and_ks_benchmark.png", dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Chart 2: Global SHAP Importance & Individual Adverse Action Waterfall
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    ax1 = axes[0]
    top10 = explain["global_feature_importance"][:10][::-1]
    feats = [item["feature"] for item in top10]
    vals = [item["mean_abs_shap"] for item in top10]
    bars = ax1.barh(feats, vals, color="#0F766E", height=0.68)
    for bar in bars:
        w = bar.get_width()
        ax1.text(
            w + 0.006,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.set_title(
        "Global TreeSHAP Importance (Top 10 Features)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlabel("Mean |SHAP| Impact on Log-Odds", fontsize=10.5)
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.grid(True, axis="x")

    ax2 = axes[1]
    denied = next(
        (s for s in explain["sample_explanations"] if s["decision"] == "DENY"),
        explain["sample_explanations"][0],
    )
    shap_items = sorted(
        denied["shap_values"].items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:8][::-1]
    s_feats = [k for k, _ in shap_items]
    s_vals = [v for _, v in shap_items]
    s_cols = ["#DC2626" if v > 0 else "#2563EB" for v in s_vals]
    bars2 = ax2.barh(s_feats, s_vals, color=s_cols, height=0.68)
    ax2.axvline(0, color="#1E293B", lw=1.1)
    for bar, v in zip(bars2, s_vals):
        offset = 0.012 if v >= 0 else -0.012
        ha = "left" if v >= 0 else "right"
        ax2.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.3f}",
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_title(
        f"Local Adverse Action Decomposition (P(Default)={denied['default_probability']:.1%}, Cutoff={denied['threshold']:.1%})",
        fontsize=11.2,
        fontweight="bold",
        pad=10,
    )
    ax2.set_xlabel(
        "SHAP Log-Odds Contribution (Red = Pushes to Deny, Blue = Protective)",
        fontsize=10,
    )
    ax2.set_xlim(min(s_vals) - 0.12, max(s_vals) + 0.12)
    ax2.grid(True, axis="x")

    fig.tight_layout()
    fig.savefig(assets_dir / "shap_and_adverse_action.png", dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Chart 3: Fairlearn Subgroup Audit & Proxy Leakage
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    ax1 = axes[0]
    age_groups = list(fb["audits"]["AGE_GROUP"]["by_group"].keys())
    denial_rates = [
        fb["audits"]["AGE_GROUP"]["by_group"][g]["selection_rate_denial"]
        for g in age_groups
    ]
    tprs = [fb["audits"]["AGE_GROUP"]["by_group"][g]["tpr"] for g in age_groups]
    fprs = [fb["audits"]["AGE_GROUP"]["by_group"][g]["fpr"] for g in age_groups]
    base_rates = [
        fb["audits"]["AGE_GROUP"]["by_group"][g]["base_default_rate"]
        for g in age_groups
    ]

    x = np.arange(len(age_groups))
    w = 0.2
    ax1.bar(x - 1.5 * w, base_rates, w, label="Base Default Rate", color="#64748B")
    ax1.bar(
        x - 0.5 * w, denial_rates, w, label="Denial Selection Rate", color="#DC2626"
    )
    ax1.bar(x + 0.5 * w, tprs, w, label="True Positive Rate (TPR)", color="#0284C7")
    ax1.bar(
        x + 1.5 * w,
        fprs,
        w,
        label="False Positive Rate (Wrongful Denial)",
        color="#F59E0B",
    )
    ax1.set_title(
        "Fairlearn MetricFrame Audit by AGE_GROUP (Unmitigated)",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(age_groups, fontsize=10)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("Rate", fontsize=10.5)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="#F8FAFC")

    ax2 = axes[1]
    proxy_labels = [
        "Random Chance\nBaseline",
        "Predict SEX\nfrom Allowed X",
        "Predict AGE_GROUP\nfrom Allowed X",
    ]
    proxy_aucs = [
        0.5000,
        explain["proxy_audit"]["SEX_proxy"]["roc_auc"],
        explain["proxy_audit"]["AGE_GROUP_proxy"]["macro_ovr_roc_auc"],
    ]
    p_cols = ["#94A3B8", "#7C3AED", "#DB2777"]
    bars_p = ax2.bar(proxy_labels, proxy_aucs, color=p_cols, width=0.48)
    ax2.axhline(0.5, color="#334155", linestyle="--", lw=1.2)
    for bar in bars_p:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012,
            f"AUC = {h:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_title(
        'Proxy Leakage Audit: Why "Unawareness" Fails',
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax2.set_ylim(0, 0.78)
    ax2.set_ylabel("Held-Out ROC-AUC", fontsize=10.5)
    ax2.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(assets_dir / "fairness_and_proxy_audit.png", dpi=200)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Chart 4: Bias Mitigation Trade-Off (Accuracy & Disparate Impact vs. DP Gap)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.3), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")

    short_names = [
        "Baseline\n(XGBoost)",
        "Sample\nReweighing",
        "ExpGrad\n(DP)",
        "ExpGrad\n(EO)",
        "ThreshOpt\n(DP)*",
        "ThreshOpt\n(EO)*",
    ]

    for idx, attr in enumerate(["SEX", "AGE_GROUP"]):
        ax = axes[idx]
        comp = fa["comparisons"][attr]
        dp_diffs = [m["demographic_parity_difference"] for m in comp]
        eod_diffs = [m["equalized_odds_difference"] for m in comp]
        accs = [m["accuracy"] for m in comp]

        x = np.arange(len(short_names))
        w = 0.26
        b_dp = ax.bar(
            x - w, dp_diffs, w, label="Demographic Parity Diff ↓", color="#EF4444"
        )
        b_eo = ax.bar(
            x, eod_diffs, w, label="Equalized Odds Diff ↓", color="#F59E0B"
        )
        b_ac = ax.bar(x + w, accs, w, label="Test Accuracy ↑", color="#0284C7")

        for bar in b_dp:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
            )
        for bar in b_ac:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
            )

        ax.set_title(
            f"Bias Mitigation Trade-Off Benchmark ({attr})",
            fontsize=11.5,
            fontweight="bold",
            pad=10,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=8.8)
        ax.set_ylim(0, 0.96)
        ax.set_ylabel("Metric Value", fontsize=10)
        ax.grid(True, axis="y")
        ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="#F8FAFC")

    fig.tight_layout()
    fig.savefig(assets_dir / "bias_mitigation_tradeoff.png", dpi=200)
    plt.close(fig)
    print(f"Generated 4 high-resolution visual charts in {assets_dir}")


if __name__ == "__main__":
    generate_all_charts()
