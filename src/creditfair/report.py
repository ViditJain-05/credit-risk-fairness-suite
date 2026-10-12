"""End-to-end pipeline orchestrator that generates all artifacts in `artifacts/`.

Running `python -m creditfair.report` executes:
- Phase 1: Data fetching, cleaning, AGE_GROUP derivation, and 70/15/15 splitting
- Phase 2: Baseline training (LogReg, HistGBDT, XGBoost), Isotonic calibration,
  and validation cost-optimal thresholding
- Phase 3: Global & local SHAP explainability, stability audit, and proxy leakage audit
- Phase 4: Fairlearn MetricFrame subgroup audit with 95% bootstrap confidence intervals
- Phase 5: Bias mitigation comparison (Reweighing, ExponentiatedGradient, ThresholdOptimizer)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from creditfair import config
from creditfair.data import prepare_splits
from creditfair.explain import run_explainability_pipeline
from creditfair.fairness import run_fairness_audit
from creditfair.mitigate import run_mitigation_pipeline
from creditfair.train import train_and_evaluate_all


def generate_all_artifacts() -> Dict[str, Any]:
    """Execute Phases 1–5 in sequence and persist all artifacts to `artifacts/`."""
    t0 = time.time()
    print("[Phase 1] Fetching, cleaning, and splitting UCI Credit Card dataset...")
    splits = prepare_splits(save_processed=True)
    print(
        f"  -> Cleaned {splits.cleaning_report['total_rows']:,} rows "
        f"(Default rate: {splits.cleaning_report['default_rate']:.2%})"
    )

    print("[Phase 2] Training models, calibrating probabilities, and selecting cost threshold...")
    metrics_payload = train_and_evaluate_all(splits=splits)
    xgb_cal_metrics = metrics_payload["models_test_set"]["XGBoost (Calibrated)"]
    print(
        f"  -> Calibrated XGBoost Test ROC-AUC: {xgb_cal_metrics['roc_auc']:.4f} | "
        f"Gini: {xgb_cal_metrics['gini']:.4f} | KS: {xgb_cal_metrics['ks_statistic']:.4f} | "
        f"Optimal Threshold: {metrics_payload['optimal_threshold']:.4f}"
    )

    print("[Phase 3] Computing SHAP values, Adverse Action reasons, stability, and proxy audit...")
    explain_payload = run_explainability_pipeline(splits=splits, sample_size=500)
    print(
        f"  -> Top feature: {explain_payload['global_feature_importance'][0]['feature']} | "
        f"Neighbor Spearman stability: {explain_payload['stability_audit']['mean_neighbor_spearman_correlation']:.4f} | "
        f"SEX proxy AUC: {explain_payload['proxy_audit']['SEX_proxy']['roc_auc']:.4f}"
    )

    print("[Phase 4] Running Fairlearn MetricFrame audit with 95% bootstrap CIs...")
    fairness_before = run_fairness_audit(splits=splits, n_boot=200)
    sex_summary = fairness_before["audits"]["SEX"]["summary_metrics"]
    age_summary = fairness_before["audits"]["AGE_GROUP"]["summary_metrics"]
    print(
        f"  -> SEX DP Diff: {sex_summary['demographic_parity_difference']:.4f}, "
        f"Approval DI Ratio: {sex_summary['disparate_impact_approval_ratio']:.4f}"
    )
    print(
        f"  -> AGE_GROUP DP Diff: {age_summary['demographic_parity_difference']:.4f}, "
        f"Approval DI Ratio: {age_summary['disparate_impact_approval_ratio']:.4f}"
    )

    print("[Phase 5] Running bias mitigation techniques & quantifying trade-offs...")
    fairness_after = run_mitigation_pipeline(splits=splits)

    elapsed = round(time.time() - t0, 2)
    manifest = {
        "status": "completed",
        "elapsed_seconds": elapsed,
        "random_seed": config.RANDOM_SEED,
        "generated_files": [
            "eda_summary.json",
            "model.joblib",
            "metrics.json",
            "shap_background.parquet",
            "shap_values_sample.parquet",
            "explainability_summary.json",
            "proxy_audit.json",
            "fairness_before.json",
            "fairness_after.json",
        ],
    }
    with open(config.get_artifact_path("manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Done] All artifacts generated in {elapsed:.1f}s at {config.ARTIFACTS_DIR}")
    return manifest


if __name__ == "__main__":
    generate_all_artifacts()
