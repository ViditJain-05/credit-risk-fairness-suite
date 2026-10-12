"""Centralized configuration for the Credit Risk Explainability & Fairness Suite.

All file paths, random seeds, column definitions, sensitive attribute mappings,
and human-readable Adverse Action templates are defined here so there are zero
hardcoded magic constants scattered across the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 1. Directory & Path Resolution (No hardcoded OS paths)
# ---------------------------------------------------------------------------
# config.py lives at: <PROJECT_ROOT>/src/creditfair/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"


def get_artifact_path(filename: str) -> Path:
    """Return the resolved path to a file inside the artifacts/ directory."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / filename


# ---------------------------------------------------------------------------
# 2. Reproducibility & Split Settings
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
# Stratified Train / Validation / Test split ratios (70% / 15% / 15%)
SPLIT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)

# ---------------------------------------------------------------------------
# 3. Dataset URLs & Identifiers (UCI Default of Credit Card Clients, ID=350)
# ---------------------------------------------------------------------------
UCI_CREDIT_CARD_URLS: List[str] = [
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
]
RAW_FILENAME: str = "uci_credit_card_clients.csv"

# ---------------------------------------------------------------------------
# 4. Target, Sensitive Attributes, and Feature Columns
# ---------------------------------------------------------------------------
TARGET_COL: str = "default"

# Protected / Sensitive attributes kept SEPARATE from model training features
# ("fairness through unawareness" baseline design)
SENSITIVE_COLS: List[str] = ["SEX", "AGE_GROUP", "MARRIAGE"]
RAW_DEMOGRAPHIC_COLS: List[str] = ["SEX", "AGE", "MARRIAGE"]

# Age binning specification: [21-29, 30-39, 40-49, 50+]
AGE_BINS: List[int] = [0, 29, 39, 49, 120]
AGE_LABELS: List[str] = ["21-29", "30-39", "40-49", "50+"]

# Human-readable group label mappings for clean audit tables
SEX_LABELS: Dict[int, str] = {1: "Male", 2: "Female"}
EDUCATION_LABELS: Dict[int, str] = {
    1: "Graduate School",
    2: "University",
    3: "High School",
    4: "Others",
}
MARRIAGE_LABELS: Dict[int, str] = {
    1: "Married",
    2: "Single",
    3: "Others",
}

# Raw non-sensitive features from UCI dataset (after renaming PAY_0 -> PAY_1)
PAY_STATUS_COLS: List[str] = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT_COLS: List[str] = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]
PAY_AMT_COLS: List[str] = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]

BASE_FEATURE_COLS: List[str] = (
    ["LIMIT_BAL", "EDUCATION"] + PAY_STATUS_COLS + BILL_AMT_COLS + PAY_AMT_COLS
)

ENGINEERED_FEATURE_COLS: List[str] = [
    "UTILIZATION_RATIO_1",
    "AVG_UTILIZATION_6M",
    "PAY_TO_BILL_RATIO_1",
    "MAX_DELINQUENCY_6M",
    "MONTHS_DELINQUENT_6M",
    "BILL_GROWTH_6M",
]

ALL_MODEL_FEATURES: List[str] = BASE_FEATURE_COLS + ENGINEERED_FEATURE_COLS

# ---------------------------------------------------------------------------
# 5. Business Cost Matrix & Threshold Configuration
# ---------------------------------------------------------------------------
# Cost of a False Negative (approving a defaulter) vs False Positive (denying a good customer)
DEFAULT_COST_FN: float = 5.0
DEFAULT_COST_FP: float = 1.0
FOUR_FIFTHS_THRESHOLD: float = 0.80

# ---------------------------------------------------------------------------
# 6. Plain-English Feature Descriptions (for SHAP & Adverse Action Notices)
# ---------------------------------------------------------------------------
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "LIMIT_BAL": "Credit limit amount (NT$)",
    "EDUCATION": "Education level (1=Grad, 2=Univ, 3=High School, 4=Other)",
    "PAY_1": "Repayment status in most recent month (Month 1 / Sept)",
    "PAY_2": "Repayment status 2 months ago (Month 2 / Aug)",
    "PAY_3": "Repayment status 3 months ago (Month 3 / July)",
    "PAY_4": "Repayment status 4 months ago (Month 4 / June)",
    "PAY_5": "Repayment status 5 months ago (Month 5 / May)",
    "PAY_6": "Repayment status 6 months ago (Month 6 / April)",
    "BILL_AMT1": "Bill statement balance in most recent month (NT$)",
    "BILL_AMT2": "Bill statement balance 2 months ago (NT$)",
    "BILL_AMT3": "Bill statement balance 3 months ago (NT$)",
    "BILL_AMT4": "Bill statement balance 4 months ago (NT$)",
    "BILL_AMT5": "Bill statement balance 5 months ago (NT$)",
    "BILL_AMT6": "Bill statement balance 6 months ago (NT$)",
    "PAY_AMT1": "Payment amount made in most recent month (NT$)",
    "PAY_AMT2": "Payment amount made 2 months ago (NT$)",
    "PAY_AMT3": "Payment amount made 3 months ago (NT$)",
    "PAY_AMT4": "Payment amount made 4 months ago (NT$)",
    "PAY_AMT5": "Payment amount made 5 months ago (NT$)",
    "PAY_AMT6": "Payment amount made 6 months ago (NT$)",
    "UTILIZATION_RATIO_1": "Recent credit line utilization ratio (Bill 1 / Credit Limit)",
    "AVG_UTILIZATION_6M": "Average 6-month credit utilization ratio",
    "PAY_TO_BILL_RATIO_1": "Recent payment-to-bill coverage ratio",
    "MAX_DELINQUENCY_6M": "Worst repayment delay (months late) over past 6 months",
    "MONTHS_DELINQUENT_6M": "Total number of months with late payment in past 6 months",
    "BILL_GROWTH_6M": "6-month bill balance trend (recent minus 6 months ago)",
}
