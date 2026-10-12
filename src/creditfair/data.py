"""Data loading, quirk cleaning, sensitive attribute separation, and splitting.

Handles the UCI 'Default of Credit Card Clients' dataset (Taiwan, 2005, n=30,000):
1. Attempts `fairlearn.datasets.fetch_credit_card()` or OpenML / UCI direct download.
2. Fixes documented dataset quirks:
   - Renames `PAY_0` -> `PAY_1` for sequential consistency (`PAY_1`..`PAY_6`).
   - Maps undocumented `EDUCATION` codes {0, 5, 6} -> 4 ('Others').
   - Maps undocumented `MARRIAGE` code {0} -> 3 ('Others').
   - Derives `AGE_GROUP` bins: '21-29', '30-39', '40-49', '50+'.
3. Separates sensitive attributes (`SEX`, `AGE_GROUP`, `MARRIAGE`) from training
   features (`X`) to enforce the 'fairness through unawareness' baseline design.
4. Performs a reproducible 70/15/15 stratified train/validation/test split.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from creditfair import config


@dataclass
class DatasetSplits:
    """Container holding separated feature, target, and sensitive attribute splits."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    S_train: pd.DataFrame
    S_val: pd.DataFrame
    S_test: pd.DataFrame
    cleaning_report: Dict[str, Any]


def _standardize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names from Fairlearn/OpenML/UCI XLS into canonical uppercase."""
    df = df.copy()

    # If the first row contains header strings (common in the raw UCI .xls file where row 0 is X1..X23)
    first_val = str(df.iloc[0, 0]).strip().upper() if len(df) > 0 else ""
    if first_val in {"ID", "LIMIT_BAL"}:
        df.columns = [str(c).strip() for c in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)

    col_map: Dict[str, str] = {}
    # Standard X1..X23 mapping in case OpenML returns x1..x23
    x_mapping = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_1",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": config.TARGET_COL,
    }

    for col in df.columns:
        c_clean = str(col).strip()
        c_upper = c_clean.upper()
        if c_upper in x_mapping:
            col_map[col] = x_mapping[c_upper]
        elif c_upper == "PAY_0":
            col_map[col] = "PAY_1"
        elif c_upper in {
            "DEFAULT PAYMENT NEXT MONTH",
            "DEFAULT.PAYMENT.NEXT.MONTH",
            "DEFAULT_PAYMENT_NEXT_MONTH",
            "DEFAULT",
        }:
            col_map[col] = config.TARGET_COL
        else:
            col_map[col] = c_upper

    df = df.rename(columns=col_map)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Convert all canonical columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)
    return df


def fetch_raw_credit_card_data(force_download: bool = False) -> pd.DataFrame:
    """Fetch the UCI Credit Card Default dataset and cache to data/raw/."""
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv_path: Path = config.DATA_RAW_DIR / config.RAW_FILENAME

    if raw_csv_path.exists() and not force_download:
        return pd.read_csv(raw_csv_path)

    df_raw: pd.DataFrame | None = None

    # Attempt 1: fairlearn.datasets.fetch_credit_card
    try:
        from fairlearn.datasets import fetch_credit_card

        bunch = fetch_credit_card(as_frame=True)
        df_candidate = bunch.frame.copy()
        if config.TARGET_COL not in df_candidate.columns and bunch.target is not None:
            df_candidate[config.TARGET_COL] = bunch.target
        df_raw = _standardize_raw_columns(df_candidate)
    except Exception:
        df_raw = None

    # Attempt 2: UCI ZIP / XLS direct download
    if df_raw is None or len(df_raw) < 1000:
        for url in config.UCI_CREDIT_CARD_URLS:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 (CreditFairnessSuite/1.0)"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    content = resp.read()
                if url.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        xls_names = [
                            n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx", ".csv"))
                        ]
                        with zf.open(xls_names[0]) as f_in:
                            data_bytes = f_in.read()
                            if xls_names[0].lower().endswith(".csv"):
                                df_candidate = pd.read_csv(io.BytesIO(data_bytes))
                            else:
                                df_candidate = pd.read_excel(io.BytesIO(data_bytes), header=1)
                else:
                    df_candidate = pd.read_excel(io.BytesIO(content), header=1)

                df_raw = _standardize_raw_columns(df_candidate)
                if len(df_raw) >= 25000:
                    break
            except Exception:
                continue

    # Attempt 3: OpenML fetch via scikit-learn
    if df_raw is None or len(df_raw) < 1000:
        try:
            from sklearn.datasets import fetch_openml

            bunch = fetch_openml(data_id=42477, as_frame=True, parser="auto")
            df_candidate = bunch.frame.copy()
            df_raw = _standardize_raw_columns(df_candidate)
        except Exception:
            df_raw = None

    if df_raw is None or len(df_raw) == 0:
        raise RuntimeError(
            "Unable to download UCI Default of Credit Card Clients dataset from Fairlearn, UCI, or OpenML."
        )

    df_raw.to_csv(raw_csv_path, index=False)
    return df_raw


def clean_credit_card_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean dataset quirks, document anomalies, and derive AGE_GROUP bins.

    Quirks handled:
    - Renames `PAY_0` -> `PAY_1` if not already renamed.
    - Maps undocumented `EDUCATION` codes {0, 5, 6} -> 4 ('Others').
    - Maps undocumented `MARRIAGE` code {0} -> 3 ('Others').
    - Creates `AGE_GROUP` with bins: '21-29', '30-39', '40-49', '50+'.
    """
    df = _standardize_raw_columns(df_raw)

    # Record pre-cleaning anomaly counts for full audit transparency
    edu_undocumented_mask = df["EDUCATION"].isin([0, 5, 6]) | ~df["EDUCATION"].isin([1, 2, 3, 4])
    mar_undocumented_mask = (df["MARRIAGE"] == 0) | ~df["MARRIAGE"].isin([1, 2, 3])

    edu_remapped_count = int(edu_undocumented_mask.sum())
    mar_remapped_count = int(mar_undocumented_mask.sum())

    edu_raw_counts = {str(int(k)): int(v) for k, v in df["EDUCATION"].value_counts().items()}
    mar_raw_counts = {str(int(k)): int(v) for k, v in df["MARRIAGE"].value_counts().items()}

    # Apply quirk fixes
    df.loc[edu_undocumented_mask, "EDUCATION"] = 4
    df.loc[mar_undocumented_mask, "MARRIAGE"] = 3

    df["SEX"] = df["SEX"].astype(int)
    df["EDUCATION"] = df["EDUCATION"].astype(int)
    df["MARRIAGE"] = df["MARRIAGE"].astype(int)
    df["AGE"] = df["AGE"].astype(int)

    # Derive AGE_GROUP bins: 21-29, 30-39, 40-49, 50+
    df["AGE_GROUP"] = pd.cut(
        df["AGE"],
        bins=config.AGE_BINS,
        labels=config.AGE_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)

    # Add human-readable sensitive attribute labels for clear reporting
    df["SEX_LABEL"] = df["SEX"].map(config.SEX_LABELS).fillna("Unknown")
    df["MARRIAGE_LABEL"] = df["MARRIAGE"].map(config.MARRIAGE_LABELS).fillna("Others")
    df["EDUCATION_LABEL"] = df["EDUCATION"].map(config.EDUCATION_LABELS).fillna("Others")

    cleaning_report: Dict[str, Any] = {
        "total_rows": int(len(df)),
        "default_rate": float(df[config.TARGET_COL].mean()),
        "default_count": int(df[config.TARGET_COL].sum()),
        "non_default_count": int((df[config.TARGET_COL] == 0).sum()),
        "education_undocumented_remapped_to_4": edu_remapped_count,
        "marriage_undocumented_remapped_to_3": mar_remapped_count,
        "education_raw_value_counts": edu_raw_counts,
        "marriage_raw_value_counts": mar_raw_counts,
        "renamed_columns": {"PAY_0": "PAY_1", "default payment next month": "default"},
    }

    return df, cleaning_report


def compute_group_base_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute base default rates across SEX, AGE_GROUP, MARRIAGE, and SEX x AGE_GROUP."""
    base_rates: Dict[str, Any] = {}

    # 1. By SEX (using human-readable label)
    sex_stats = (
        df.groupby("SEX_LABEL", observed=False)[config.TARGET_COL]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "default_rate"})
    )
    base_rates["SEX"] = {
        str(idx): {"n": int(row["n"]), "default_rate": round(float(row["default_rate"]), 4)}
        for idx, row in sex_stats.iterrows()
    }

    # 2. By AGE_GROUP
    age_stats = (
        df.groupby("AGE_GROUP", observed=False)[config.TARGET_COL]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "default_rate"})
    )
    base_rates["AGE_GROUP"] = {
        str(idx): {"n": int(row["n"]), "default_rate": round(float(row["default_rate"]), 4)}
        for idx, row in age_stats.iterrows()
    }

    # 3. By MARRIAGE
    mar_stats = (
        df.groupby("MARRIAGE_LABEL", observed=False)[config.TARGET_COL]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "default_rate"})
    )
    base_rates["MARRIAGE"] = {
        str(idx): {"n": int(row["n"]), "default_rate": round(float(row["default_rate"]), 4)}
        for idx, row in mar_stats.iterrows()
    }

    # 4. Intersectional: SEX_LABEL x AGE_GROUP
    inter_col = df["SEX_LABEL"].astype(str) + " | " + df["AGE_GROUP"].astype(str)
    inter_stats = (
        df.groupby(inter_col, observed=False)[config.TARGET_COL]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "default_rate"})
    )
    base_rates["SEX_x_AGE_GROUP"] = {
        str(idx): {"n": int(row["n"]), "default_rate": round(float(row["default_rate"]), 4)}
        for idx, row in inter_stats.iterrows()
    }

    return base_rates


def prepare_splits(
    df_clean: pd.DataFrame | None = None,
    save_processed: bool = True,
) -> DatasetSplits:
    """Create stratified 70/15/15 train/val/test splits with sensitive attributes separated.

    Sensitive columns (`SEX`, `AGE`, `AGE_GROUP`, `MARRIAGE`) are stored in `S_*`
    and strictly excluded from `X_*`.
    """
    from creditfair.features import engineer_features

    if df_clean is None:
        df_raw = fetch_raw_credit_card_data()
        df_clean, cleaning_report = clean_credit_card_data(df_raw)
    else:
        _, cleaning_report = clean_credit_card_data(df_clean)

    # Engineer domain features (row-wise deterministic ratios/summaries; zero cross-row leakage)
    df_feat = engineer_features(df_clean)

    X_all = df_feat[config.ALL_MODEL_FEATURES].copy()
    y_all = df_feat[config.TARGET_COL].copy()

    # Verify zero sensitive columns leaked into X_all
    forbidden = set(config.SENSITIVE_COLS + config.RAW_DEMOGRAPHIC_COLS)
    leaked = forbidden.intersection(set(X_all.columns))
    if leaked:
        raise ValueError(f"Sensitive columns leaked into model feature matrix X: {leaked}")

    S_all = pd.DataFrame(
        {
            "SEX": df_clean["SEX_LABEL"].astype(str),
            "SEX_CODE": df_clean["SEX"].astype(int),
            "AGE": df_clean["AGE"].astype(int),
            "AGE_GROUP": df_clean["AGE_GROUP"].astype(str),
            "MARRIAGE": df_clean["MARRIAGE_LABEL"].astype(str),
            "MARRIAGE_CODE": df_clean["MARRIAGE"].astype(int),
            "SEX_x_AGE_GROUP": df_clean["SEX_LABEL"].astype(str)
            + " | "
            + df_clean["AGE_GROUP"].astype(str),
        },
        index=df_clean.index,
    )

    # First split: 70% Train, 30% Temp (stratified by target y)
    train_ratio, val_ratio, test_ratio = config.SPLIT_RATIOS
    temp_ratio = val_ratio + test_ratio

    X_train, X_temp, y_train, y_temp, S_train, S_temp = train_test_split(
        X_all,
        y_all,
        S_all,
        test_size=temp_ratio,
        stratify=y_all,
        random_state=config.RANDOM_SEED,
    )

    # Second split: split 30% Temp into 15% Val and 15% Test (50/50 of Temp)
    relative_test_ratio = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test, S_val, S_test = train_test_split(
        X_temp,
        y_temp,
        S_temp,
        test_size=relative_test_ratio,
        stratify=y_temp,
        random_state=config.RANDOM_SEED,
    )

    base_rates = compute_group_base_rates(df_clean)
    cleaning_report["group_base_rates"] = base_rates
    cleaning_report["split_sizes"] = {
        "train": int(len(X_train)),
        "val": int(len(X_val)),
        "test": int(len(X_test)),
    }
    cleaning_report["split_default_rates"] = {
        "train": round(float(y_train.mean()), 4),
        "val": round(float(y_val.mean()), 4),
        "test": round(float(y_test.mean()), 4),
    }

    # Feature correlations with target on training set
    train_corrs = (
        X_train.apply(lambda col: float(col.corr(y_train)))
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .round(4)
        .to_dict()
    )
    cleaning_report["feature_target_correlations_train"] = train_corrs

    if save_processed:
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        X_train.to_parquet(config.DATA_PROCESSED_DIR / "X_train.parquet")
        X_val.to_parquet(config.DATA_PROCESSED_DIR / "X_val.parquet")
        X_test.to_parquet(config.DATA_PROCESSED_DIR / "X_test.parquet")
        y_train.to_frame(name=config.TARGET_COL).to_parquet(
            config.DATA_PROCESSED_DIR / "y_train.parquet"
        )
        y_val.to_frame(name=config.TARGET_COL).to_parquet(
            config.DATA_PROCESSED_DIR / "y_val.parquet"
        )
        y_test.to_frame(name=config.TARGET_COL).to_parquet(
            config.DATA_PROCESSED_DIR / "y_test.parquet"
        )
        S_train.to_parquet(config.DATA_PROCESSED_DIR / "S_train.parquet")
        S_val.to_parquet(config.DATA_PROCESSED_DIR / "S_val.parquet")
        S_test.to_parquet(config.DATA_PROCESSED_DIR / "S_test.parquet")

        eda_path = config.get_artifact_path("eda_summary.json")
        with open(eda_path, "w", encoding="utf-8") as f:
            json.dump(cleaning_report, f, indent=2)

    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        S_train=S_train,
        S_val=S_val,
        S_test=S_test,
        cleaning_report=cleaning_report,
    )


def load_saved_splits() -> DatasetSplits:
    """Load pre-saved parquet splits from data/processed/ if available, else generate them."""
    req_files = [
        "X_train.parquet",
        "X_val.parquet",
        "X_test.parquet",
        "y_train.parquet",
        "y_val.parquet",
        "y_test.parquet",
        "S_train.parquet",
        "S_val.parquet",
        "S_test.parquet",
    ]
    if all((config.DATA_PROCESSED_DIR / f).exists() for f in req_files):
        X_train = pd.read_parquet(config.DATA_PROCESSED_DIR / "X_train.parquet")
        X_val = pd.read_parquet(config.DATA_PROCESSED_DIR / "X_val.parquet")
        X_test = pd.read_parquet(config.DATA_PROCESSED_DIR / "X_test.parquet")
        y_train = pd.read_parquet(config.DATA_PROCESSED_DIR / "y_train.parquet")[config.TARGET_COL]
        y_val = pd.read_parquet(config.DATA_PROCESSED_DIR / "y_val.parquet")[config.TARGET_COL]
        y_test = pd.read_parquet(config.DATA_PROCESSED_DIR / "y_test.parquet")[config.TARGET_COL]
        S_train = pd.read_parquet(config.DATA_PROCESSED_DIR / "S_train.parquet")
        S_val = pd.read_parquet(config.DATA_PROCESSED_DIR / "S_val.parquet")
        S_test = pd.read_parquet(config.DATA_PROCESSED_DIR / "S_test.parquet")

        eda_path = config.get_artifact_path("eda_summary.json")
        cleaning_report = {}
        if eda_path.exists():
            with open(eda_path, "r", encoding="utf-8") as f:
                cleaning_report = json.load(f)

        return DatasetSplits(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            S_train=S_train,
            S_val=S_val,
            S_test=S_test,
            cleaning_report=cleaning_report,
        )

    return prepare_splits(save_processed=True)
