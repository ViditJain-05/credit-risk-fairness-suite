"""Domain feature engineering for credit risk modeling.

All engineered features are computed purely row-wise (within a single applicant's
6-month billing and payment history). Because no cross-applicant statistics are used
here, there is zero train-to-test data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditfair import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute row-wise credit risk features from 6-month billing and repayment history.

    Engineered Features:
    - `UTILIZATION_RATIO_1`: Most recent bill balance divided by credit limit.
    - `AVG_UTILIZATION_6M`: 6-month mean bill balance divided by credit limit.
    - `PAY_TO_BILL_RATIO_1`: Most recent payment divided by most recent positive bill.
    - `MAX_DELINQUENCY_6M`: Maximum months past due across PAY_1..PAY_6.
    - `MONTHS_DELINQUENT_6M`: Count of months (out of 6) where repayment was delayed (PAY_k > 0).
    - `BILL_GROWTH_6M`: Recent bill minus bill 6 months ago, normalized by credit limit.
    """
    out = df.copy()
    limit_safe = np.maximum(out["LIMIT_BAL"].astype(float), 1.0)

    # 1. Recent & 6-month average credit utilization
    out["UTILIZATION_RATIO_1"] = np.clip(
        out["BILL_AMT1"].astype(float) / limit_safe, -1.0, 5.0
    )
    mean_bill_6m = out[config.BILL_AMT_COLS].astype(float).mean(axis=1)
    out["AVG_UTILIZATION_6M"] = np.clip(mean_bill_6m / limit_safe, -1.0, 5.0)

    # 2. Recent payment-to-bill ratio
    bill1_pos = np.maximum(out["BILL_AMT1"].astype(float), 1.0)
    out["PAY_TO_BILL_RATIO_1"] = np.clip(
        out["PAY_AMT1"].astype(float) / bill1_pos, 0.0, 5.0
    )

    # 3. Delinquency severity and frequency over 6 months
    pay_matrix = out[config.PAY_STATUS_COLS].astype(float)
    out["MAX_DELINQUENCY_6M"] = np.maximum(pay_matrix.max(axis=1), 0.0)
    out["MONTHS_DELINQUENT_6M"] = (pay_matrix > 0).sum(axis=1).astype(float)

    # 4. 6-month balance growth relative to credit limit
    out["BILL_GROWTH_6M"] = np.clip(
        (out["BILL_AMT1"].astype(float) - out["BILL_AMT6"].astype(float)) / limit_safe,
        -5.0,
        5.0,
    )

    return out
