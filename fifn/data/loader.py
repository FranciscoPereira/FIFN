from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fifn.data.schema import NUMERIC_COLS, BOOL_COLS, CAT_COLS


def load_claims_features(
    data_path: Path,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a claims parquet file and return (X, y) ready for training.

    Dollar amounts are log-normalised before scaling to reduce cross-insurer
    distribution shift without requiring any data sharing.
    """
    df = pd.read_parquet(data_path)

    for col in ["coverage_amount", "premium_monthly", "reported_amount"]:
        df[col] = np.log1p(df[col])

    num   = df[NUMERIC_COLS].fillna(0).values.astype(float)
    bools = df[BOOL_COLS].astype(float).values
    cats  = pd.get_dummies(df[CAT_COLS], drop_first=True).values.astype(float)

    X = np.hstack([num, bools, cats])

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    y = df["fraud_label"].values.astype(float)
    return X, y
