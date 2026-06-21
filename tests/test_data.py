import tempfile
from pathlib import Path

import numpy as np
import pytest

from fifn.data.synthetic import generate_claims
from fifn.data.loader import load_claims_features


def test_generate_claims_shape():
    df = generate_claims(n_samples=200, fraud_rate=0.1, seed=0)
    assert len(df) == 200
    assert "fraud_label" in df.columns
    assert df["fraud_label"].isin([0, 1]).all()


def test_generate_claims_fraud_rate():
    df = generate_claims(n_samples=2000, fraud_rate=0.1, seed=42)
    actual_rate = df["fraud_label"].mean()
    assert 0.05 < actual_rate < 0.20, f"Unexpected fraud rate: {actual_rate}"


def test_generate_claims_to_parquet():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "claims.parquet"
        df = generate_claims(n_samples=100, output_path=out)
        assert out.exists()
        assert len(df) == 100


def test_load_claims_features():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "claims.parquet"
        generate_claims(n_samples=300, output_path=out)
        X, y = load_claims_features(out)
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(X) == len(y) == 300
        assert X.dtype == np.float64 or X.dtype == np.float32
        # StandardScaler output: mean ~ 0, std ~ 1 for numeric cols
        assert np.abs(X[:, 0].mean()) < 1.0


def test_load_claims_features_no_pii():
    """Feature matrix must not contain claim_id or insurer_id (string) columns."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "claims.parquet"
        generate_claims(n_samples=100, output_path=out)
        X, y = load_claims_features(out)
        # All values must be numeric (no NaN after fillna)
        assert not np.any(np.isnan(X))
