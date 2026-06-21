"""Tests for the Kaggle dataset preprocessor (no real CSV required)."""
import io
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from fifn.data.kaggle import preprocess, prepare_kaggle
from fifn.data.loader import load_claims_features


def _make_fake_csv(n: int = 50) -> pd.DataFrame:
    """Minimal fake fraud_oracle.csv with the columns we read."""
    rng = np.random.default_rng(0)
    makes = rng.choice(["Toyota", "Honda", "Ford", "BMW", "Audi", "Nissan", "Mazda"], n)
    return pd.DataFrame({
        "PolicyNumber":          range(n),
        "Make":                  makes,
        "Year":                  rng.choice([2013, 2014, 2015], n),
        "MonthClaimed":          rng.choice(["Jan", "Mar", "Jun", "Sep", "Dec"], n),
        "Days:Policy-Accident":  rng.integers(0, 730, n),
        "Days:Policy-Claim":     rng.integers(0, 60, n),
        "Deductible":            rng.choice([300, 400, 500, 700], n),
        "PastNumberOfClaims":    rng.choice(["none", "1", "2 to 4"], n),
        "VehicleCategory":       rng.choice(["Sport", "Utility", "Sedan"], n),
        "AccidentArea":          rng.choice(["Urban", "Rural"], n),
        "BasePolicy":            rng.choice(["Collision", "All Perils", "Liability"], n),
        "VehiclePrice":          rng.choice(["less than 20000", "20000 to 29000", "30000 to 39000"], n),
        "Witnesses":             rng.integers(0, 4, n),
        "PoliceReportFiled":     rng.choice(["Yes", "No"], n),
        "AgentType":             rng.choice(["Internal", "External"], n),
        "RepNumber":             rng.integers(1, 16, n),
        "AddressChange-Claim":   rng.choice(["no change", "1 year", "2 to 3 years"], n),
        "NumberOfSuppliments":   rng.choice(["none", "1 to 2", "3 to 5"], n),
        "FraudFound_P":          rng.choice([0, 1], n, p=[0.94, 0.06]),
    })


def test_preprocess_shape_and_columns():
    df_raw = _make_fake_csv(100)
    df = preprocess(df_raw)
    expected_cols = [
        "claim_id", "insurer_id", "incident_date", "filed_date", "days_to_file",
        "policy_age_days", "policy_type", "coverage_amount", "premium_monthly",
        "prior_claims_12m", "incident_type", "reported_amount", "witnesses",
        "police_report", "third_party", "contact_channel", "legal_rep",
        "address_changes_6m", "fraud_label",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 100


def test_preprocess_no_nulls():
    df = preprocess(_make_fake_csv(200))
    assert df.isnull().sum().sum() == 0


def test_preprocess_label_binary():
    df = preprocess(_make_fake_csv(200))
    assert set(df["fraud_label"].unique()).issubset({0, 1})


def test_preprocess_insurer_ids_valid():
    from fifn.data.kaggle import _NODE_INSURERS
    df = preprocess(_make_fake_csv(200))
    assert df["insurer_id"].isin(_NODE_INSURERS).all()


def test_prepare_kaggle_writes_parquets(tmp_path):
    csv_path = tmp_path / "fraud_oracle.csv"
    _make_fake_csv(200).to_csv(csv_path, index=False)

    written = prepare_kaggle(csv_path, tmp_path / "nodes", n_nodes=3, seed=0)
    assert len(written) == 3
    for path in written.values():
        assert path.exists()
        df = pd.read_parquet(path)
        assert "fraud_label" in df.columns
        assert len(df) > 0


def test_prepare_kaggle_loader_compatible(tmp_path):
    """Ensure preprocessed parquet can be loaded by load_claims_features."""
    csv_path = tmp_path / "fraud_oracle.csv"
    _make_fake_csv(300).to_csv(csv_path, index=False)

    written = prepare_kaggle(csv_path, tmp_path / "nodes", n_nodes=2, seed=0)
    for path in written.values():
        X, y = load_claims_features(path)
        assert X.ndim == 2
        assert X.shape[0] == y.shape[0]
        assert not np.isnan(X).any()
