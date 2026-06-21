"""Synthetic claims data generator for testing and demos."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_claims(
    n_samples: int = 1000,
    fraud_rate: float = 0.08,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate a realistic synthetic claims dataset.

    Fraudulent claims exhibit: higher reported amounts, faster filing,
    more legal representation, and more address changes.
    """
    from typing import Optional  # noqa: F401 — forward ref used above

    rng = np.random.default_rng(seed)
    n = n_samples
    fraud = rng.random(n) < fraud_rate

    policy_types = rng.choice(["auto", "health", "property", "liability"], n)
    incident_types = rng.choice(["collision", "theft", "medical", "fire", "other"], n)
    contact_channels = rng.choice(["web", "phone", "agent", "walk-in"], n)

    # Base values
    coverage = rng.lognormal(mean=10.5, sigma=1.2, size=n)
    premium = coverage / rng.uniform(50, 200, n)
    policy_age = rng.integers(30, 3650, n)
    prior_claims = rng.integers(0, 5, n)
    reported = rng.lognormal(mean=8.5, sigma=1.5, size=n)
    days_to_file = rng.integers(0, 90, n)
    witnesses = rng.integers(0, 4, n)
    address_changes = rng.integers(0, 3, n)

    # Fraud signals: inflate reported amount, faster filing, more legal rep, address changes
    reported = np.where(fraud, reported * rng.uniform(1.5, 4.0, n), reported)
    days_to_file = np.where(fraud, rng.integers(0, 10, n), days_to_file)
    address_changes = np.where(fraud, rng.integers(2, 6, n), address_changes)
    police_report = rng.random(n) > np.where(fraud, 0.7, 0.3)
    third_party = rng.random(n) > np.where(fraud, 0.4, 0.6)
    legal_rep = rng.random(n) > np.where(fraud, 0.3, 0.85)

    incident_dates = pd.date_range("2023-01-01", periods=n, freq="6h")
    filed_dates = incident_dates + pd.to_timedelta(days_to_file, unit="D")

    df = pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:06d}" for i in range(n)],
            "insurer_id": "synthetic",
            "incident_date": incident_dates.strftime("%Y-%m-%d"),
            "filed_date": filed_dates.strftime("%Y-%m-%d"),
            "days_to_file": days_to_file,
            "policy_age_days": policy_age,
            "policy_type": policy_types,
            "coverage_amount": coverage,
            "premium_monthly": premium,
            "prior_claims_12m": prior_claims,
            "incident_type": incident_types,
            "reported_amount": reported,
            "witnesses": witnesses,
            "police_report": police_report,
            "third_party": third_party,
            "contact_channel": contact_channels,
            "legal_rep": legal_rep,
            "address_changes_6m": address_changes,
            "fraud_label": fraud.astype(int),
        }
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    return df
