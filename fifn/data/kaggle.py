"""Preprocessor for the Kaggle vehicle-claim fraud dataset.

Dataset: https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
File:    fraud_oracle.csv  (~15k rows, ~33 columns)

Usage:
    python -m fifn prepare-kaggle --input data/raw/fraud_oracle.csv \\
                                   --output-dir data/nodes/ \\
                                   --nodes 7

The CSV is mapped to FIFN's canonical schema and split across N node
parquet files, partitioned by vehicle Make so each node has a realistic
sub-population of claims.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Maps Kaggle months (Jan, Feb, …) → integer day-of-year (mid-month approx)
_MONTH_TO_DAY = {
    "Jan": 15, "Feb": 46, "Mar": 74, "Apr": 105, "May": 135, "Jun": 166,
    "Jul": 196, "Aug": 227, "Sep": 258, "Oct": 288, "Nov": 319, "Dec": 349,
}

# Node assignment: map Make groups to our 7 named insurers
_NODE_INSURERS = [
    "meridian-auto",
    "coastal-shield",
    "bluepeak-mutual",
    "hartwell-general",
    "lakeside-health",
    "redwood-pc",
    "summit-life",
]


def _map_policy_type(row: pd.Series) -> str:
    """Approximate policy type from vehicle/accident context."""
    vcat = str(row.get("VehicleCategory", "")).lower()
    if "sport" in vcat:
        return "auto"
    area = str(row.get("AccidentArea", "")).lower()
    if "urban" in area:
        return "liability"
    return "auto"


def _map_incident_type(row: pd.Series) -> str:
    base_pol = str(row.get("BasePolicy", "")).lower()
    if "collision" in base_pol:
        return "collision"
    if "all" in base_pol:
        return "theft"
    return "collision"


def _map_contact_channel(row: pd.Series) -> str:
    rep = str(row.get("RepNumber", "0"))
    try:
        n = int(rep)
    except ValueError:
        n = 0
    channels = ["web", "phone", "agent", "walk-in"]
    return channels[n % 4]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Map fraud_oracle.csv columns to FIFN canonical schema."""
    out = pd.DataFrame()

    n = len(df)
    out["claim_id"] = [f"CLM-{i:06d}" for i in range(n)]

    # Assign insurer based on vehicle Make (modular bucketing)
    makes = df["Make"].fillna("Unknown").astype(str)
    unique_makes = sorted(makes.unique())
    make_to_idx = {m: i % len(_NODE_INSURERS) for i, m in enumerate(unique_makes)}
    out["insurer_id"] = makes.map(make_to_idx).map(
        lambda i: _NODE_INSURERS[i]
    )

    # Temporal — Kaggle has MonthClaimed + Year (or just Year)
    year = pd.to_numeric(df.get("Year", pd.Series(2015, index=df.index)), errors="coerce").fillna(2015).astype(int)
    month_claimed = df.get("MonthClaimed", pd.Series("Jan", index=df.index)).fillna("Jan")
    day_of_year = month_claimed.map(_MONTH_TO_DAY).fillna(180).astype(int)
    incident_dates = pd.to_datetime(
        year.astype(str) + "-" + day_of_year.astype(str), format="%Y-%j", errors="coerce"
    ).fillna(pd.Timestamp("2015-06-01"))

    days_to_file = pd.to_numeric(df.get("Days:Policy-Claim", pd.Series(14, index=df.index)), errors="coerce").fillna(14).clip(0, 365).astype(int)
    filed_dates = incident_dates + pd.to_timedelta(days_to_file, unit="D")

    out["incident_date"] = incident_dates.dt.strftime("%Y-%m-%d")
    out["filed_date"]    = filed_dates.dt.strftime("%Y-%m-%d")
    out["days_to_file"]  = days_to_file

    # Policy context
    policy_age_raw = df.get("Days:Policy-Accident", pd.Series(365, index=df.index))
    out["policy_age_days"] = pd.to_numeric(policy_age_raw, errors="coerce").fillna(365).clip(0).astype(int)

    out["policy_type"] = df.apply(_map_policy_type, axis=1)

    deductible = pd.to_numeric(df.get("Deductible", pd.Series(500, index=df.index)), errors="coerce").fillna(500)
    out["coverage_amount"]  = deductible * 20  # rough proxy
    out["premium_monthly"]  = deductible * 0.08

    out["prior_claims_12m"] = pd.to_numeric(
        df.get("PastNumberOfClaims", pd.Series(0, index=df.index)).replace({"none": 0, "1": 1, "2 to 4": 3, "more than 4": 5}),
        errors="coerce",
    ).fillna(0).clip(0, 10).astype(int)

    # Incident
    out["incident_type"]    = df.apply(_map_incident_type, axis=1)
    out["reported_amount"]  = pd.to_numeric(df.get("VehiclePrice", pd.Series(15000, index=df.index)).replace({"less than 20000": 15000, "20000 to 29000": 24500, "30000 to 39000": 34500, "40000 to 59000": 49500, "60000 to 69000": 64500, "more than 69000": 80000}), errors="coerce").fillna(15000)
    out["witnesses"]        = pd.to_numeric(df.get("Witnesses", pd.Series(1, index=df.index)), errors="coerce").fillna(1).clip(0, 5).astype(int)
    out["police_report"]    = df.get("PoliceReportFiled", pd.Series("No", index=df.index)).str.lower().eq("yes")
    out["third_party"]      = df.get("AgentType", pd.Series("Internal", index=df.index)).str.lower().eq("external")

    # Claimant behaviour
    out["contact_channel"]  = df.apply(_map_contact_channel, axis=1)
    out["legal_rep"]        = df.get("AddressChange-Claim", pd.Series("no change", index=df.index)).str.lower().ne("no change")
    out["address_changes_6m"] = pd.to_numeric(
        df.get("NumberOfSuppliments", pd.Series(0, index=df.index)).replace({"none": 0, "1 to 2": 1, "3 to 5": 4, "more than 5": 6}),
        errors="coerce",
    ).fillna(0).clip(0, 10).astype(int)

    # Label
    out["fraud_label"] = pd.to_numeric(df.get("FraudFound_P", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int)

    return out


def prepare_kaggle(
    input_path: Path,
    output_dir: Path,
    n_nodes: int = 7,
    seed: int = 42,
) -> dict[str, Path]:
    """Load fraud_oracle.csv, preprocess, and split into per-node parquet files.

    Returns a dict mapping insurer_id → output path.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_path)
    df = preprocess(df_raw)

    # Limit to the requested number of nodes
    active_insurers = _NODE_INSURERS[:n_nodes]
    df = df[df["insurer_id"].isin(active_insurers)].copy()

    written: dict[str, Path] = {}
    for insurer_id in active_insurers:
        subset = df[df["insurer_id"] == insurer_id].reset_index(drop=True)
        if subset.empty:
            # Fallback: give this node a random slice if no makes mapped here
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df), size=max(100, len(df) // n_nodes), replace=False)
            subset = df.iloc[idx].copy()
            subset["insurer_id"] = insurer_id

        out_path = output_dir / f"{insurer_id}.parquet"
        subset.to_parquet(out_path, index=False)
        written[insurer_id] = out_path
        print(f"  {insurer_id:25s}  {len(subset):5d} claims  "
              f"  fraud rate: {subset['fraud_label'].mean():.1%}  →  {out_path}")

    return written
