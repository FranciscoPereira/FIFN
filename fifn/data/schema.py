from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ClaimStatus(str, Enum):
    PENDING  = "pending"
    APPROVED = "approved"
    DENIED   = "denied"
    FLAGGED  = "flagged"


@dataclass
class Claim:
    # Identifiers — never shared outside the node
    claim_id:           str
    insurer_id:         str

    # Temporal
    incident_date:      str   # ISO 8601
    filed_date:         str
    days_to_file:       int

    # Policy context
    policy_age_days:    int
    policy_type:        str   # auto / health / property / liability
    coverage_amount:    float
    premium_monthly:    float
    prior_claims_12m:   int

    # Incident
    incident_type:      str   # collision / theft / medical / fire / …
    reported_amount:    float
    witnesses:          int
    police_report:      bool
    third_party:        bool

    # Claimant behaviour
    contact_channel:    str   # web / phone / agent / walk-in
    legal_rep:          bool
    address_changes_6m: int

    # Label
    fraud_label:        Optional[int]   = None   # 1 = fraud, 0 = clean
    fraud_score:        Optional[float] = None   # model output [0, 1]


# Columns used to build the feature vector (no PII).
# Categorical columns are one-hot encoded at load time.
NUMERIC_COLS = [
    "days_to_file", "policy_age_days", "coverage_amount", "premium_monthly",
    "prior_claims_12m", "reported_amount", "witnesses", "address_changes_6m",
]
BOOL_COLS = ["police_report", "third_party", "legal_rep"]
CAT_COLS  = ["policy_type", "incident_type", "contact_channel"]
