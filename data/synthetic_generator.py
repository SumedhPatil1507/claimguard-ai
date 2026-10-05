"""
Synthetic insurance data generator for ClaimGuard AI.

Generates:
  - data/sample_claims.csv  : 500 rows, 14 columns
  - data/sample_policies.csv: 300 rows, 13 columns

Usage:
    python data/synthetic_generator.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
CLAIMS_CSV = DATA_DIR / "sample_claims.csv"
POLICIES_CSV = DATA_DIR / "sample_policies.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
CLAIM_TYPES = ["auto", "health", "property"]
CLAIM_SEVERITIES = ["low", "medium", "high"]
COVERAGE_TYPES = ["comprehensive", "third-party", "basic"]
OCCUPATIONS = [
    "salaried", "self-employed", "business-owner",
    "professional", "retired", "student", "homemaker",
]
REGIONS = ["North", "South", "East", "West", "Central"]

REPAIR_SHOPS = [f"RS{str(i).zfill(3)}" for i in range(1, 21)]   # RS001-RS020
MEDICAL_PROVIDERS = [f"MP{str(i).zfill(3)}" for i in range(1, 16)]  # MP001-MP015


def _random_date(rng: np.random.Generator, start: str, end: str) -> pd.Series:
    """Return a numpy array of ISO-date strings between *start* and *end*."""
    start_ts = pd.Timestamp(start).value // 10**9
    end_ts = pd.Timestamp(end).value // 10**9
    seconds = rng.integers(start_ts, end_ts, endpoint=False)
    return pd.to_datetime(seconds, unit="s").strftime("%Y-%m-%d")


def generate_claims(rng: np.random.Generator, n: int = 500) -> pd.DataFrame:
    """Generate synthetic claims data."""

    claim_ids = [f"CLM{str(i).zfill(5)}" for i in range(1, n + 1)]
    policy_ids = [f"POL{str(rng.integers(1, 301)).zfill(5)}" for _ in range(n)]
    claimant_ids = [f"CLT{str(rng.integers(1, 401)).zfill(5)}" for _ in range(n)]

    claim_amounts = np.round(rng.uniform(500, 500_000, n), 2)

    claim_dates = _random_date(rng, "2020-01-01", "2024-12-31")
    # Policy start is 0-1095 days before the claim date
    days_before = rng.integers(0, 1096, n)
    policy_start_dates = (
        pd.to_datetime(claim_dates) - pd.to_timedelta(days_before, unit="D")
    ).strftime("%Y-%m-%d")
    days_since_policy_start = days_before  # kept as ndarray; written to CSV as ints

    claim_types = rng.choice(CLAIM_TYPES, n)
    num_prior_claims = rng.integers(0, 6, n)  # 0-5

    # Repair shop: relevant mainly for auto, sometimes property; None for health
    repair_shop_ids = []
    medical_provider_ids = []
    for ct in claim_types:
        if ct == "auto":
            repair_shop_ids.append(rng.choice(REPAIR_SHOPS + [None] * 5))
            medical_provider_ids.append(None)
        elif ct == "health":
            repair_shop_ids.append(None)
            medical_provider_ids.append(rng.choice(MEDICAL_PROVIDERS))
        else:  # property
            repair_shop_ids.append(rng.choice(REPAIR_SHOPS[:10] + [None] * 10))
            medical_provider_ids.append(None)

    # Witness IDs: 0-3 witnesses per claim
    witness_ids_col = []
    for _ in range(n):
        k = int(rng.integers(0, 4))
        if k == 0:
            witness_ids_col.append("")
        else:
            wids = [f"WIT{str(rng.integers(1, 201)).zfill(4)}" for _ in range(k)]
            witness_ids_col.append(",".join(wids))

    # Severity: correlated with amount
    severity_thresholds = np.percentile(claim_amounts, [33, 66])
    claim_severities = np.where(
        claim_amounts <= severity_thresholds[0],
        "low",
        np.where(claim_amounts <= severity_thresholds[1], "medium", "high"),
    )

    # Fraud label: ~15% positive, influenced by risk factors
    fraud_score = (
        0.05
        + 0.08 * (num_prior_claims > 2).astype(float)
        + 0.06 * (days_since_policy_start < 90).astype(float)
        + 0.04 * (claim_amounts > 200_000).astype(float)
        + 0.03 * (claim_severities == "high").astype(float)
        + rng.uniform(0, 0.05, n)  # small random nudge
    )
    fraud_label = (rng.uniform(0, 1, n) < fraud_score).astype(int)

    df = pd.DataFrame(
        {
            "claim_id": claim_ids,
            "policy_id": policy_ids,
            "claimant_id": claimant_ids,
            "claim_amount": claim_amounts,
            "claim_date": claim_dates,
            "policy_start_date": policy_start_dates,
            "days_since_policy_start": days_since_policy_start,
            "claim_type": claim_types,
            "num_prior_claims": num_prior_claims,
            "repair_shop_id": repair_shop_ids,
            "medical_provider_id": medical_provider_ids,
            "witness_ids": witness_ids_col,
            "claim_severity": claim_severities,
            "fraud_label": fraud_label,
        }
    )
    return df


def generate_policies(rng: np.random.Generator, n: int = 300) -> pd.DataFrame:
    """Generate synthetic policy/applicant data."""

    policy_ids = [f"POL{str(i).zfill(5)}" for i in range(1, n + 1)]
    applicant_ids = [f"APP{str(i).zfill(5)}" for i in range(1, n + 1)]

    ages = rng.integers(18, 71, n)  # 18-70
    occupations = rng.choice(OCCUPATIONS, n)
    annual_incomes = np.round(rng.uniform(120_000, 5_000_000, n), 2)
    sum_insured = np.round(rng.uniform(50_000, 10_000_000, n), 2)
    coverage_types = rng.choice(COVERAGE_TYPES, n)
    credit_scores = rng.integers(300, 851, n)  # 300-850
    num_dependents = rng.integers(0, 6, n)  # 0-5
    regions = rng.choice(REGIONS, n)
    prior_claims_count = rng.integers(0, 6, n)  # 0-5

    # Risk tier: heuristic based on credit score + prior claims + age
    risk_score = (
        (850 - credit_scores) / 550.0 * 0.4
        + prior_claims_count / 5.0 * 0.4
        + np.where(ages < 25, 0.2, np.where(ages > 60, 0.1, 0.0))
    )
    risk_tier = np.where(
        risk_score < 0.35,
        "low",
        np.where(risk_score < 0.60, "medium", "high"),
    )

    # Premium adjustment: 0.8 (good risk) to 1.5 (high risk)
    premium_adj_base = 0.8 + risk_score * 0.7 / 1.0  # maps 0-1 risk to 0.8-1.5
    premium_adjustment = np.round(
        np.clip(premium_adj_base + rng.uniform(-0.05, 0.05, n), 0.8, 1.5), 4
    )

    df = pd.DataFrame(
        {
            "policy_id": policy_ids,
            "applicant_id": applicant_ids,
            "age": ages,
            "occupation": occupations,
            "annual_income": annual_incomes,
            "sum_insured": sum_insured,
            "coverage_type": coverage_types,
            "credit_score": credit_scores,
            "num_dependents": num_dependents,
            "region": regions,
            "prior_claims_count": prior_claims_count,
            "risk_tier": risk_tier,
            "premium_adjustment": premium_adjustment,
        }
    )
    return df


def main() -> None:
    rng = np.random.default_rng(42)

    print("Generating sample_claims.csv …")
    claims_df = generate_claims(rng, n=500)
    CLAIMS_CSV.parent.mkdir(parents=True, exist_ok=True)
    claims_df.to_csv(CLAIMS_CSV, index=False)
    fraud_rate = claims_df["fraud_label"].mean() * 100
    print(
        f"  ✓ {len(claims_df)} rows, {len(claims_df.columns)} columns written → {CLAIMS_CSV}"
    )
    print(f"  Fraud label positive rate: {fraud_rate:.1f}%")

    print("Generating sample_policies.csv …")
    policies_df = generate_policies(rng, n=300)
    policies_df.to_csv(POLICIES_CSV, index=False)
    print(
        f"  ✓ {len(policies_df)} rows, {len(policies_df.columns)} columns written → {POLICIES_CSV}"
    )

    print("Done.")


if __name__ == "__main__":
    main()
