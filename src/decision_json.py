"""
decision_json.py
================
Standardized **Explainable Decision JSON** shared by every ClaimGuard AI
underwriting, claims-fraud, and Policy Copilot API response.

Exact output schema (v1.9 — contract for all downstream consumers):

    {
      "verdict":       "CLEAR_DECISION_BANNER (e.g., REJECT / ESCALATE TO FIU)",
      "reasoning":     "Detailed TreeSHAP feature driver breakdown & Qdrant similarity scores",
      "recommendation":"High-level risk action (e.g., Auto-Approve, Request Field Audit, HITL Review)",
      "next_steps":    [
        "Step 1: Specific actionable instruction for analyst",
        "Step 2: Verification or document request requirement",
        "Step 3: Statutory limit audit check"
      ]
    }

Design notes
------------
* ``verdict`` is a short, ALL-CAPS banner safe to render directly in a UI chip.
* ``reasoning`` embeds the TreeSHAP feature-driver breakdown and — when
  available — Qdrant hybrid-search similarity scores (dense + BM25 RRF).
* ``recommendation`` is the high-level risk action for the workflow engine.
* ``next_steps`` always contains at least three actionable analyst
  instructions ending with an IRDAI statutory-limit audit check.
* Every builder is defensive: missing keys degrade to safe defaults so the
  schema contract never breaks scoring pipelines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IRDAI statutory limits (kept in sync with src/guardrails_config.py)
# ---------------------------------------------------------------------------

try:
    from src.guardrails_config import _IRDAI_STATUTORY_LIMITS
except Exception:  # pragma: no cover — offline / partial installs
    _IRDAI_STATUTORY_LIMITS = {
        "motor_tp": {"max_payout": 750000.0,
                     "regulation": "IRDAI Motor Third Party Claims Guidelines 2023"},
        "health":   {"max_payout": 5000000.0,
                     "regulation": "IRDAI Health Insurance Regulations 2020"},
        "property": {"max_payout": 10000000.0,
                     "regulation": "IRDAI Property Insurance Guidelines 2022"},
        "life":     {"max_payout": 100000000.0,
                     "regulation": "IRDAI Life Insurance Regulations 2019"},
    }


def _statutory_limit_for(claim_type: str) -> Dict[str, Any]:
    """Resolve the IRDAI statutory limit configuration for a claim type."""
    key = (claim_type or "motor").lower()
    if key.startswith("motor"):
        return _IRDAI_STATUTORY_LIMITS.get(
            "motor_od" if key in ("motor", "motor_od", "motor own damage")
            else "motor_tp",
            _IRDAI_STATUTORY_LIMITS.get("motor_tp", {}),
        )
    return _IRDAI_STATUTORY_LIMITS.get(
        key, _IRDAI_STATUTORY_LIMITS.get("motor_tp", {})
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ExplainableDecision(BaseModel):
    """The standardized explainable decision JSON returned by all scoring APIs."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "verdict": "ESCALATE TO FIU",
                "reasoning": (
                    "TreeSHAP drivers: claim_amount (+0.41 increases_risk), "
                    "days_since_policy_start (+0.19 increases_risk). "
                    "Qdrant hybrid similarity: irdai_guidelines.txt@0.91."
                ),
                "recommendation": "HITL Review",
                "next_steps": [
                    "Step 1: Review the top TreeSHAP fraud drivers with the assigned analyst",
                    "Step 2: Request repair invoices and medical receipts for verification",
                    "Step 3: Confirm the payout is within IRDAI statutory limits before settlement",
                ],
            }
        },
    )

    verdict: str = Field(
        ...,
        min_length=1,
        description="Clear decision banner, e.g. REJECT / ESCALATE TO FIU / CLEAR — AUTO-APPROVE",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Detailed TreeSHAP feature driver breakdown & Qdrant similarity scores",
    )
    recommendation: str = Field(
        ...,
        min_length=1,
        description="High-level risk action, e.g. Auto-Approve, Request Field Audit, HITL Review",
    )
    next_steps: List[str] = Field(
        ...,
        min_length=3,
        description="Ordered actionable instructions for the analyst (>= 3 steps)",
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_shap_drivers(drivers: Any, top_n: int = 5) -> str:
    """Render TreeSHAP drivers as a compact, human-readable breakdown."""
    if not drivers or not isinstance(drivers, list):
        return "TreeSHAP drivers: unavailable (model explainability not returned)."

    parts: List[str] = []
    for drv in drivers[:top_n]:
        if not isinstance(drv, dict):
            continue
        feature = (
            drv.get("feature")
            or drv.get("name")
            or drv.get("column")
            or "unknown_feature"
        )
        raw_value = drv.get("shap_value", drv.get("value", drv.get("contribution", 0.0)))
        try:
            shap_value = float(raw_value)
        except (TypeError, ValueError):
            shap_value = 0.0
        direction = (
            str(drv.get("direction") or "")
            or ("increases_risk" if shap_value >= 0 else "decreases_risk")
        )
        raw_feat_value = drv.get("feature_value", drv.get("input_value"))
        feat_val = f" (input={raw_feat_value})" if raw_feat_value is not None else ""
        parts.append(f"{feature}{feat_val} → {shap_value:+.4f} [{direction}]")

    if not parts:
        return "TreeSHAP drivers: none reported by the model."
    return "TreeSHAP feature drivers: " + "; ".join(parts) + "."


def _format_similarity_scores(scores: Any) -> str:
    """Render Qdrant hybrid-search similarity scores for the reasoning field."""
    if not scores or not isinstance(scores, list):
        return (
            "Qdrant similarity scores: not attached for this response — "
            "invoke /copilot/decide for retrieval-grounded citations, or set "
            "CLAIMGUARD_INLINE_RETRIEVAL=1 to embed hybrid retrieval in scoring tasks."
        )
    parts: List[str] = []
    for hit in scores[:5]:
        if not isinstance(hit, dict):
            continue
        source = hit.get("source", hit.get("chunk_id", "unknown_source"))
        score = hit.get("score", hit.get("relevance_score", 0.0))
        chunk_id = hit.get("chunk_id")
        suffix = f" [{chunk_id}]" if chunk_id else ""
        try:
            parts.append(f"{source}{suffix}@{float(score):.3f}")
        except (TypeError, ValueError):
            parts.append(f"{source}{suffix}")
    if not parts:
        return "Qdrant similarity scores: no hybrid hits returned."
    return "Qdrant hybrid similarity scores: " + ", ".join(parts) + "."


def _statutory_step(claim_type: str, amount: Optional[float]) -> str:
    """Step 3 — IRDAI statutory-limit audit instruction."""
    limit = _statutory_limit_for(claim_type)
    max_payout = float(limit.get("max_payout", 0.0) or 0.0)
    regulation = limit.get("regulation", "IRDAI statutory limits")
    amount_txt = f"₹{amount:,.0f}" if amount is not None else "the requested amount"
    return (
        f"Step 3: Statutory limit audit check — validate {amount_txt} against "
        f"{regulation} (max ₹{max_payout:,.0f}) before any settlement or issuance."
    )


def _decision(
    verdict: str,
    reasoning: str,
    recommendation: str,
    next_steps: List[str],
) -> ExplainableDecision:
    """Assemble the schema, guaranteeing the mandatory 3-step shape."""
    steps = [s for s in next_steps if s]
    if len(steps) < 3:  # pragma: no cover — defensive
        steps = (steps + [
            "Step 1: Route the case to the assigned underwriting/fraud analyst",
            "Step 2: Request missing supporting documents from the claimant",
            "Step 3: Statutory limit audit check against IRDAI regulations",
        ])[:3]
    # Enforce monotonically numbered steps for a stable contract.
    numbered = [
        f"Step {i}: {s.split(':', 1)[1].strip() if ':' in s else s}"
        for i, s in enumerate(steps, start=1)
    ]
    return ExplainableDecision(
        verdict=verdict,
        reasoning=reasoning,
        recommendation=recommendation,
        next_steps=numbered,
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_underwriting_decision(
    result: Dict[str, Any],
    similarity_scores: Optional[List[Dict[str, Any]]] = None,
) -> ExplainableDecision:
    """
    Build the explainable decision JSON for an underwriting risk score.

    Parameters
    ----------
    result : dict
        Serialised ``UnderwritingResult`` (risk_tier, risk_score,
        premium_adjustment, shap_drivers).
    similarity_scores : list, optional
        Qdrant hybrid-search hits ({source, score, chunk_id}).
    """
    result = result if isinstance(result, dict) else {}
    tier = str(result.get("risk_tier", "medium")).lower()
    try:
        risk_score = float(result.get("risk_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        risk_score = 0.0
    try:
        premium = float(result.get("premium_adjustment", 1.0) or 1.0)
    except (TypeError, ValueError):
        premium = 1.0
    drivers = _format_shap_drivers(result.get("shap_drivers"))
    similarity = _format_similarity_scores(similarity_scores)

    if tier == "high":
        verdict = "REJECT — HIGH RISK APPLICANT"
        recommendation = "HITL Review"
        step1 = (
            "Step 1: Analyst must review the top TreeSHAP risk drivers "
            "(credit_score, prior_claims_count, sum_insured) and document rationale"
        )
        step2 = (
            "Step 2: Request income proof, prior-claim declarations, and "
            "inspection reports before any reconsideration"
        )
    elif tier == "medium":
        verdict = "ESCALATE — MANUAL UNDERWRITING REQUIRED"
        recommendation = "Request Field Audit"
        step1 = (
            "Step 1: Analyst must verify the medium-risk TreeSHAP drivers and "
            "confirm the proposed premium adjustment"
        )
        step2 = (
            "Step 2: Request identity, income, and occupancy verification "
            "documents to complete the underwriting file"
        )
    else:
        verdict = "CLEAR — AUTO-APPROVE"
        recommendation = "Auto-Approve"
        step1 = (
            "Step 1: Auto-approve subject to routine sample audit; record the "
            "low-risk TreeSHAP drivers in the underwriting file"
        )
        step2 = (
            "Step 2: Confirm KYC and proposal disclosures are complete before "
            "policy issuance"
        )

    reasoning = (
        f"Underwriting risk_tier={tier}, risk_score={risk_score:.4f}, "
        f"premium_adjustment={premium:.3f}. {drivers} {similarity}"
    )
    return _decision(
        verdict=verdict,
        reasoning=reasoning,
        recommendation=recommendation,
        next_steps=[
            step1,
            step2,
            _statutory_step("motor", result.get("sum_insured")),
        ],
    )



def build_fraud_decision(
    result: Dict[str, Any],
    similarity_scores: Optional[List[Dict[str, Any]]] = None,
    claim_type: Optional[str] = None,
) -> ExplainableDecision:
    """
    Build the explainable decision JSON for a claims fraud score.

    Parameters
    ----------
    result : dict
        Serialised ``FraudScoringResult`` (fraud_score, fraud_flag,
        confidence_tier, shap_drivers).
    similarity_scores : list, optional
        Qdrant hybrid-search hits ({source, score, chunk_id}).
    claim_type : str, optional
        IRDAI claim class (motor/health/property/life) for the limit audit.
    """
    result = result if isinstance(result, dict) else {}
    try:
        fraud_score = float(result.get("fraud_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        fraud_score = 0.0
    flag = bool(result.get("fraud_flag", fraud_score >= 0.5))
    confidence = str(result.get("confidence_tier", "medium"))
    claim_type = claim_type or str(result.get("claim_type", "motor"))
    try:
        claim_amount = float(
            result.get("claim_amount") or result.get("recommended_payout") or 0.0
        )
    except (TypeError, ValueError):
        claim_amount = 0.0
    claim_amount = claim_amount if claim_amount > 0 else None

    drivers = _format_shap_drivers(result.get("shap_drivers"))
    similarity = _format_similarity_scores(similarity_scores)
    claim_id = result.get("claim_id", "unknown")

    if fraud_score >= 0.70:
        verdict = "ESCALATE TO FIU"
        recommendation = "HITL Review"
        step1 = (
            "Step 1: Analyst must review the top TreeSHAP fraud drivers for "
            f"claim {claim_id} and file an FIU escalation note within 24 hours"
        )
        step2 = (
            "Step 2: Request original repair invoices, medical receipts, and "
            "first-information report before any settlement release"
        )
    elif flag or fraud_score >= 0.50:
        verdict = "REJECT PENDING INVESTIGATION"
        recommendation = "Request Field Audit"
        step1 = (
            "Step 1: Analyst must validate the flagged TreeSHAP drivers for "
            f"claim {claim_id} against the FNOL narrative"
        )
        step2 = (
            "Step 2: Request supporting bills, provider KYC, and bank "
            "remittance details for independent field verification"
        )
    else:
        verdict = "CLEAR — AUTO-APPROVE"
        recommendation = "Auto-Approve"
        step1 = (
            "Step 1: Auto-approve settlement for "
            f"claim {claim_id}; retain the fraud score ({fraud_score:.4f}) "
            "in the audit trail"
        )
        step2 = (
            "Step 2: Confirm all mandatory claim documents are on file "
            f"(confidence tier: {confidence})"
        )

    reasoning = (
        f"Fraud score={fraud_score:.4f}, fraud_flag={str(flag).lower()}, "
        f"confidence_tier={confidence}. {drivers} {similarity}"
    )
    return _decision(
        verdict=verdict,
        reasoning=reasoning,
        recommendation=recommendation,
        next_steps=[
            step1,
            step2,
            _statutory_step(claim_type, claim_amount),
        ],
    )



def build_batch_decision(
    kind: str,
    processed: int,
    flagged: int,
    errors: int,
) -> ExplainableDecision:
    """
    Build the explainable decision JSON summarising a CSV batch ingestion run.

    Parameters
    ----------
    kind : str       — 'claims' | 'policies' | 'generic'
    processed : int  — rows successfully scored/ingested
    flagged : int    — rows that breached risk/fraud thresholds
    errors : int     — rows that failed validation
    """
    if errors:
        verdict = f"ESCALATE — {errors} ROWS FAILED VALIDATION"
        recommendation = "HITL Review"
    elif flagged:
        verdict = f"ESCALATE — {flagged} FLAGGED FOR MANUAL REVIEW"
        recommendation = "Request Field Audit"
    else:
        verdict = f"CLEAR — {processed} ROWS AUTO-APPROVED"
        recommendation = "Auto-Approve"

    unit = "claims" if kind == "claims" else ("policies" if kind == "policies" else "rows")
    reasoning = (
        f"Batch ingestion classified {processed + errors} {unit} as '{kind}'. "
        f"processed={processed}, flagged={flagged}, validation_errors={errors}. "
        f"{_format_similarity_scores(None)}"
    )
    return _decision(
        verdict=verdict,
        reasoning=reasoning,
        recommendation=recommendation,
        next_steps=[
            f"Step 1: Analyst must review the {flagged} flagged {unit} "
            "(TreeSHAP drivers included per row in `items`)",
            f"Step 2: Request missing supporting documents for the {errors} "
            "rows that failed schema validation",
            _statutory_step("motor", None),
        ],
    )


def build_copilot_decision(
    decision: Dict[str, Any],
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
) -> ExplainableDecision:
    """
    Build the explainable decision JSON for a Policy Copilot response by
    mapping the Guardrails-validated ``CopilotDecision`` fields onto the
    standardized schema (and appending chunk-ID citations).
    """
    decision = decision if isinstance(decision, dict) else {}
    coverage = str(decision.get("coverage_status", "requires_review"))
    try:
        confidence = float(decision.get("confidence_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    requires_hitl = bool(decision.get("requires_human_review", True))

    scores = None
    if retrieved_docs:
        scores = [
            {
                "source": d.get("title") or d.get("source", "retrieved_doc"),
                "chunk_id": d.get("chunk_id"),
                "score": d.get("score", d.get("relevance_score", 0.0)),
            }
            for d in retrieved_docs
            if isinstance(d, dict)
        ]

    if coverage == "covered" and not requires_hitl:
        verdict = "CLEAR — AUTO-APPROVE"
        recommendation = "Auto-Approve"
    elif coverage == "not_covered":
        verdict = "REJECT — COVERAGE NOT APPLICABLE"
        recommendation = "HITL Review"
    else:
        verdict = "ESCALATE TO FIU" if confidence < 0.6 else "ESCALATE — HITL REQUIRED"
        recommendation = "HITL Review"

    try:
        payout_f = float(decision.get("recommended_payout", 0.0) or 0.0)
    except (TypeError, ValueError):
        payout_f = 0.0

    reasoning = (
        f"coverage_status={coverage}, confidence_score={confidence:.4f}, "
        f"recommended_payout=₹{payout_f:,.2f}, "
        f"requires_human_review={str(requires_hitl).lower()}. "
        f"{_format_similarity_scores(scores)}"
    )
    return _decision(
        verdict=verdict,
        reasoning=reasoning,
        recommendation=recommendation,
        next_steps=[
            "Step 1: Analyst must validate every coverage assertion against the "
            "cited policy chunk IDs before communicating the decision",
            "Step 2: Request any outstanding documents required by the cited "
            "policy clauses (zero-hallucination policy: no uncited approvals)",
            _statutory_step(str(decision.get("claim_type", "motor")), payout_f),
        ],
    )

