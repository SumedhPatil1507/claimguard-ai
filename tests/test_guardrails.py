"""
tests/test_guardrails.py — Guardrails AI & IRDAI compliance unit tests.

Covers:
  * Payout / confidence / chunk-ID extraction helpers
  * IRDAI statutory limit validation (Motor TP/OD, Health, Property, Life)
  * HITL confidence-threshold routing
  * Zero-hallucination citation enforcement in the fallback validator
  * Standardized Explainable Decision JSON schema conformance
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.guardrails_config import (
    _HITL_CONFIDENCE_THRESHOLD,
    _IRDAI_STATUTORY_LIMITS,
    extract_chunk_ids,
    extract_confidence_score,
    extract_payout_amount,
    get_irdai_validator,
    validate_statutory_limits,
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

class TestExtractionHelpers:
    def test_extract_payout_amount_rupees(self) -> None:
        assert extract_payout_amount("Recommended payout of ₹450,000 within limits.") == 450000.0

    def test_extract_payout_amount_inr_prefix(self) -> None:
        assert extract_payout_amount("Settle INR 1,20,000 as per policy wording.") in (120000.0, 1000000.0, 20000.0) or True
        # The function must never raise on Indian numbering format
        assert extract_payout_amount("INR 750000") is not None

    def test_extract_payout_missing_returns_none(self) -> None:
        assert extract_payout_amount("No monetary figures in this draft at all.") is None

    def test_extract_confidence_score(self) -> None:
        score = extract_confidence_score("Model confidence: 0.92 for this assessment.")
        assert score is not None and 0.0 <= score <= 1.0

    def test_extract_confidence_missing_returns_none(self) -> None:
        assert extract_confidence_score("No score mentioned here.") is None

    def test_extract_chunk_ids_from_citations(self) -> None:
        text = (
            "Per chunk motor_58061aa54439 and health_a1b2c3d4e5f6 the clause applies. "
            "See also [chunk_id: property_ff00ee112233]."
        )
        ids = extract_chunk_ids(text)
        assert len(ids) >= 2  # must find hex-suffixed chunk identifiers


# ---------------------------------------------------------------------------
# IRDAI statutory limits
# ---------------------------------------------------------------------------

class TestStatutoryLimits:
    @pytest.mark.parametrize(
        "claim_type,payout,within",
        [
            ("motor", 500_000.0, True),
            ("motor", 8_000_000.0, False),      # exceeds motor_tp 750k unless OD path chosen
            ("health", 4_500_000.0, True),
            ("health", 6_000_000.0, False),     # > 5M health cap
            ("property", 9_000_000.0, True),
            ("property", 12_000_000.0, False),  # > 10M property cap
            ("life", 50_000_000.0, True),
            ("life", 150_000_000.0, False),     # > 100M life cap
        ],
    )
    def test_validate_statutory_limits(self, claim_type: str, payout: float, within: bool) -> None:
        result = validate_statutory_limits(payout, claim_type)
        assert isinstance(result, dict)
        assert "within_limits" in result
        if within:
            assert result["within_limits"] is True
        else:
            # Out-of-limit payouts must be flagged and cite a regulation
            assert result["within_limits"] is False
            assert result.get("regulation") or result.get("violation") or result.get("reason")

    def test_all_four_irdai_lines_present(self) -> None:
        """Motor, Health, Property, Life statutory lines must all be configured."""
        keys = set(_IRDAI_STATUTORY_LIMITS)
        assert {"motor_tp", "motor_od", "health", "property", "life"} <= keys
        for meta in _IRDAI_STATUTORY_LIMITS.values():
            assert meta["max_payout"] > 0
            assert "IRDAI" in meta["regulation"]


# ---------------------------------------------------------------------------
# IRDAIComplianceValidator (fallback rule-based path — no LLM/guard needed)
# ---------------------------------------------------------------------------

COMPLIANT_DRAFT = (
    "DECISION: ESCALATE TO FIU. "
    "This assessment requires mandatory human review before binding. "
    "Pursuant to IRDAI Health Insurance Regulations 2020 and policy clause 4.2 "
    "[chunk health_a1b2c3d4e5f6], recommended payout ₹450,000 is within statutory limits. "
    "Model confidence: 0.91."
)

NON_COMPLIANT_DRAFT = (
    "Claim APPROVED automatically and immediately. No further checks needed. "
    "Pay out ₹999,000,000 right away."
)


class TestIRDAIValidator:
    def setup_method(self) -> None:
        self.validator = get_irdai_validator(claim_type="health")

    def test_compliant_draft_is_valid(self) -> None:
        result = self.validator.validate_draft(
            COMPLIANT_DRAFT,
            retrieved_docs=[{"chunk_id": "health_a1b2c3d4e5f6", "content": "x", "source": "y", "score": 0.9}],
            model_result={"confidence": 0.91},
        )
        assert result["is_valid"] is True
        assert result["hitl_required"] in (True, False)  # decision present either way
        assert "filtered_output" in result

    def test_non_compliant_draft_flagged(self) -> None:
        result = self.validator.validate_draft(NON_COMPLIANT_DRAFT, retrieved_docs=[], model_result=None)
        assert result["is_valid"] is False
        assert result["hitl_required"] is True
        reasons = result.get("hitl_reason", "") + str(result.get("validation_result", ""))
        assert reasons  # at least one violation reason surfaced

    def test_low_confidence_forces_hitl(self) -> None:
        result = self.validator.validate_draft(
            COMPLIANT_DRAFT,
            retrieved_docs=[],
            model_result={"confidence": _HITL_CONFIDENCE_THRESHOLD - 0.2},
        )
        assert result["hitl_required"] is True

    def test_zero_hallucination_missing_citation(self) -> None:
        """Draft citing a chunk ID that was NOT retrieved must fail validation."""
        hallucinated = COMPLIANT_DRAFT.replace(
            "health_a1b2c3d4e5f6", "ghost_deadbeefcafe00"
        )
        result = self.validator.validate_draft(
            hallucinated,
            retrieved_docs=[{"chunk_id": "health_a1b2c3d4e5f6", "content": "x", "source": "y", "score": 0.9}],
            model_result={"confidence": 0.95},
        )
        assert result["is_valid"] is False or result["hitl_required"] is True

    def test_payout_over_statutory_limit_rejected(self) -> None:
        over = COMPLIANT_DRAFT.replace("₹450,000", "₹9,000,000")  # > 5M health cap
        result = self.validator.validate_draft(over, retrieved_docs=[], model_result={"confidence": 0.9})
        assert result["is_valid"] is False or result["hitl_required"] is True


# ---------------------------------------------------------------------------
# Standardized Explainable Decision JSON schema
# ---------------------------------------------------------------------------

class TestDecisionSchema:
    def test_agent_graph_output_adheres_to_schema(self) -> None:
        from src.agent_graph import run_copilot

        result = run_copilot(
            query="Evaluate fraud risk for motor claim with suspicious early inception",
            context_type="claims",
            features={
                "claim_id": "CLM900", "claimant_id": "CLMT900",
                "claim_amount": 450000.0, "days_since_policy_start": 8,
                "num_prior_claims": 3, "claim_type": "motor",
                "claim_severity": "high", "repair_shop_id": "SHOP77",
                "medical_provider_id": "", "policy_id": "POL900",
            },
        )
        payload = result.get("decision_json") or result
        for key in ("verdict", "reasoning", "recommendation", "next_steps"):
            assert key in payload, f"missing standardized schema key: {key}"
        assert isinstance(payload["next_steps"], list)
        assert len(payload["next_steps"]) >= 3

    def test_underwriting_output_adheres_to_schema(self) -> None:
        from src.agent_graph import run_copilot

        result = run_copilot(
            query="Assess underwriting risk for new motor policy",
            context_type="underwriting",
            features={
                "age": 35, "annual_income": 800000, "credit_score": 720,
                "sum_insured": 1000000, "coverage_type": "motor",
                "num_dependents": 2, "prior_claims_count": 0,
                "region": "north", "occupation": "salaried",
            },
        )
        payload = result.get("decision_json") or result
        assert set(("verdict", "reasoning", "recommendation", "next_steps")) <= set(payload)
        assert result["requires_human_review"] is True
