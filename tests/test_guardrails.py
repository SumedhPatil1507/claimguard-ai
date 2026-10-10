"""
tests/test_guardrails.py
=========================
Unit tests for Guardrails AI enforcement, `config.rail` integrity, IRDAI
statutory limits, and the standardized ExplainableDecision JSON contract.

Test groups
-----------
A  config.rail — exists, parses, and enforces the required policies
B  Guardrails AI guard loading (Guard.from_rail) — skipped when not installed
C  IRDAIComplianceValidator — compliant vs. non-compliant drafts (deterministic
   fallback path, independent of whether guardrails-ai is installed)
D  validate_statutory_limits — Motor / Health / Property / Life payout caps
E  ExplainableDecision schema — exact 4-key contract + 3-step next_steps
F  agent_graph wiring — WriterAgent validation is Guardrails-enabled
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

RAIL_PATH = Path(__file__).parent.parent / "config.rail"


# ===========================================================================
# A — config.rail integrity
# ===========================================================================

class TestRailSpecification:
    def test_config_rail_exists(self) -> None:
        assert RAIL_PATH.exists(), "config.rail must exist at the repo root"

    def test_rail_enforces_zero_hallucination(self) -> None:
        rail = RAIL_PATH.read_text(encoding="utf-8")
        assert "Zero-Hallucination Policy" in rail
        assert "hallucination_check_passed" in rail

    def test_rail_enforces_chunk_id_citations(self) -> None:
        rail = RAIL_PATH.read_text(encoding="utf-8")
        assert "clause_id" in rail
        assert "chunk" in rail.lower()
        assert "citation_check_passed" in rail

    def test_rail_enforces_statutory_limits(self) -> None:
        rail = RAIL_PATH.read_text(encoding="utf-8")
        assert "statutory_limits" in rail
        assert "max_allowed_payout" in rail
        assert "limit_check_passed" in rail
        assert 'max="10000000.0"' in rail, "payout cap must be encoded in the rail"

    def test_rail_enforces_confidence_based_hitl(self) -> None:
        rail = RAIL_PATH.read_text(encoding="utf-8")
        assert "confidence_score" in rail
        assert "0.85" in rail
        assert "requires_human_review" in rail


# ===========================================================================
# B — Guardrails AI guard loading
# ===========================================================================

class TestGuardLoading:
    def test_guard_loads_from_rail(self) -> None:
        """Guard rail loading (for_rail / from_rail) must initialise without errors."""
        try:
            import guardrails  # noqa: F401
        except ImportError:
            pytest.skip("guardrails-ai is not installed")

        from guardrails import Guard
        if hasattr(Guard, "for_rail"):          # guardrails-ai >= 0.10
            guard = Guard.for_rail(str(RAIL_PATH))
        else:                                    # legacy API
            guard = Guard.from_rail(RAIL_PATH)
        assert guard is not None

    def test_validator_reflects_package_availability(self) -> None:
        from src.guardrails_config import get_irdai_validator

        validator = get_irdai_validator(claim_type="motor")
        assert validator is not None
        assert validator._enabled in (True, False)

    def test_validator_cached_per_claim_type(self) -> None:
        from src.guardrails_config import get_irdai_validator

        a = get_irdai_validator(claim_type="health")
        b = get_irdai_validator(claim_type="health")
        assert a is b, "Validators must be cached per claim type"



# ===========================================================================
# C — Draft validation (deterministic fallback path)
# ===========================================================================

class TestDraftValidation:
    @pytest.fixture()
    def fallback_validator(self):
        from src.guardrails_config import IRDAIComplianceValidator
        validator = IRDAIComplianceValidator()
        validator._enabled = False  # force the deterministic rule-based path
        return validator

    def test_compliant_draft_passes(self, fallback_validator) -> None:
        draft = (
            "This decision requires mandatory human analyst review per IRDAI "
            "guidelines. Grounded in [Chunk ID: policy_section_4_2_chunk_1]."
        )
        result = fallback_validator.validate_draft(
            draft,
            retrieved_docs=[{"chunk_id": "policy_section_4_2_chunk_1"}],
        )
        assert result["is_valid"] is True
        assert result["hitl_required"] is False

    def test_unretrieved_chunk_citation_fails_closed(self, fallback_validator) -> None:
        draft = (
            "This decision requires mandatory human analyst review per IRDAI "
            "guidelines. Grounded in [Chunk ID: invented_chunk_99]."
        )
        result = fallback_validator.validate_draft(
            draft,
            retrieved_docs=[{"chunk_id": "actual_policy_chunk_1"}],
        )
        assert result["is_valid"] is False
        assert "not present in retrieved evidence" in result["hitl_reason"]

    def test_missing_retrieval_fails_closed(self, fallback_validator) -> None:
        draft = (
            "This decision requires mandatory human analyst review per IRDAI "
            "guidelines. Grounded in [Chunk ID: policy_chunk_1]."
        )
        result = fallback_validator.validate_draft(draft, retrieved_docs=[])
        assert result["is_valid"] is False
        assert "No retrieved documents" in result["hitl_reason"]

    def test_unverified_approval_rejected(self, fallback_validator) -> None:
        draft = (
            "This claim is approved automatically without human review. "
            "Coverage is granted immediately."
        )
        result = fallback_validator.validate_draft(draft)
        assert result["is_valid"] is False
        assert "approval" in result["hitl_reason"].lower()

    def test_missing_human_review_rejected(self, fallback_validator) -> None:
        draft = "Policy clause 3.1 applies here, chunk_id policy_x_chunk_9."
        result = fallback_validator.validate_draft(draft)
        assert result["is_valid"] is False
        assert "human review" in result["hitl_reason"].lower()

    def test_missing_chunk_citation_rejected(self, fallback_validator) -> None:
        draft = "This decision requires mandatory human analyst review per IRDAI guidelines."
        result = fallback_validator.validate_draft(draft)
        assert result["is_valid"] is False
        assert "chunk" in result["hitl_reason"].lower()

    def test_over_limit_payout_rejected(self, fallback_validator) -> None:
        # ₹2,00,00,000 exceeds the motor third-party cap of ₹7,50,000
        draft = (
            "Payout: 20000000 approved subject to mandatory human analyst "
            "review per IRDAI guidelines, cite chunk policy_a_chunk_1."
        )
        result = fallback_validator.validate_draft(draft)
        assert result["is_valid"] is False
        reason = result["hitl_reason"].lower()
        assert "statutory" in reason or "exceeds" in reason


# ===========================================================================
# D — IRDAI statutory limits (Motor / Health / Property / Life)
# ===========================================================================

class TestStatutoryLimits:
    def test_motor_third_party_cap(self) -> None:
        from src.guardrails_config import validate_statutory_limits
        assert validate_statutory_limits(500_000, "motor")["within_limits"] is True
        assert validate_statutory_limits(900_000, "motor")["within_limits"] is False

    def test_motor_own_damage_uses_own_damage_limit(self) -> None:
        from src.guardrails_config import validate_statutory_limits
        assert validate_statutory_limits(8_000_000, "motor_od")["within_limits"] is True

    def test_health_cap(self) -> None:
        from src.guardrails_config import validate_statutory_limits
        assert validate_statutory_limits(4_000_000, "health")["within_limits"] is True
        assert validate_statutory_limits(6_000_000, "health")["within_limits"] is False

    def test_property_cap(self) -> None:
        from src.guardrails_config import validate_statutory_limits
        assert validate_statutory_limits(9_000_000, "property")["within_limits"] is True
        assert validate_statutory_limits(11_000_000, "property")["within_limits"] is False

    def test_life_cap(self) -> None:
        from src.guardrails_config import validate_statutory_limits
        assert validate_statutory_limits(50_000_000, "life")["within_limits"] is True
        assert validate_statutory_limits(150_000_000, "life")["within_limits"] is False

    def test_excess_amount_reported(self) -> None:
        from src.guardrails_config import validate_statutory_limits
        result = validate_statutory_limits(1_000_000, "motor")
        assert result["excess_amount"] == pytest.approx(250_000.0)
        assert result["regulation"]


# ===========================================================================
# E — Standardized ExplainableDecision JSON contract
# ===========================================================================

class TestExplainableDecisionSchema:
    KEYS = {"verdict", "reasoning", "recommendation", "next_steps"}

    def _assert_contract(self, payload: dict) -> None:
        assert set(payload.keys()) == self.KEYS, "exact 4-key schema required"
        assert isinstance(payload["verdict"], str) and payload["verdict"].strip()
        assert isinstance(payload["reasoning"], str) and payload["reasoning"].strip()
        assert isinstance(payload["recommendation"], str) and payload["recommendation"].strip()
        assert isinstance(payload["next_steps"], list)
        assert len(payload["next_steps"]) >= 3
        for i, step in enumerate(payload["next_steps"], start=1):
            assert step.startswith(f"Step {i}:"), f"step {i} must be numbered"
        assert "Statutory limit audit" in payload["next_steps"][-1]

    def test_underwriting_decision_contract(self) -> None:
        from src.decision_json import build_underwriting_decision
        payload = build_underwriting_decision({
            "risk_tier": "high",
            "risk_score": 0.91,
            "premium_adjustment": 1.35,
            "shap_drivers": [{"feature": "credit_score", "shap_value": -0.22}],
        }).model_dump()
        self._assert_contract(payload)
        assert payload["verdict"] == "REJECT — HIGH RISK APPLICANT"

    def test_fraud_decision_contract(self) -> None:
        from src.decision_json import build_fraud_decision
        payload = build_fraud_decision({
            "claim_id": "CLM-9",
            "fraud_score": 0.88,
            "fraud_flag": True,
            "confidence_tier": "high",
            "shap_drivers": [{"feature": "claim_amount", "shap_value": 0.44}],
            "claim_amount": 500_000,
        }, claim_type="motor").model_dump()
        self._assert_contract(payload)
        assert payload["verdict"] == "ESCALATE TO FIU"

    def test_batch_decision_contract(self) -> None:
        from src.decision_json import build_batch_decision
        payload = build_batch_decision("claims", processed=120, flagged=4, errors=0).model_dump()
        self._assert_contract(payload)

    def test_copilot_decision_contract(self) -> None:
        from src.decision_json import build_copilot_decision
        payload = build_copilot_decision(
            {"coverage_status": "requires_review", "confidence_score": 0.72,
             "requires_human_review": True, "recommended_payout": 0.0},
            retrieved_docs=[{"chunk_id": "pol_chunk_1", "score": 0.93}],
        ).model_dump()
        self._assert_contract(payload)
        assert "pol_chunk_1" in payload["reasoning"] or "0.930" in payload["reasoning"]

    def test_reasoning_embeds_treeshap_and_qdrant(self) -> None:
        from src.decision_json import build_fraud_decision
        payload = build_fraud_decision(
            {"claim_id": "CLM-1", "fraud_score": 0.2, "fraud_flag": False,
             "shap_drivers": [{"feature": "num_prior_claims", "shap_value": 0.11}]},
            similarity_scores=[{"source": "irdai_guidelines.txt", "score": 0.91,
                                "chunk_id": "irdai_7_1_chunk_0"}],
        ).model_dump()
        assert "TreeSHAP" in payload["reasoning"]
        assert "Qdrant" in payload["reasoning"]
        assert "irdai_guidelines.txt" in payload["reasoning"]
        assert "0.910" in payload["reasoning"]


# ===========================================================================
# F — agent_graph WriterAgent wiring
# ===========================================================================

class TestAgentGraphWiring:
    def test_writer_node_uses_guardrails(self) -> None:
        import src.agent_graph as ag
        assert hasattr(ag, "HAS_GUARDRAILS")
        assert hasattr(ag, "get_irdai_validator")
        assert ag.HAS_GUARDRAILS or ag.get_irdai_validator is None

    def test_agent_graph_import_is_safe(self) -> None:
        from src.agent_graph import run_copilot  # noqa: F401
        assert callable(run_copilot)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

