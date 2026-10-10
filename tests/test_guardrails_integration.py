"""
Test Guardrails AI integration for IRDAI compliance validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_guardrails_validator_initialization():
    """Test that the IRDAI compliance validator can be initialized."""
    from src.guardrails_config import get_irdai_validator, HAS_GUARDRAILS

    validator = get_irdai_validator()
    assert validator is not None, "Validator should be initialized"

    if HAS_GUARDRAILS:
        assert validator._enabled, "Validator should be enabled when Guardrails AI is available"
    else:
        assert not validator._enabled, "Validator should be disabled when Guardrails AI is unavailable"

    print("[PASS] Guardrails validator initialization test passed")


def test_guardrails_compliant_draft():
    """Test validation of a compliant decision draft."""
    from src.guardrails_config import get_irdai_validator

    validator = get_irdai_validator()

    compliant_draft = """
    This decision draft requires mandatory human analyst review per IRDAI guidelines.
    Based on IRDAI regulation 3.2, the claim must be verified by a human analyst.
    No approval can be issued without human review.
    """

    result = validator.validate_draft(compliant_draft)

    assert result is not None, "Validation result should not be None"
    assert "is_valid" in result, "Result should contain is_valid"
    assert "validation_result" in result, "Result should contain validation_result"

    print(f"[PASS] Compliant draft validation: is_valid={result['is_valid']}")


def test_guardrails_non_compliant_draft():
    """Test validation of a non-compliant decision draft (unverified approval)."""
    from src.guardrails_config import get_irdai_validator

    validator = get_irdai_validator()

    non_compliant_draft = """
    This claim is approved automatically without human review.
    Coverage is granted immediately.
    """

    result = validator.validate_draft(non_compliant_draft)

    assert result is not None, "Validation result should not be None"
    assert "is_valid" in result, "Result should contain is_valid"

    # Should be invalid due to unverified approval
    if not validator._enabled:
        # Fallback validation should still catch this
        assert not result["is_valid"], "Fallback validation should reject unverified approval"

    print(f"[PASS] Non-compliant draft validation: is_valid={result['is_valid']}")


def test_guardrails_fallback_validation():
    """Test fallback rule-based validation when Guardrails AI is disabled."""
    from src.guardrails_config import IRDAIComplianceValidator

    # Create validator without Guardrails AI
    validator = IRDAIComplianceValidator()
    validator._enabled = False  # Force fallback mode

    # A fully compliant draft: mandatory human review + IRDAI reference +
    # vector-store chunk citation (zero-hallucination grounding).
    compliant_draft = """
    This decision draft requires mandatory human analyst review per IRDAI guidelines.
    Grounded in policy_section_6_1_chunk_2 of the retrieved policy wording.
    """

    result = validator.validate_draft(compliant_draft)

    assert result is not None, "Fallback validation should return a result"
    assert result["is_valid"], "Fallback should validate compliant draft"
    assert result["validation_result"]["validation_method"] == "fallback_comprehensive"
    assert result["validation_result"]["has_chunk_citations"], "Chunk ID citation required"
    assert not result["hitl_required"], "No HITL reasons expected for compliant draft"

    print("[PASS] Fallback validation test passed")


if __name__ == "__main__":
    test_guardrails_validator_initialization()
    test_guardrails_compliant_draft()
    test_guardrails_non_compliant_draft()
    test_guardrails_fallback_validation()
    print("\n[PASS] All Guardrails integration tests passed!")
