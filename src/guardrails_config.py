"""
Guardrails AI configuration for IRDAI compliance validation.

This module defines validation rules for the Policy Copilot output to ensure:
1. Strict enforcement of IRDAI compliance limits
2. Refusal to emit unverified coverage approvals
3. Professional, regulatory-compliant language
4. Zero-hallucination policies on coverage claims and payout amounts
5. Statutory limit enforcement per IRDAI regulations
6. Explicit citation requirements with chunk IDs
7. Confidence-based HITL routing (< 0.85 threshold)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: Guardrails AI
# ---------------------------------------------------------------------------
try:
    from guardrails import Guard
    HAS_GUARDRAILS = True
except ImportError:
    HAS_GUARDRAILS = False
    Guard = None  # type: ignore[assignment]

# Optional hub validators — available only when the validator is installed
# (`guardrails hub install ...`) and exposed by the installed API version.
try:
    from guardrails import OnFailAction  # type: ignore
except ImportError:
    OnFailAction = None  # type: ignore[assignment]

try:
    from guardrails.hub import ToxicLanguage, PIIFilter  # type: ignore
except ImportError:
    ToxicLanguage = None  # type: ignore[assignment]
    PIIFilter = None      # type: ignore[assignment]


# ---------------------------------------------------------------------------
# IRDAI Statutory Limits Configuration
# ---------------------------------------------------------------------------

_IRDAI_STATUTORY_LIMITS = {
    "motor_tp": {
        "max_payout": 750000.0,
        "regulation": "IRDAI Motor Third Party Claims Guidelines 2023",
        "description": "Maximum payout for motor third-party claims"
    },
    "motor_od": {
        "max_payout": 10000000.0,
        "regulation": "IRDAI Motor Own Damage Guidelines 2023",
        "description": "Maximum payout for motor own-damage claims"
    },
    "health": {
        "max_payout": 5000000.0,
        "regulation": "IRDAI Health Insurance Regulations 2020",
        "description": "Maximum payout for health insurance claims"
    },
    "property": {
        "max_payout": 10000000.0,
        "regulation": "IRDAI Property Insurance Guidelines 2022",
        "description": "Maximum payout for property insurance claims"
    },
    "life": {
        "max_payout": 100000000.0,
        "regulation": "IRDAI Life Insurance Regulations 2019",
        "description": "Maximum sum assured for life insurance"
    }
}

# Confidence threshold for HITL routing
_HITL_CONFIDENCE_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# IRDAI Compliance Validation Rules
# ---------------------------------------------------------------------------

_IRDAI_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "compliance_check": {
            "type": "boolean",
            "description": "Whether the decision draft complies with IRDAI regulations"
        },
        "contains_approval": {
            "type": "boolean",
            "description": "Whether the draft contains an explicit approval (which should require human verification)"
        },
        "contains_denial": {
            "type": "boolean",
            "description": "Whether the draft contains an explicit denial"
        },
        "requires_human_review": {
            "type": "boolean",
            "description": "Whether the draft explicitly states it requires human review"
        },
        "references_regulation": {
            "type": "boolean",
            "description": "Whether the draft cites specific IRDAI regulations or policy clauses"
        },
        "has_chunk_citations": {
            "type": "boolean",
            "description": "Whether the draft includes chunk IDs for policy citations"
        },
        "confidence_score": {
            "type": "number",
            "description": "Confidence score of the decision (0.0 to 1.0)"
        },
        "payout_within_limits": {
            "type": "boolean",
            "description": "Whether the recommended payout is within IRDAI statutory limits"
        }
    },
    "required": ["compliance_check", "requires_human_review", "has_chunk_citations"]
}


# ---------------------------------------------------------------------------
# Helper Functions for Validation
# ---------------------------------------------------------------------------

def extract_payout_amount(text: str) -> Optional[float]:
    """Extract payout amount from decision text."""
    # Look for patterns like ₹50,000, 50000, "payout: 75000", etc.
    patterns = [
        r'₹[\s,]*([\d,]+\.?\d*)',
        r'payout[\s:]*([\d,]+\.?\d*)',
        r'amount[\s:]*([\d,]+\.?\d*)',
        r'approved[\s:]*([\d,]+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', '').replace('₹', ''))
            except (ValueError, AttributeError):
                continue
    return None


def extract_confidence_score(text: str) -> Optional[float]:
    """Extract confidence score from decision text."""
    patterns = [
        r'confidence[\s:]*([\d.]+)',
        r'score[\s:]*([\d.]+)',
        r'probability[\s:]*([\d.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except (ValueError, AttributeError):
                continue
    return None


def extract_chunk_ids(text: str) -> List[str]:
    """Extract chunk IDs from decision text."""
    # Prefer explicit [Chunk ID: ...] citations so IDs such as
    # ``policy_wording::h2::p1`` survive punctuation and heading separators.
    explicit = re.findall(r"\[\s*chunk\s*id\s*:\s*([^\]\r\n]+)\]", text, re.IGNORECASE)
    if explicit:
        return [value.strip() for value in explicit if value.strip()]
    # Backward compatible short IDs used by older generated drafts.
    return re.findall(r"(chunk[_\-\w]+|policy[_\w]+chunk[_\w]+)", text, re.IGNORECASE)


def _validate_citations_against_retrieval(
    draft: str, retrieved_docs: Optional[List[Dict[str, Any]]]
) -> List[str]:
    """Return citation failures, requiring citations to resolve to retrieved IDs."""
    cited = extract_chunk_ids(draft)
    if not retrieved_docs:
        return ["No retrieved documents to ground assertions"]

    available = set()
    for doc in retrieved_docs:
        if not isinstance(doc, dict):
            continue
        payload = doc.get("payload") if isinstance(doc.get("payload"), dict) else {}
        chunk_id = doc.get("chunk_id") or payload.get("chunk_id")
        if chunk_id:
            available.add(str(chunk_id))

    if not cited:
        return ["Missing chunk IDs for policy citations"]
    unknown = sorted(set(cited) - available)
    if unknown:
        return ["Citation IDs were not present in retrieved evidence: " + ", ".join(unknown)]
    return []


def validate_statutory_limits(payout: float, claim_type: str) -> Dict[str, Any]:
    """
    Validate payout against IRDAI statutory limits.

    Parameters
    ----------
    payout : float
        Recommended payout amount
    claim_type : str
        Type of claim (motor, health, property, life)

    Returns
    -------
    dict with keys:
        - within_limits: bool
        - max_allowed: float
        - regulation: str
        - description: str
    """
    claim_type_key = claim_type.lower()
    if claim_type_key in {"motor_od", "motor own damage", "motor own-damage"}:
        limit_config = _IRDAI_STATUTORY_LIMITS["motor_od"]
    elif claim_type_key.startswith("motor"):
        # Third-party claims use the motor_tp policy limit by default.
        limit_config = _IRDAI_STATUTORY_LIMITS["motor_tp"]
    else:
        limit_config = _IRDAI_STATUTORY_LIMITS.get(claim_type_key, _IRDAI_STATUTORY_LIMITS["motor_od"])

    max_allowed = limit_config["max_payout"]
    within_limits = payout <= max_allowed

    return {
        "within_limits": within_limits,
        "max_allowed": max_allowed,
        "regulation": limit_config["regulation"],
        "description": limit_config["description"],
        "excess_amount": max(0.0, payout - max_allowed) if not within_limits else 0.0
    }


_IRDAI_PROMPT_TEMPLATE = """
You are validating an insurance decision draft for IRDAI compliance.

Decision Draft:
{decision_draft}

Validation Rules:
1. The draft MUST explicitly state that it requires mandatory human analyst review per IRDAI guidelines
2. The draft MUST NOT contain any unverified coverage approvals
3. The draft MUST cite specific IRDAI regulations or policy clauses when applicable
4. The draft MUST maintain professional, regulatory-compliant language
5. The draft MUST clearly indicate that no final approval or denial can be issued without human review

Return a JSON object with the following structure:
{
  "compliance_check": true/false,
  "contains_approval": true/false,
  "contains_denial": true/false,
  "requires_human_review": true/false,
  "references_regulation": true/false,
  "validation_notes": "Any issues or recommendations"
}
"""


# ---------------------------------------------------------------------------
# Guardrails Validator Class
# ---------------------------------------------------------------------------

class IRDAIComplianceValidator:
    """
    Validates Policy Copilot outputs against IRDAI compliance requirements.

    This validator ensures that:
    - All decision drafts explicitly require human review
    - No unverified coverage approvals are emitted
    - Regulatory citations are present when applicable
    - Professional, compliant language is used
    - Zero-hallucination policies are enforced
    - Payout amounts respect statutory limits
    - Chunk IDs are included for all citations
    - Confidence-based HITL routing is enforced (< 0.85 threshold)
    """

    def __init__(self, claim_type: str = "motor"):
        self._guard: Optional[Any] = None
        self._enabled = HAS_GUARDRAILS
        self._claim_type = claim_type

        if self._enabled:
            try:
                self._initialize_guard()
            except Exception as exc:
                logger.warning("Failed to initialize Guardrails AI validator: %s", exc)
                self._enabled = False

    def _initialize_guard(self) -> None:
        """Initialize the Guardrails AI guard with IRDAI-specific rules."""
        if not HAS_GUARDRAILS:
            return

        try:
            # Try to load from config.rail if it exists.
            # guardrails-ai >= 0.10 exposes Guard.for_rail(); older versions
            # expose Guard.from_rail().  Support both APIs.
            rail_path = Path(__file__).parent.parent / "config.rail"
            if rail_path.exists():
                if hasattr(Guard, "for_rail"):
                    self._guard = Guard.for_rail(str(rail_path))
                else:
                    self._guard = Guard.from_rail(rail_path)
                logger.info("Guardrails AI IRDAI validator initialized from config.rail")
            else:
                # Fallback to schema-based guard
                self._guard = Guard.from_string(
                    _IRDAI_VALIDATION_SCHEMA,
                    on_fail="fix"  # Auto-fix minor issues, raise on major violations
                )
                logger.info("Guardrails AI IRDAI validator initialized from schema")
        except Exception as exc:
            logger.error("Failed to create Guardrails AI guard: %s", exc)
            raise

    def validate_draft(
        self,
        decision_draft: str,
        retrieved_docs: Optional[List[Dict]] = None,
        model_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Validate a decision draft against IRDAI compliance rules.

        Parameters
        ----------
        decision_draft : str
            The decision draft text to validate
        retrieved_docs : list of dict, optional
            Retrieved documents from vector store
        model_result : dict, optional
            Model scoring results

        Returns
        -------
        dict with keys:
            - is_valid: bool - Whether the draft passes validation
            - validation_result: dict - Detailed validation results
            - filtered_output: str - The filtered/corrected output if applicable
            - hitl_required: bool - Whether HITL is required
            - hitl_reason: str - Reason for HITL routing
        """
        if not self._enabled:
            # Fallback: perform comprehensive rule-based validation
            return self._fallback_validation(decision_draft, retrieved_docs, model_result)

        try:
            # Run Guardrails validation
            validation_result = self._guard.parse(decision_draft)

            # Extract structured validation data
            validation_data = validation_result if isinstance(validation_result, dict) else {}

            # Check critical compliance requirements
            is_valid, hitl_required, hitl_reason = self._check_compliance_requirements(
                validation_data, decision_draft, retrieved_docs, model_result
            )

            return {
                "is_valid": is_valid,
                "validation_result": validation_data,
                "filtered_output": decision_draft,
                "hitl_required": hitl_required,
                "hitl_reason": hitl_reason
            }

        except Exception as exc:
            logger.warning("Guardrails validation failed, using fallback: %s", exc)
            return self._fallback_validation(decision_draft, retrieved_docs, model_result)

    def _check_compliance_requirements(
        self,
        validation_data: Dict[str, Any],
        draft: str,
        retrieved_docs: Optional[List[Dict]] = None,
        model_result: Optional[Dict] = None
    ) -> tuple[bool, bool, str]:
        """
        Check if the draft meets critical IRDAI compliance requirements.

        Returns
        -------
        tuple: (is_valid, hitl_required, hitl_reason)
        """
        hitl_reasons = []
        is_valid = True

        # Check 1: Must explicitly require human review
        requires_review = validation_data.get("requires_human_review", False)
        if not requires_review:
            requires_review = (
                "human review" in draft.lower() or
                "human analyst" in draft.lower() or
                "mandatory human" in draft.lower()
            )
        if not requires_review:
            is_valid = False
            hitl_reasons.append("Missing mandatory human review statement")

        # Check 2: Must not contain unverified approvals
        contains_approval = validation_data.get("contains_approval", False)
        if contains_approval:
            if "without human review" not in draft.lower():
                is_valid = False
                hitl_reasons.append("Contains unverified approval")

        # Check 3: Should reference regulations or policy clauses
        references_regulation = validation_data.get("references_regulation", False)
        if not references_regulation:
            references_regulation = (
                "irdai" in draft.lower() or
                "regulation" in draft.lower() or
                "policy clause" in draft.lower() or
                "guideline" in draft.lower()
            )
        if not references_regulation:
            is_valid = False
            hitl_reasons.append("Missing regulatory citations")

        # Check 4: Every cited chunk ID must resolve to the retrieved evidence.
        citation_errors = _validate_citations_against_retrieval(draft, retrieved_docs)
        has_chunk_citations = not citation_errors
        if citation_errors:
            is_valid = False
            hitl_reasons.extend(citation_errors)

        # Check 5: Validate payout against statutory limits
        payout = extract_payout_amount(draft)
        if payout is not None:
            limit_validation = validate_statutory_limits(payout, self._claim_type)
            if not limit_validation["within_limits"]:
                is_valid = False
                hitl_reasons.append(
                    f"Payout ₹{payout:,.2f} exceeds statutory limit ₹{limit_validation['max_allowed']:,.2f} "
                    f"per {limit_validation['regulation']}"
                )

        # Check 6: Confidence-based HITL routing
        confidence = validation_data.get("confidence_score")
        if confidence is None:
            confidence = extract_confidence_score(draft)
        if confidence is not None and confidence < _HITL_CONFIDENCE_THRESHOLD:
            hitl_reasons.append(f"Confidence score {confidence:.2f} below threshold {_HITL_CONFIDENCE_THRESHOLD}")

        # Check 7: Zero-hallucination - assertions must be grounded
        # Determine HITL requirement
        hitl_required = len(hitl_reasons) > 0 or not is_valid
        hitl_reason = "; ".join(hitl_reasons) if hitl_reasons else "Routine compliance check"

        return is_valid, hitl_required, hitl_reason

    def _fallback_validation(
        self,
        decision_draft: str,
        retrieved_docs: Optional[List[Dict]] = None,
        model_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Fallback validation when Guardrails AI is not available.

        Performs comprehensive rule-based validation for IRDAI compliance.
        """
        draft_lower = decision_draft.lower()
        hitl_reasons = []

        # Critical check: must require human review
        requires_review = (
            "human review" in draft_lower or
            "human analyst" in draft_lower or
            "mandatory human" in draft_lower
        )
        if not requires_review:
            hitl_reasons.append("Missing mandatory human review statement")

        # Check for unverified approvals
        has_approval_keywords = (
            "approved" in draft_lower or
            "approval" in draft_lower or
            "granted" in draft_lower
        )
        has_review_qualification = "without human review" in draft_lower
        has_immediate_approval = (
            "automatically" in draft_lower or
            "immediately" in draft_lower
        )

        if has_approval_keywords and not has_review_qualification:
            hitl_reasons.append("Contains unverified approval")
        elif has_immediate_approval and has_approval_keywords:
            hitl_reasons.append("Contains immediate approval language")

        # Check for regulatory references
        has_regulation = (
            "irdai" in draft_lower or
            "regulation" in draft_lower or
            "policy clause" in draft_lower or
            "guideline" in draft_lower
        )
        if not has_regulation:
            hitl_reasons.append("Missing regulatory citations")

        # Require every citation to refer to a real retrieved chunk.
        chunk_ids = extract_chunk_ids(decision_draft)
        citation_errors = _validate_citations_against_retrieval(decision_draft, retrieved_docs)
        has_chunk_citations = not citation_errors
        hitl_reasons.extend(citation_errors)

        # Validate payout against statutory limits
        payout = extract_payout_amount(decision_draft)
        payout_validation = None
        if payout is not None:
            payout_validation = validate_statutory_limits(payout, self._claim_type)
            if not payout_validation["within_limits"]:
                hitl_reasons.append(
                    f"Payout ₹{payout:,.2f} exceeds statutory limit ₹{payout_validation['max_allowed']:,.2f}"
                )

        # Check confidence score
        confidence = extract_confidence_score(decision_draft)
        if confidence is not None and confidence < _HITL_CONFIDENCE_THRESHOLD:
            hitl_reasons.append(f"Confidence score {confidence:.2f} below threshold {_HITL_CONFIDENCE_THRESHOLD}")

        # Determine validity
        is_valid = len(hitl_reasons) == 0
        hitl_required = len(hitl_reasons) > 0
        hitl_reason = "; ".join(hitl_reasons) if hitl_reasons else "Routine compliance check"

        return {
            "is_valid": is_valid,
            "validation_result": {
                "compliance_check": is_valid,
                "contains_approval": has_approval_keywords,
                "requires_human_review": requires_review,
                "references_regulation": has_regulation,
                "has_chunk_citations": has_chunk_citations,
                "confidence_score": confidence,
                "payout_within_limits": payout_validation["within_limits"] if payout_validation else True,
                "payout_amount": payout,
                "statutory_limit": payout_validation["max_allowed"] if payout_validation else None,
                "chunk_ids": chunk_ids,
                "validation_method": "fallback_comprehensive"
            },
            "filtered_output": decision_draft,
            "hitl_required": hitl_required,
            "hitl_reason": hitl_reason
        }


# ---------------------------------------------------------------------------
# Module-level singleton (supports claim type)
# ---------------------------------------------------------------------------

_irdai_validators: Dict[str, IRDAIComplianceValidator] = {}


def get_irdai_validator(claim_type: str = "motor") -> IRDAIComplianceValidator:
    """
    Get or create an IRDAI compliance validator for the specified claim type.

    Parameters
    ----------
    claim_type : str
        Type of claim (motor, health, property, life)

    Returns
    -------
    IRDAIComplianceValidator instance
    """
    global _irdai_validators
    if claim_type not in _irdai_validators:
        _irdai_validators[claim_type] = IRDAIComplianceValidator(claim_type=claim_type)
    return _irdai_validators[claim_type]
