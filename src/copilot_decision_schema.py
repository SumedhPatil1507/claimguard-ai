"""
ClaimGuard AI - JSON Schema Enforcement Contract for /copilot/decide
======================================================================

This module defines the Pydantic schema for automated decision outputs
returned by the /copilot/decide endpoint, enforcing:
1. Structured decision format
2. Required citation fields with chunk IDs
3. Confidence-based HITL routing
4. IRDAI statutory limit compliance
5. Zero-hallucination validation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Citation Schemas
# ---------------------------------------------------------------------------

class PolicyClauseCitation(BaseModel):
    """Citation for a policy clause with chunk ID."""

    clause_id: str = Field(
        ...,
        description="Unique chunk ID from vector store (e.g., 'policy_section_4_2_chunk_1')",
        min_length=1,
        examples=["policy_section_4_2_chunk_1", "policy_motor_od_3_1_chunk_2"]
    )

    section: str = Field(
        ...,
        description="Policy section number (e.g., 'Section 4.2')",
        min_length=1,
        examples=["Section 4.2", "Section 5.1", "Clause 7.3"]
    )

    title: str = Field(
        ...,
        description="Title of the policy clause",
        min_length=1,
        examples=["Non-Disclosure of Material Facts", "Claim Filing Timeline"]
    )

    excerpt: str = Field(
        ...,
        description="Relevant excerpt from the policy clause (max 500 chars)",
        min_length=1,
        max_length=500
    )

    relevance_score: float = Field(
        ...,
        description="Retrieval relevance score from vector search (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

    @field_validator("relevance_score")
    @classmethod
    def validate_relevance(cls, v: float) -> float:
        """Ensure relevance score is in valid range."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Relevance score must be between 0.0 and 1.0")
        return v


class RegulatoryReference(BaseModel):
    """Citation for an IRDAI regulatory reference."""

    regulation_id: str = Field(
        ...,
        description="IRDAI regulation identifier",
        min_length=1,
        examples=["IRDAI-Motor-Claims-Guidelines-Cl-7", "IRDAI-Health-Regulations-2020"]
    )

    citation: str = Field(
        ...,
        description="Full regulatory citation text",
        min_length=1
    )

    applicability: str = Field(
        ...,
        description="Explanation of how this regulation applies to the current case",
        min_length=1
    )


class Citations(BaseModel):
    """Container for all citations."""

    policy_clauses: List[PolicyClauseCitation] = Field(
        default_factory=list,
        description="List of policy clause citations"
    )

    regulatory_references: List[RegulatoryReference] = Field(
        default_factory=list,
        description="List of IRDAI regulatory references"
    )

    @model_validator(mode="after")
    def validate_citations_present(self) -> "Citations":
        """Ensure at least one policy clause citation is present."""
        if not self.policy_clauses:
            raise ValueError("At least one policy clause citation is required")
        return self


# ---------------------------------------------------------------------------
# Decision Factor Schema
# ---------------------------------------------------------------------------

class DecisionFactor(BaseModel):
    """Single factor influencing the decision."""

    factor_name: str = Field(
        ...,
        description="Name of the factor",
        min_length=1,
        examples=["days_since_policy_start", "claim_amount", "credit_score"]
    )

    value: str = Field(
        ...,
        description="Value of the factor",
        min_length=1
    )

    impact: str = Field(
        ...,
        description="Impact on decision",
        examples=["increases_risk", "decreases_risk", "neutral"]
    )

    @field_validator("impact")
    @classmethod
    def validate_impact(cls, v: str) -> str:
        """Ensure impact is one of the allowed values."""
        allowed = ["increases_risk", "decreases_risk", "neutral"]
        if v not in allowed:
            raise ValueError(f"Impact must be one of {allowed}")
        return v


# ---------------------------------------------------------------------------
# Statutory Limits Schema
# ---------------------------------------------------------------------------

class StatutoryLimits(BaseModel):
    """IRDAI statutory limit validations."""

    max_allowed_payout: float = Field(
        ...,
        description="Maximum payout allowed per IRDAI regulations for this claim type",
        ge=0.0,
        examples=[750000.0, 10000000.0, 5000000.0]
    )

    within_limits: bool = Field(
        ...,
        description="True if recommended_payout does not exceed max_allowed_payout"
    )

    limit_source: str = Field(
        ...,
        description="Source of the statutory limit",
        min_length=1,
        examples=["IRDAI Motor TP Guidelines 2023", "IRDAI Health Insurance Regulations 2020"]
    )

    excess_amount: float = Field(
        default=0.0,
        description="Amount by which payout exceeds the limit (0 if within limits)",
        ge=0.0
    )


# ---------------------------------------------------------------------------
# Compliance Checks Schema
# ---------------------------------------------------------------------------

class ComplianceChecks(BaseModel):
    """IRDAI compliance validation results."""

    hallucination_check_passed: bool = Field(
        ...,
        description="True if all assertions are grounded in retrieved documents"
    )

    citation_check_passed: bool = Field(
        ...,
        description="True if at least one policy clause with chunk ID is cited"
    )

    limit_check_passed: bool = Field(
        ...,
        description="True if payout does not exceed statutory limits"
    )

    language_check_passed: bool = Field(
        ...,
        description="True if language is professional and regulatory-compliant"
    )

    compliance_notes: Optional[str] = Field(
        default=None,
        description="Any compliance issues or warnings"
    )


# ---------------------------------------------------------------------------
# Reasoning Schema
# ---------------------------------------------------------------------------

class Reasoning(BaseModel):
    """Decision reasoning with evidence."""

    summary: str = Field(
        ...,
        description="Executive summary of the decision (max 300 chars)",
        min_length=1,
        max_length=300
    )

    key_factors: List[DecisionFactor] = Field(
        default_factory=list,
        description="List of key factors influencing the decision"
    )

    evidence_summary: str = Field(
        ...,
        description="Summary of evidence from retrieved documents and model scoring",
        min_length=1
    )


# ---------------------------------------------------------------------------
# Main Decision Schema
# ---------------------------------------------------------------------------

class CopilotDecision(BaseModel):
    """
    Structured decision output from Policy Copilot with IRDAI compliance enforcement.

    This schema enforces:
    - Required citation fields with chunk IDs
    - Confidence-based HITL routing (< 0.85 threshold)
    - Statutory limit validation
    - Zero-hallucination policies
    """

    decision_type: str = Field(
        ...,
        description="Type of decision",
        examples=["recommendation", "escalation", "information_only"]
    )

    coverage_status: str = Field(
        ...,
        description="Coverage determination",
        examples=["covered", "not_covered", "partially_covered", "requires_review"]
    )

    recommended_payout: float = Field(
        ...,
        description="Recommended payout amount (0 if coverage is not_covered)",
        ge=0.0,
        le=100000000.0
    )

    confidence_score: float = Field(
        ...,
        description="Model confidence score (0.0 to 1.0). Values < 0.85 require mandatory HITL routing",
        ge=0.0,
        le=1.0
    )

    requires_human_review: bool = Field(
        ...,
        description="Mandatory flag: must be true if confidence_score < 0.85 or coverage_status is 'requires_review'"
    )

    citations: Citations = Field(
        ...,
        description="Policy clause and regulatory citations with chunk IDs"
    )

    statutory_limits: StatutoryLimits = Field(
        ...,
        description="IRDAI statutory limit validations"
    )

    reasoning: Reasoning = Field(
        ...,
        description="Decision reasoning with evidence"
    )

    compliance_checks: ComplianceChecks = Field(
        ...,
        description="IRDAI compliance validation results"
    )

    hitl_routing_reason: Optional[str] = Field(
        default=None,
        description="Reason for routing to Human-in-the-Loop (if requires_human_review is true)"
    )

    # Metadata fields
    session_id: str = Field(
        ...,
        description="Session identifier for traceability",
        min_length=1
    )

    timestamp: str = Field(
        ...,
        description="ISO timestamp of the decision",
        examples=["2024-01-15T10:30:00Z"]
    )

    model_version: str = Field(
        default="1.0.0",
        description="Version of the decision model"
    )

    @field_validator("decision_type")
    @classmethod
    def validate_decision_type(cls, v: str) -> str:
        """Ensure decision_type is valid."""
        allowed = ["recommendation", "escalation", "information_only"]
        if v not in allowed:
            raise ValueError(f"decision_type must be one of {allowed}")
        return v

    @field_validator("coverage_status")
    @classmethod
    def validate_coverage_status(cls, v: str) -> str:
        """Ensure coverage_status is valid."""
        allowed = ["covered", "not_covered", "partially_covered", "requires_review"]
        if v not in allowed:
            raise ValueError(f"coverage_status must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def validate_hitl_routing(self) -> "CopilotDecision":
        """Ensure HITL routing logic is consistent."""
        # If confidence < 0.85, requires_human_review must be true
        if self.confidence_score < 0.85 and not self.requires_human_review:
            raise ValueError(
                f"Confidence score {self.confidence_score} < 0.85 requires "
                "requires_human_review to be true"
            )

        # If coverage_status is 'requires_review', requires_human_review must be true
        if self.coverage_status == "requires_review" and not self.requires_human_review:
            raise ValueError(
                "coverage_status 'requires_review' requires requires_human_review to be true"
            )

        # If coverage is 'not_covered', recommended_payout must be 0
        if self.coverage_status == "not_covered" and self.recommended_payout > 0:
            raise ValueError(
                "coverage_status 'not_covered' requires recommended_payout to be 0"
            )

        # If requires_human_review is true, hitl_routing_reason must be provided
        if self.requires_human_review and not self.hitl_routing_reason:
            self.hitl_routing_reason = (
                f"Confidence score {self.confidence_score:.2f} below threshold 0.85"
                if self.confidence_score < 0.85
                else f"Coverage status '{self.coverage_status}' requires review"
            )

        return self

    @model_validator(mode="after")
    def validate_statutory_limits(self) -> "CopilotDecision":
        """Ensure statutory limits are consistent with payout."""
        if not self.statutory_limits.within_limits:
            # If payout exceeds limits, HITL must be required
            if not self.requires_human_review:
                raise ValueError(
                    f"Payout ₹{self.recommended_payout:,.2f} exceeds statutory limit "
                    f"₹{self.statutory_limits.max_allowed_payout:,.2f}, requires HITL routing"
                )
            # Update excess amount
            self.statutory_limits.excess_amount = (
                self.recommended_payout - self.statutory_limits.max_allowed_payout
            )
        return self

    @model_validator(mode="after")
    def validate_compliance_checks(self) -> "CopilotDecision":
        """Ensure compliance checks are consistent with decision."""
        # If citation_check_passed is false, at least one citation is required
        if not self.compliance_checks.citation_check_passed:
            if not self.citations.policy_clauses:
                raise ValueError(
                    "citation_check_passed is false but no policy clauses are cited"
                )

        # If hallucination_check_passed is false, HITL must be required
        if not self.compliance_checks.hallucination_check_passed:
            if not self.requires_human_review:
                raise ValueError(
                    "hallucination_check_passed is false requires HITL routing"
                )

        return self


# ---------------------------------------------------------------------------
# Request/Response Schemas for API
# ---------------------------------------------------------------------------

class CopilotDecisionRequest(BaseModel):
    """Request schema for /copilot/decide endpoint."""

    query: str = Field(
        ...,
        description="Natural-language question or case description",
        min_length=1
    )

    context_type: str = Field(
        default="underwriting",
        description="Context type: 'underwriting' or 'claims'",
        examples=["underwriting", "claims"]
    )

    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model input features"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for traceability"
    )

    claim_type: str = Field(
        default="motor",
        description="Claim type for statutory limit validation",
        examples=["motor", "health", "property", "life"]
    )


class CopilotDecisionResponse(BaseModel):
    """Response schema for /copilot/decide endpoint."""

    decision: CopilotDecision = Field(
        ...,
        description="Structured decision output"
    )

    validation_passed: bool = Field(
        ...,
        description="Whether the decision passed all validation checks"
    )

    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors (if any)"
    )

    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds"
    )
