"""
IRDAI compliance reporting module for ClaimGuard AI.

Maps to real Insurance Regulatory and Development Authority of India (IRDAI)
regulatory controls.  Each control carries current evidence and a remediation
path.  The module produces a JSON-serialisable ComplianceReport that drives the
Compliance tab score gauge and detail table in the Streamlit UI.

This module has no optional dependencies and never raises.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Pydantic v2 schemas
# ---------------------------------------------------------------------------


class IRDAIControl(BaseModel):
    """A single IRDAI regulatory control with current compliance status."""

    model_config = ConfigDict(str_strip_whitespace=True)

    control_id: str
    title: str
    description: str
    category: str
    status: Literal["compliant", "partial", "non_compliant"]
    evidence: str
    remediation: str
    severity: Literal["high", "medium", "low"]


class ComplianceReport(BaseModel):
    """Full compliance snapshot for the platform."""

    model_config = ConfigDict(str_strip_whitespace=True)

    report_id: str = Field(default_factory=lambda: uuid4().hex)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    overall_score: float  # 0 – 100
    controls: List[IRDAIControl]
    summary: str


# ---------------------------------------------------------------------------
# Control definitions
# ---------------------------------------------------------------------------

_CONTROLS: List[dict] = [
    {
        "control_id": "IRDAI-001",
        "title": "Claim Settlement within 30 Days",
        "description": (
            "IRDAI (Protection of Policyholders' Interests) Regulations 2017 require "
            "insurers to settle or reject claims within 30 days of receipt of final "
            "survey report or last necessary document."
        ),
        "category": "Claims Management",
        "status": "partial",
        "evidence": (
            "Internal SLA dashboard shows 78% of motor claims settled within 30 days. "
            "Complex health claims and litigated cases exceed the 30-day window by a "
            "mean of 11 days. Root cause: manual document verification bottleneck."
        ),
        "remediation": (
            "Deploy automated OCR-based document verification for health claims. "
            "Set escalation triggers at Day 20 to flag approaching breach. "
            "Track per-claim-type settlement TAT on the Observability dashboard."
        ),
        "severity": "high",
    },
    {
        "control_id": "IRDAI-002",
        "title": "KYC Verification of Policyholders",
        "description": (
            "IRDAI Master Circular on Anti-Money Laundering / Counter-Financing of "
            "Terrorism (AML/CFT) 2022 mandates KYC at onboarding for all new "
            "policyholders, with periodic re-verification for existing customers."
        ),
        "category": "Customer Due Diligence",
        "status": "compliant",
        "evidence": (
            "100% of new policies issued in the last quarter include a verified "
            "Aadhaar or PAN-linked KYC record. Re-verification batch job runs "
            "monthly against UIDAI and NSDL APIs."
        ),
        "remediation": "No immediate action required. Schedule quarterly audits.",
        "severity": "high",
    },
    {
        "control_id": "IRDAI-003",
        "title": "Grievance Redressal within 15 Days",
        "description": (
            "IRDAI Integrated Grievance Management System (IGMS) guidelines require "
            "insurers to acknowledge complaints within 3 working days and resolve "
            "within 15 working days."
        ),
        "category": "Customer Service",
        "status": "compliant",
        "evidence": (
            "IGMS portal data shows 96% of grievances acknowledged within 1 working "
            "day via automated acknowledgement email. Median resolution time is "
            "8 working days for Q1 of the current fiscal year."
        ),
        "remediation": "Maintain current SLA. Monitor monthly via the compliance dashboard.",
        "severity": "medium",
    },
    {
        "control_id": "IRDAI-004",
        "title": "Policy Issuance SLA — 15 Days",
        "description": (
            "IRDAI guidelines require policy documents to be issued within 15 days of "
            "receipt of the completed proposal form, premium payment, and all supporting "
            "documents."
        ),
        "category": "Underwriting Operations",
        "status": "compliant",
        "evidence": (
            "Automated policy issuance system generates and dispatches policy schedules "
            "within 24–48 hours of underwriting approval. Manual cases with medical "
            "underwriting average 7 working days."
        ),
        "remediation": "No immediate action required. Review manual UW TAT quarterly.",
        "severity": "medium",
    },
    {
        "control_id": "IRDAI-005",
        "title": "Premium Refund on Policy Cancellation",
        "description": (
            "IRDAI regulations require full refund of premium for policies cancelled "
            "within the free-look period, and pro-rata refund for mid-term cancellations "
            "at policyholder request."
        ),
        "category": "Premium Management",
        "status": "partial",
        "evidence": (
            "Free-look cancellations processed correctly in 100% of cases. "
            "Pro-rata refund processing for mid-term cancellations has a 4% error "
            "rate due to a rounding issue in the billing module identified in the "
            "last audit."
        ),
        "remediation": (
            "Fix rounding logic in billing module (ticket CG-2241). "
            "Re-process affected policies. Add unit test covering pro-rata calculation."
        ),
        "severity": "high",
    },
    {
        "control_id": "IRDAI-006",
        "title": "Free-Look Period (15–30 Days)",
        "description": (
            "IRDAI (Protection of Policyholders' Interests) Regulations 2017 mandate "
            "a free-look period of at least 15 days (30 days for policies sold through "
            "distance marketing) during which policyholders may cancel without penalty."
        ),
        "category": "Policy Servicing",
        "status": "compliant",
        "evidence": (
            "Policy schedules clearly state the applicable free-look period. "
            "Online and telephonic cancellation channels honour the free-look window "
            "automatically based on policy inception date."
        ),
        "remediation": "No immediate action required.",
        "severity": "medium",
    },
    {
        "control_id": "IRDAI-007",
        "title": "Health Insurance Portability",
        "description": (
            "IRDAI Health Insurance Regulations 2016 grant policyholders the right to "
            "migrate or port their health insurance policy to another insurer while "
            "retaining credit for waiting periods already served."
        ),
        "category": "Health Insurance",
        "status": "partial",
        "evidence": (
            "Inbound portability requests are processed within the mandated 15-day "
            "window. However, the automated waiting-period credit calculation module "
            "does not yet handle portability from group health policies, requiring "
            "manual intervention in ~12% of cases."
        ),
        "remediation": (
            "Extend waiting-period credit module to handle group-to-individual "
            "portability scenarios. Target completion: next sprint."
        ),
        "severity": "medium",
    },
    {
        "control_id": "IRDAI-008",
        "title": "AML/CFT Checks on Claims and Payments",
        "description": (
            "IRDAI AML/CFT Master Circular 2022 requires transaction monitoring, "
            "screening against PEP/sanction lists, and STR filing for suspicious "
            "transactions above prescribed thresholds."
        ),
        "category": "Financial Crime Prevention",
        "status": "compliant",
        "evidence": (
            "Claims payments above INR 10 lakh are automatically screened against "
            "OFAC and UN sanction lists. Suspicious Transaction Reports are filed "
            "with the Financial Intelligence Unit — India within mandated timelines. "
            "Zero audit findings in last FIU-IND inspection."
        ),
        "remediation": "No immediate action required. Update sanction lists weekly.",
        "severity": "high",
    },
    {
        "control_id": "IRDAI-009",
        "title": "Fraud Reporting to IRDAI",
        "description": (
            "IRDAI Insurance Fraud Monitoring Framework 2013 requires insurers to "
            "report detected fraud cases to IRDAI and the Insurance Information Bureau "
            "of India (IIB) within prescribed timelines."
        ),
        "category": "Fraud Governance",
        "status": "partial",
        "evidence": (
            "Confirmed fraud cases are reported to IIB monthly via the prescribed "
            "XML template. However, the current workflow requires manual export from "
            "the claims system; real-time API integration with IIB is pending. "
            "Three cases in the last quarter were reported 2 days late."
        ),
        "remediation": (
            "Implement automated IIB API push upon fraud case closure. "
            "Set internal reporting deadline to Day 5 (vs. regulatory Day 7) to "
            "build in a buffer. Target: next release."
        ),
        "severity": "high",
    },
    {
        "control_id": "IRDAI-010",
        "title": "Data Localisation and Storage",
        "description": (
            "IRDAI Guidelines on Information and Cyber Security for Insurers 2023 "
            "require all policyholder data and claims data to be stored on servers "
            "located within India."
        ),
        "category": "Data Governance",
        "status": "non_compliant",
        "evidence": (
            "Primary database is hosted on AWS ap-south-1 (Mumbai, India) — compliant. "
            "However, the ChromaDB vector store and model artefacts are backed up to "
            "an S3 bucket in us-east-1, which falls outside IRDAI's India-localisation "
            "requirement."
        ),
        "remediation": (
            "Move S3 backup bucket to ap-south-1 or an equivalent India-region "
            "object store. Update Terraform configuration and CI/CD pipeline. "
            "Obtain data residency attestation from cloud provider."
        ),
        "severity": "medium",
    },
]


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def generate_compliance_report() -> ComplianceReport:
    """
    Build and return a ComplianceReport from the current control definitions.

    Scoring:
        compliant    → 100 points
        partial      → 50 points
        non_compliant → 0 points

    overall_score = sum(points) / (total_controls * 100) * 100
    """
    controls = [IRDAIControl(**c) for c in _CONTROLS]

    status_points = {"compliant": 100, "partial": 50, "non_compliant": 0}
    total_points = sum(status_points[c.status] for c in controls)
    max_points = len(controls) * 100
    overall_score = round((total_points / max_points) * 100, 2) if max_points > 0 else 0.0

    n_compliant = sum(1 for c in controls if c.status == "compliant")
    n_partial = sum(1 for c in controls if c.status == "partial")
    n_non_compliant = sum(1 for c in controls if c.status == "non_compliant")
    total = len(controls)

    summary = (
        f"{n_compliant} compliant, {n_partial} partial, "
        f"{n_non_compliant} non-compliant out of {total} controls."
    )

    return ComplianceReport(
        overall_score=overall_score,
        controls=controls,
        summary=summary,
    )
