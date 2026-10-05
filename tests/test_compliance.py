from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.compliance_irdai import generate_compliance_report, ComplianceReport

def test_generate_compliance_report_returns_correct_type():
    report = generate_compliance_report()
    assert isinstance(report, ComplianceReport)

def test_overall_score_in_range():
    report = generate_compliance_report()
    assert 0 <= report.overall_score <= 100

def test_minimum_controls():
    report = generate_compliance_report()
    assert len(report.controls) >= 5

def test_control_required_fields():
    report = generate_compliance_report()
    for ctrl in report.controls:
        assert ctrl.control_id, f'control_id missing on {ctrl}'
        assert ctrl.title, f'title missing on {ctrl}'
        assert ctrl.status in ('compliant', 'partial', 'non_compliant')
        assert ctrl.evidence, f'evidence missing on {ctrl.control_id}'
        assert ctrl.remediation, f'remediation missing on {ctrl.control_id}'

def test_report_metadata():
    report = generate_compliance_report()
    assert report.report_id
    assert report.generated_at is not None
