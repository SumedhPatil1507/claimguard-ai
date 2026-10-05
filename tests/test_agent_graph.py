from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.agent_graph import run_copilot_fallback, run_copilot

MINIMAL_UW_FEATURES = {
    'age': 35,
    'annual_income': 800000,
    'credit_score': 720,
    'sum_insured': 1000000,
    'coverage_type': 'motor',
    'num_dependents': 2,
    'prior_claims_count': 0,
    'region': 'north',
    'occupation': 'salaried',
}

MINIMAL_CLAIM_FEATURES = {
    'claim_id': 'CLM001',
    'claimant_id': 'CLMT001',
    'claim_amount': 50000.0,
    'days_since_policy_start': 90,
    'num_prior_claims': 0,
    'claim_type': 'motor',
    'claim_severity': 'medium',
    'repair_shop_id': 'SHOP001',
    'medical_provider_id': '',
    'policy_id': 'POL001',
}

def test_run_copilot_underwriting():
    result = run_copilot(
        query='Assess underwriting risk for new motor policy',
        context_type='underwriting',
        features=MINIMAL_UW_FEATURES,
    )
    assert result['requires_human_review'] is True
    assert isinstance(result['decision_draft'], str)
    assert len(result['decision_draft']) > 0
    assert result['session_id'] is not None

def test_run_copilot_claims():
    result = run_copilot(
        query='Evaluate fraud risk for motor claim',
        context_type='claims',
        features=MINIMAL_CLAIM_FEATURES,
    )
    assert result['requires_human_review'] is True
    assert isinstance(result['decision_draft'], str)

def test_run_copilot_fallback_direct():
    state = {
        'query': 'test query',
        'context_type': 'underwriting',
        'features': MINIMAL_UW_FEATURES,
        'retrieved_docs': [],
        'model_result': {},
        'decision_draft': '',
        'requires_human_review': False,
        'session_id': 'test-session-001',
        'timestamp': '2024-01-01T00:00:00Z',
    }
    result = run_copilot_fallback(state)
    assert result['requires_human_review'] is True
    assert result['decision_draft'] != ''

def test_graceful_missing_features():
    """run_copilot must not raise even if features dict has missing/wrong keys."""
    result = run_copilot(
        query='test',
        context_type='underwriting',
        features={},
    )
    assert isinstance(result, dict)
    assert 'decision_draft' in result
