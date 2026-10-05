"""
Unit tests for src/claims_fraud.py.

Tests run offline — no network access required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.claims_fraud import (
    ClaimFeatures,
    FraudDetectionEngine,
    FraudScoringResult,
    _RuleBasedFraudScorer,
    _score_to_confidence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CLAIM = ClaimFeatures(
    claim_id="CLM-TEST-001",
    claim_amount=250_000.0,
    days_since_policy_start=120,
    num_prior_claims=2,
    claim_type="auto",
    claim_severity="medium",
    repair_shop_id="RS001",
    medical_provider_id=None,
    claimant_id="CLT-TEST-001",
)


@pytest.fixture(scope="module")
def engine() -> FraudDetectionEngine:
    """Shared engine instance; training happens once per test session."""
    return FraudDetectionEngine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFraudScoreRange:
    """Happy-path: engine loaded from CSV + real ML model."""

    def test_returns_fraud_scoring_result_type(self, engine: FraudDetectionEngine) -> None:
        result = engine.predict(_SAMPLE_CLAIM)
        assert isinstance(result, FraudScoringResult)

    def test_fraud_score_in_range(self, engine: FraudDetectionEngine) -> None:
        result = engine.predict(_SAMPLE_CLAIM)
        assert 0.0 <= result.fraud_score <= 1.0, f"fraud_score out of range: {result.fraud_score}"

    def test_fraud_flag_is_bool(self, engine: FraudDetectionEngine) -> None:
        result = engine.predict(_SAMPLE_CLAIM)
        assert isinstance(result.fraud_flag, bool), f"fraud_flag type: {type(result.fraud_flag)}"

    def test_fraud_flag_consistent_with_score(self, engine: FraudDetectionEngine) -> None:
        result = engine.predict(_SAMPLE_CLAIM)
        expected_flag = result.fraud_score >= 0.5
        assert result.fraud_flag == expected_flag

    def test_confidence_tier_valid(self, engine: FraudDetectionEngine) -> None:
        result = engine.predict(_SAMPLE_CLAIM)
        assert result.confidence_tier in {"low", "medium", "high"}

    def test_claim_id_echoed(self, engine: FraudDetectionEngine) -> None:
        result = engine.predict(_SAMPLE_CLAIM)
        assert result.claim_id == _SAMPLE_CLAIM.claim_id

    def test_model_version_present(self, engine: FraudDetectionEngine) -> None:
        result = engine.predict(_SAMPLE_CLAIM)
        assert result.model_version

    def test_timestamp_present(self, engine: FraudDetectionEngine) -> None:
        result = engine.predict(_SAMPLE_CLAIM)
        assert result.timestamp is not None

    def test_shap_drivers_is_list(self, engine: FraudDetectionEngine) -> None:
        result = engine.predict(_SAMPLE_CLAIM)
        assert isinstance(result.shap_drivers, list)


class TestScoreToConfidence:
    """Unit tests for the confidence-tier mapping helper."""

    def test_low_tier(self) -> None:
        assert _score_to_confidence(0.0) == "low"
        assert _score_to_confidence(0.39) == "low"

    def test_medium_tier(self) -> None:
        assert _score_to_confidence(0.40) == "medium"
        assert _score_to_confidence(0.69) == "medium"

    def test_high_tier(self) -> None:
        assert _score_to_confidence(0.70) == "high"
        assert _score_to_confidence(1.0) == "high"


class TestFraudFallback:
    """Engine must still initialise and score when the training CSV is absent."""

    def test_rule_based_scorer_predict_proba(self) -> None:
        import pandas as pd

        scorer = _RuleBasedFraudScorer()
        df = pd.DataFrame(
            [{"claim_amount": 300_000, "num_prior_claims": 3, "days_since_policy_start": 20}]
        )
        proba = scorer.predict_proba(df)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0][1] <= 1.0

    def test_early_claim_elevates_score(self) -> None:
        """Claims filed very early (< 30 days) should score higher than late ones."""
        import pandas as pd

        scorer = _RuleBasedFraudScorer()
        early = pd.DataFrame([{"claim_amount": 50_000, "num_prior_claims": 0, "days_since_policy_start": 5}])
        late = pd.DataFrame([{"claim_amount": 50_000, "num_prior_claims": 0, "days_since_policy_start": 500}])
        assert scorer.predict_proba(early)[0][1] > scorer.predict_proba(late)[0][1]

    def test_engine_with_patched_csv_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Monkey-patch the CSV path to a non-existent file; engine must
        initialise using the rule-based fallback and still score correctly.
        """
        import src.claims_fraud as cf

        fake_csv = tmp_path / "nonexistent_claims.csv"
        fake_model = tmp_path / "fraud_model.pkl"

        monkeypatch.setattr(cf, "_CLAIMS_CSV", fake_csv)
        monkeypatch.setattr(cf, "_MODEL_PATH", fake_model)
        monkeypatch.setattr(cf, "_MODELS_DIR", tmp_path)

        eng = cf.FraudDetectionEngine()
        result = eng.predict(_SAMPLE_CLAIM)

        assert isinstance(result, FraudScoringResult)
        assert 0.0 <= result.fraud_score <= 1.0
        assert isinstance(result.fraud_flag, bool)
        assert result.confidence_tier in {"low", "medium", "high"}
