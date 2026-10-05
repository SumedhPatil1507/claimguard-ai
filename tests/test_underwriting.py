"""
Unit tests for src/underwriting.py.

Tests run offline — no network access required.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src.underwriting import (
    UnderwritingEngine,
    UnderwritingFeatures,
    UnderwritingResult,
    _RuleBasedScorer,
    _score_to_tier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_FEATURES = UnderwritingFeatures(
    age=35,
    annual_income=800_000.0,
    credit_score=650,
    sum_insured=2_000_000.0,
    coverage_type="comprehensive",
    num_dependents=2,
    prior_claims_count=1,
    region="South",
    occupation="salaried",
)


@pytest.fixture(scope="module")
def engine() -> UnderwritingEngine:
    """Shared engine instance; training happens once per test session."""
    return UnderwritingEngine()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictReturnsValidResult:
    """Happy-path: engine loaded from CSV + real ML model."""

    def test_returns_underwriting_result_type(self, engine: UnderwritingEngine) -> None:
        result = engine.predict(_SAMPLE_FEATURES)
        assert isinstance(result, UnderwritingResult)

    def test_risk_score_in_range(self, engine: UnderwritingEngine) -> None:
        result = engine.predict(_SAMPLE_FEATURES)
        assert 0.0 <= result.risk_score <= 1.0, f"risk_score out of range: {result.risk_score}"

    def test_risk_tier_valid(self, engine: UnderwritingEngine) -> None:
        result = engine.predict(_SAMPLE_FEATURES)
        assert result.risk_tier in {"low", "medium", "high"}, f"unexpected tier: {result.risk_tier}"

    def test_premium_adjustment_positive(self, engine: UnderwritingEngine) -> None:
        result = engine.predict(_SAMPLE_FEATURES)
        assert result.premium_adjustment > 0, f"premium_adjustment <= 0: {result.premium_adjustment}"

    def test_premium_adjustment_bounds(self, engine: UnderwritingEngine) -> None:
        """Adjustment is 0.8 + risk_score * 0.7, so range is [0.8, 1.5]."""
        result = engine.predict(_SAMPLE_FEATURES)
        assert 0.79 <= result.premium_adjustment <= 1.51

    def test_model_version_present(self, engine: UnderwritingEngine) -> None:
        result = engine.predict(_SAMPLE_FEATURES)
        assert result.model_version

    def test_timestamp_present(self, engine: UnderwritingEngine) -> None:
        result = engine.predict(_SAMPLE_FEATURES)
        assert result.timestamp is not None

    def test_shap_drivers_is_list(self, engine: UnderwritingEngine) -> None:
        result = engine.predict(_SAMPLE_FEATURES)
        assert isinstance(result.shap_drivers, list)

    def test_shap_driver_structure(self, engine: UnderwritingEngine) -> None:
        result = engine.predict(_SAMPLE_FEATURES)
        for driver in result.shap_drivers:
            assert "feature" in driver
            assert "shap_value" in driver
            assert "direction" in driver
            assert driver["direction"] in {"increases_risk", "decreases_risk"}


class TestScoreToTier:
    """Unit tests for the tier-mapping helper."""

    def test_low(self) -> None:
        assert _score_to_tier(0.0) == "low"
        assert _score_to_tier(0.33) == "low"

    def test_medium(self) -> None:
        assert _score_to_tier(0.34) == "medium"
        assert _score_to_tier(0.66) == "medium"

    def test_high(self) -> None:
        assert _score_to_tier(0.67) == "high"
        assert _score_to_tier(1.0) == "high"


class TestFallbackWhenNoCSV:
    """Engine must still initialise and score when the training CSV is absent."""

    def test_rule_based_scorer_predict_proba(self) -> None:
        """_RuleBasedScorer itself returns valid probabilities."""
        import pandas as pd

        scorer = _RuleBasedScorer()
        df = pd.DataFrame(
            [{"credit_score": 500, "prior_claims_count": 2}]
        )
        proba = scorer.predict_proba(df)
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0][1] <= 1.0

    def test_engine_with_patched_csv_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Monkey-patch the CSV path to a non-existent file and the model path
        to tmp_path so training falls back to rule-based and saves a new pkl.
        """
        import src.underwriting as uw

        fake_csv = tmp_path / "nonexistent.csv"
        fake_model = tmp_path / "underwriting_model.pkl"

        monkeypatch.setattr(uw, "_POLICIES_CSV", fake_csv)
        monkeypatch.setattr(uw, "_MODEL_PATH", fake_model)
        monkeypatch.setattr(uw, "_MODELS_DIR", tmp_path)

        eng = uw.UnderwritingEngine()
        result = eng.predict(_SAMPLE_FEATURES)

        assert isinstance(result, UnderwritingResult)
        assert 0.0 <= result.risk_score <= 1.0
        assert result.risk_tier in {"low", "medium", "high"}
        assert result.premium_adjustment > 0
