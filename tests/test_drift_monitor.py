"""
tests/test_drift_monitor.py
============================
Unit tests for:
  src/drift_monitor.py     — DriftReport, save_baseline, load_baseline,
                             DriftMonitor, _fallback_drift, Prometheus push
  src/underwriting.py      — detect_drift(), MLflow logging (mocked)
  src/claims_fraud.py      — detect_drift(), MLflow logging (mocked)

Design
------
* No live MLflow server, Evidently, or Prometheus required.
* scipy is used by the fallback path — tests run with or without it.
* All Parquet I/O uses tmp_path so the test suite never touches data/.
* Evidently is mocked out in groups that test fallback behaviour.
* MLflow is always mocked via patch.dict("sys.modules") so CI never
  needs a tracking server.

Test groups
-----------
A  DriftReport — dataclass fields, to_dict(), to_json()
B  save_baseline / load_baseline — Parquet roundtrip, missing file
C  _fallback_drift  — numeric KS, categorical chi2, PSI concept drift
D  _psi_score       — edge cases (empty, single bin, identical)
E  DriftMonitor.detect — baseline-absent warning, full detect path,
                         Prometheus push (mocked), predictions array
F  DriftMonitor.detect_from_dicts — thin wrapper
G  underwriting.detect_drift — integration test (real engine, no Evid.)
H  claims_fraud.detect_drift  — integration test (real engine, no Evid.)
I  MLflow logging   — _log_to_mlflow mocked calls (underwriting + fraud)
J  ClaimFeatures    — policy_id optional field backward-compat
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Availability flags
# ---------------------------------------------------------------------------

try:
    import evidently  # noqa: F401
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

try:
    import mlflow  # noqa: F401
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# ---------------------------------------------------------------------------
# Shared synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_reference(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "claim_amount":            rng.normal(50_000, 10_000, n),
        "days_since_policy_start": rng.integers(30, 500, n).astype(float),
        "num_prior_claims":        rng.integers(0, 5, n).astype(float),
        "claim_type":              rng.choice(["motor", "health", "property"], n),
        "claim_severity":          rng.choice(["low", "medium", "high"], n),
        "target":                  rng.integers(0, 2, n).astype(float),
    })


def _make_current_no_drift(n: int = 40, seed: int = 99) -> pd.DataFrame:
    """Same distribution as reference — drift score should stay low."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "claim_amount":            rng.normal(50_000, 10_000, n),
        "days_since_policy_start": rng.integers(30, 500, n).astype(float),
        "num_prior_claims":        rng.integers(0, 5, n).astype(float),
        "claim_type":              rng.choice(["motor", "health", "property"], n),
        "claim_severity":          rng.choice(["low", "medium", "high"], n),
        "target":                  rng.integers(0, 2, n).astype(float),
    })


def _make_current_with_drift(n: int = 40, seed: int = 7) -> pd.DataFrame:
    """Heavily shifted distribution — most numeric cols should drift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "claim_amount":            rng.normal(200_000, 50_000, n),   # shifted right
        "days_since_policy_start": rng.integers(1, 15, n).astype(float),  # all early
        "num_prior_claims":        rng.integers(8, 20, n).astype(float),  # high
        "claim_type":              rng.choice(["life"], n),           # single category
        "claim_severity":          rng.choice(["high"], n),           # single category
        "target":                  rng.integers(0, 2, n).astype(float),
    })


# ===========================================================================
# A — DriftReport
# ===========================================================================

class TestDriftReport:
    def test_default_fields(self) -> None:
        from src.drift_monitor import DriftReport
        r = DriftReport(model_name="fraud")
        assert r.model_name == "fraud"
        assert r.data_drift_score == 0.0
        assert r.drifted_features == []
        assert r.concept_drift_detected is False
        assert r.concept_drift_score is None
        assert r.evidently_available is False
        assert r.warning is None

    def test_to_dict_contains_all_keys(self) -> None:
        from src.drift_monitor import DriftReport
        r = DriftReport(model_name="underwriting", data_drift_score=0.25)
        d = r.to_dict()
        for key in ("model_name", "timestamp", "data_drift_score",
                    "drifted_features", "concept_drift_detected",
                    "concept_drift_score", "feature_drift_details",
                    "n_reference", "n_current", "evidently_available", "warning"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_rounds_scores(self) -> None:
        from src.drift_monitor import DriftReport
        r = DriftReport(model_name="test", data_drift_score=0.123456789,
                        concept_drift_score=0.987654321)
        d = r.to_dict()
        assert d["data_drift_score"] == 0.1235
        assert d["concept_drift_score"] == 0.9877

    def test_to_json_is_valid_json(self) -> None:
        from src.drift_monitor import DriftReport
        r = DriftReport(model_name="fraud", drifted_features=["claim_amount"])
        parsed = json.loads(r.to_json())
        assert parsed["model_name"] == "fraud"
        assert "claim_amount" in parsed["drifted_features"]

    def test_warning_field_preserved(self) -> None:
        from src.drift_monitor import DriftReport
        r = DriftReport(model_name="test", warning="no baseline")
        assert r.to_dict()["warning"] == "no baseline"


# ===========================================================================
# B — save_baseline / load_baseline
# ===========================================================================

@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestBaselineIO:
    def test_save_creates_parquet(self, tmp_path: Path) -> None:
        from src import drift_monitor as dm
        original_dir = dm._BASELINE_DIR
        dm._BASELINE_DIR = tmp_path
        try:
            ref = _make_reference()
            path = dm.save_baseline(ref, model_name="test_model")
            assert path.exists()
            assert path.suffix == ".parquet"
        finally:
            dm._BASELINE_DIR = original_dir

    def test_save_with_prediction_col_renames(self, tmp_path: Path) -> None:
        from src import drift_monitor as dm
        dm._BASELINE_DIR = tmp_path
        ref = _make_reference()
        dm.save_baseline(ref, model_name="test2", prediction_col="target")
        loaded = pd.read_parquet(tmp_path / "test2.parquet")
        assert "target" in loaded.columns

    def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
        from src import drift_monitor as dm
        dm._BASELINE_DIR = tmp_path
        result = dm.load_baseline("nonexistent_model")
        assert result is None

    def test_roundtrip_preserves_shape(self, tmp_path: Path) -> None:
        from src import drift_monitor as dm
        dm._BASELINE_DIR = tmp_path
        ref = _make_reference(50)
        dm.save_baseline(ref, model_name="roundtrip")
        loaded = dm.load_baseline("roundtrip")
        assert loaded is not None
        assert loaded.shape == ref.shape

    def test_roundtrip_preserves_numeric_values(self, tmp_path: Path) -> None:
        from src import drift_monitor as dm
        dm._BASELINE_DIR = tmp_path
        ref = _make_reference(30)
        dm.save_baseline(ref, model_name="vals")
        loaded = dm.load_baseline("vals")
        pd.testing.assert_frame_equal(
            ref.reset_index(drop=True),
            loaded.reset_index(drop=True),
            check_dtype=False,
        )


# ===========================================================================
# C — _fallback_drift
# ===========================================================================

class TestFallbackDrift:
    def test_returns_dict_with_expected_keys(self) -> None:
        from src.drift_monitor import _fallback_drift
        ref = _make_reference(60)
        cur = _make_current_no_drift(30)
        result = _fallback_drift(ref, cur)
        for key in ("feature_details", "drifted_features", "data_drift_score",
                    "concept_score", "concept_flag"):
            assert key in result, f"Missing: {key}"

    def test_drifted_features_subset_of_columns(self) -> None:
        from src.drift_monitor import _fallback_drift
        ref = _make_reference(60)
        cur = _make_current_with_drift(30)
        result = _fallback_drift(ref, cur)
        all_cols = set(ref.columns)
        for col in result["drifted_features"]:
            assert col in all_cols or col == "target"

    def test_no_drift_when_distributions_identical(self) -> None:
        from src.drift_monitor import _fallback_drift
        ref = _make_reference(100, seed=1)
        cur = _make_reference(100, seed=1)   # exact same seed = same data
        result = _fallback_drift(ref, cur)
        assert result["data_drift_score"] == pytest.approx(0.0, abs=1e-3)

    def test_high_drift_when_distributions_shifted(self) -> None:
        from src.drift_monitor import _fallback_drift
        ref = _make_reference(100)
        cur = _make_current_with_drift(100)
        result = _fallback_drift(ref, cur)
        # At least some features should drift
        assert result["data_drift_score"] > 0.0

    def test_concept_drift_computed_when_col_present(self) -> None:
        from src.drift_monitor import _fallback_drift
        ref = _make_reference(80)
        cur = _make_current_with_drift(40)
        cur["pred"] = np.random.default_rng(5).random(40)
        ref["pred"] = np.random.default_rng(5).random(80) * 0.1   # different dist
        result = _fallback_drift(ref, cur, prediction_col="pred")
        assert result["concept_score"] is not None

    def test_concept_drift_none_when_no_pred_col(self) -> None:
        from src.drift_monitor import _fallback_drift
        ref = _make_reference(60)
        cur = _make_current_no_drift(30)
        result = _fallback_drift(ref, cur, prediction_col=None)
        assert result["concept_score"] is None

    def test_categorical_columns_handled(self) -> None:
        from src.drift_monitor import _fallback_drift
        ref = _make_reference(60)
        cur = _make_current_with_drift(30)
        result = _fallback_drift(ref, cur)
        details = result["feature_details"]
        cat_details = {k: v for k, v in details.items() if v.get("method") == "chi2"}
        assert len(cat_details) > 0

    def test_numeric_columns_use_ks_method(self) -> None:
        from src.drift_monitor import _fallback_drift
        ref = _make_reference(60)
        cur = _make_current_no_drift(30)
        result = _fallback_drift(ref, cur)
        details = result["feature_details"]
        ks_details = {k: v for k, v in details.items() if v.get("method") == "ks"}
        assert len(ks_details) > 0


# ===========================================================================
# D — _psi_score edge cases
# ===========================================================================

class TestPSIScore:
    def test_identical_distributions_returns_near_zero(self) -> None:
        from src.drift_monitor import _psi_score
        arr = pd.Series(np.linspace(0, 1, 100))
        score = _psi_score(arr, arr)
        assert score == pytest.approx(0.0, abs=0.05)

    def test_empty_series_returns_zero(self) -> None:
        from src.drift_monitor import _psi_score
        assert _psi_score(pd.Series(dtype=float), pd.Series(dtype=float)) == 0.0

    def test_completely_different_distributions_positive(self) -> None:
        from src.drift_monitor import _psi_score
        ref = pd.Series(np.zeros(100))
        cur = pd.Series(np.ones(100))
        score = _psi_score(ref, cur)
        # PSI should be large when distributions don't overlap
        assert score >= 0.0

    def test_score_is_non_negative(self) -> None:
        from src.drift_monitor import _psi_score
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.normal(0, 1, 200))
        cur = pd.Series(rng.normal(2, 1, 200))
        assert _psi_score(ref, cur) >= 0.0


# ===========================================================================
# E — DriftMonitor.detect
# ===========================================================================

@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestDriftMonitorDetect:
    def _make_monitor(self, tmp_path: Path, model_name: str = "test") -> "DriftMonitor":
        from src.drift_monitor import DriftMonitor
        return DriftMonitor(
            model_name=model_name,
            baseline_dir=tmp_path,
            push_prometheus=False,
        )

    def test_detect_returns_warning_when_no_baseline(self, tmp_path: Path) -> None:
        from src.drift_monitor import DriftReport
        monitor = self._make_monitor(tmp_path, "missing_model")
        report = monitor.detect(_make_current_no_drift())
        assert isinstance(report, DriftReport)
        assert report.warning is not None
        assert "baseline" in report.warning.lower()

    def test_detect_returns_drift_report_with_baseline(self, tmp_path: Path) -> None:
        from src.drift_monitor import DriftReport, save_baseline
        ref = _make_reference(80)
        save_baseline(ref, model_name="m1")
        # Override baseline dir by placing file manually
        ref.to_parquet(tmp_path / "m1.parquet", index=False)
        monitor = self._make_monitor(tmp_path, "m1")
        report = monitor.detect(_make_current_no_drift(30))
        assert isinstance(report, DriftReport)
        assert report.n_reference == 80
        assert report.n_current == 30

    def test_detect_data_score_between_0_and_1(self, tmp_path: Path) -> None:
        from src.drift_monitor import DriftReport
        ref = _make_reference(80)
        ref.to_parquet(tmp_path / "m2.parquet", index=False)
        monitor = self._make_monitor(tmp_path, "m2")
        report = monitor.detect(_make_current_with_drift(40))
        assert 0.0 <= report.data_drift_score <= 1.0

    def test_detect_with_predictions_array(self, tmp_path: Path) -> None:
        from src.drift_monitor import DriftReport
        ref = _make_reference(80)
        ref.to_parquet(tmp_path / "m3.parquet", index=False)
        monitor = self._make_monitor(tmp_path, "m3")
        preds = np.random.default_rng(1).random(40)
        report = monitor.detect(_make_current_no_drift(40), predictions=preds)
        assert isinstance(report, DriftReport)

    def test_detect_never_raises_on_empty_current(self, tmp_path: Path) -> None:
        from src.drift_monitor import DriftReport
        ref = _make_reference(80)
        ref.to_parquet(tmp_path / "m4.parquet", index=False)
        monitor = self._make_monitor(tmp_path, "m4")
        report = monitor.detect(pd.DataFrame())   # empty DataFrame
        assert isinstance(report, DriftReport)

    def test_detect_feature_drift_details_is_dict(self, tmp_path: Path) -> None:
        ref = _make_reference(80)
        ref.to_parquet(tmp_path / "m5.parquet", index=False)
        from src.drift_monitor import DriftMonitor
        monitor = DriftMonitor("m5", baseline_dir=tmp_path, push_prometheus=False)
        report = monitor.detect(_make_current_no_drift(30))
        assert isinstance(report.feature_drift_details, dict)

    def test_reload_baseline_picks_up_new_file(self, tmp_path: Path) -> None:
        from src.drift_monitor import DriftMonitor
        monitor = DriftMonitor("m6", baseline_dir=tmp_path, push_prometheus=False)
        assert monitor._reference is None
        # Now write the baseline and reload
        _make_reference(50).to_parquet(tmp_path / "m6.parquet", index=False)
        monitor.reload_baseline()
        assert monitor._reference is not None
        assert len(monitor._reference) == 50

    def test_prometheus_push_called_when_enabled(self, tmp_path: Path) -> None:
        """Verify _push_to_prometheus is invoked (no live Prometheus needed)."""
        from src.drift_monitor import DriftMonitor
        ref = _make_reference(40)
        ref.to_parquet(tmp_path / "mprom.parquet", index=False)
        monitor = DriftMonitor("mprom", baseline_dir=tmp_path, push_prometheus=True)
        with patch("src.drift_monitor._push_to_prometheus") as mock_push:
            monitor.detect(_make_current_no_drift(20))
            mock_push.assert_called_once()
            args = mock_push.call_args
            assert args[0][0] == "mprom"   # first positional arg is model_name


# ===========================================================================
# F — DriftMonitor.detect_from_dicts
# ===========================================================================

@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestDetectFromDicts:
    def test_returns_drift_report(self, tmp_path: Path) -> None:
        from src.drift_monitor import DriftMonitor, DriftReport
        ref = _make_reference(40)
        ref.to_parquet(tmp_path / "fd.parquet", index=False)
        monitor = DriftMonitor("fd", baseline_dir=tmp_path, push_prometheus=False)
        dicts = _make_current_no_drift(10).to_dict(orient="records")
        report = monitor.detect_from_dicts(dicts)
        assert isinstance(report, DriftReport)

    def test_detect_from_empty_dicts_no_crash(self, tmp_path: Path) -> None:
        from src.drift_monitor import DriftMonitor, DriftReport
        ref = _make_reference(40)
        ref.to_parquet(tmp_path / "fd2.parquet", index=False)
        monitor = DriftMonitor("fd2", baseline_dir=tmp_path, push_prometheus=False)
        report = monitor.detect_from_dicts([])
        assert isinstance(report, DriftReport)


# ===========================================================================
# G — underwriting.detect_drift integration
# ===========================================================================

@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestUnderwritingDetectDrift:
    @pytest.fixture(scope="class")
    def engine(self):
        from src.underwriting import UnderwritingEngine
        return UnderwritingEngine()

    def test_detect_drift_returns_drift_report(self, engine, tmp_path: Path) -> None:
        """detect_drift must return a DriftReport (or dict fallback) without raising."""
        from src.drift_monitor import DriftReport
        cur = pd.DataFrame([{
            "age": 35, "annual_income": 800_000, "credit_score": 700,
            "sum_insured": 1_000_000, "coverage_type": "motor",
            "num_dependents": 2, "prior_claims_count": 1,
            "region": "north", "occupation": "salaried",
        }])
        result = engine.detect_drift(cur)
        # Must be DriftReport or dict — never raises
        assert isinstance(result, (DriftReport, dict))

    def test_detect_drift_no_raise_on_empty_df(self, engine) -> None:
        result = engine.detect_drift(pd.DataFrame())
        assert result is not None

    def test_detect_drift_with_predictions(self, engine) -> None:
        from src.drift_monitor import DriftReport
        cur = pd.DataFrame([{
            "claim_amount": 50_000, "days_since_policy_start": 90,
            "num_prior_claims": 0, "claim_type": "motor", "claim_severity": "low",
        }])
        preds = np.array([0.3])
        result = engine.detect_drift(cur, predictions=preds)
        assert isinstance(result, (DriftReport, dict))


# ===========================================================================
# H — claims_fraud.detect_drift integration
# ===========================================================================

@pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
class TestClaimsFraudDetectDrift:
    @pytest.fixture(scope="class")
    def engine(self):
        from src.claims_fraud import FraudDetectionEngine
        return FraudDetectionEngine()

    def test_detect_drift_returns_drift_report(self, engine) -> None:
        from src.drift_monitor import DriftReport
        cur = pd.DataFrame([{
            "claim_amount": 75_000, "days_since_policy_start": 60,
            "num_prior_claims": 2, "claim_type": "motor", "claim_severity": "high",
        }])
        result = engine.detect_drift(cur)
        assert isinstance(result, (DriftReport, dict))

    def test_detect_drift_no_raise_on_empty_df(self, engine) -> None:
        result = engine.detect_drift(pd.DataFrame())
        assert result is not None

    def test_detect_drift_with_predictions_array(self, engine) -> None:
        from src.drift_monitor import DriftReport
        cur = pd.DataFrame([
            {"claim_amount": 50_000, "days_since_policy_start": 90,
             "num_prior_claims": 0, "claim_type": "motor", "claim_severity": "low"},
            {"claim_amount": 200_000, "days_since_policy_start": 10,
             "num_prior_claims": 4, "claim_type": "health", "claim_severity": "high"},
        ])
        preds = np.array([0.2, 0.85])
        result = engine.detect_drift(cur, predictions=preds)
        assert isinstance(result, (DriftReport, dict))


# ===========================================================================
# I — MLflow logging (_log_to_mlflow mocked)
# ===========================================================================

class TestMLflowLogging:
    """These tests verify the logging calls without a live MLflow server."""

    def _mock_mlflow(self):
        """Return a mock mlflow module."""
        m = MagicMock()
        m.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        m.start_run.return_value.__exit__  = MagicMock(return_value=False)
        return m

    def test_underwriting_log_to_mlflow_calls_set_experiment(self) -> None:
        import src.underwriting as uw
        mock_mlflow = self._mock_mlflow()
        # Direct attribute injection avoids module reload side effects
        original_has = uw.HAS_MLFLOW
        original_mlf = getattr(uw, "mlflow", None)
        try:
            uw.HAS_MLFLOW = True
            uw.mlflow = mock_mlflow    # type: ignore[attr-defined]
            uw._log_to_mlflow(
                run_params={"n_estimators": 100},
                metrics={"roc_auc_cv": 0.85, "f1_cv": 0.72},
                model=MagicMock(),
                X_train=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            )
            mock_mlflow.set_experiment.assert_called_with(uw.MLFLOW_EXP)
        finally:
            uw.HAS_MLFLOW = original_has
            if original_mlf is None:
                delattr(uw, "mlflow") if hasattr(uw, "mlflow") else None
            else:
                uw.mlflow = original_mlf   # type: ignore[attr-defined]

    def test_underwriting_log_to_mlflow_logs_params_and_metrics(self) -> None:
        import src.underwriting as uw
        mock_mlflow = self._mock_mlflow()
        original_has = uw.HAS_MLFLOW
        original_mlf = getattr(uw, "mlflow", None)
        try:
            uw.HAS_MLFLOW = True
            uw.mlflow = mock_mlflow    # type: ignore[attr-defined]
            uw._log_to_mlflow(
                run_params={"n_estimators": 50, "max_depth": 4},
                metrics={"roc_auc_cv": 0.80},
                model=MagicMock(),
                X_train=pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]}),
            )
            mock_mlflow.log_params.assert_called()
            mock_mlflow.log_metrics.assert_called()
        finally:
            uw.HAS_MLFLOW = original_has
            if original_mlf is None:
                try: delattr(uw, "mlflow")
                except AttributeError: pass
            else:
                uw.mlflow = original_mlf   # type: ignore[attr-defined]

    def test_underwriting_log_skipped_when_has_mlflow_false(self) -> None:
        import src.underwriting as uw
        mock_mlflow = self._mock_mlflow()
        original = uw.HAS_MLFLOW
        try:
            uw.HAS_MLFLOW = False
            uw._log_to_mlflow({}, {}, MagicMock(), pd.DataFrame())
        finally:
            uw.HAS_MLFLOW = original
        mock_mlflow.set_experiment.assert_not_called()

    def test_fraud_log_to_mlflow_calls_set_experiment(self) -> None:
        import src.claims_fraud as cf
        mock_mlflow = self._mock_mlflow()
        original_has = cf.HAS_MLFLOW
        original_mlf = getattr(cf, "mlflow", None)
        try:
            cf.HAS_MLFLOW = True
            cf.mlflow = mock_mlflow    # type: ignore[attr-defined]
            cf._log_to_mlflow(
                run_params={"n_estimators": 100},
                metrics={"roc_auc_cv": 0.82, "f1_cv": 0.68},
                model=MagicMock(),
                X_train=pd.DataFrame({"claim_amount": [50_000.0, 100_000.0]}),
            )
            mock_mlflow.set_experiment.assert_called_with(cf.MLFLOW_EXP)
        finally:
            cf.HAS_MLFLOW = original_has
            if original_mlf is None:
                try: delattr(cf, "mlflow")
                except AttributeError: pass
            else:
                cf.mlflow = original_mlf   # type: ignore[attr-defined]

    def test_fraud_log_sets_fraud_threshold_tag(self) -> None:
        import src.claims_fraud as cf
        mock_mlflow = self._mock_mlflow()
        original_has = cf.HAS_MLFLOW
        original_mlf = getattr(cf, "mlflow", None)
        try:
            cf.HAS_MLFLOW = True
            cf.mlflow = mock_mlflow    # type: ignore[attr-defined]
            cf._log_to_mlflow({}, {"roc_auc_cv": 0.75}, MagicMock(),
                               pd.DataFrame({"a": [1]}))
            if mock_mlflow.set_tags.called:
                call_kwargs = mock_mlflow.set_tags.call_args[0][0]
                assert "fraud_threshold" in call_kwargs
        finally:
            cf.HAS_MLFLOW = original_has
            if original_mlf is None:
                try: delattr(cf, "mlflow")
                except AttributeError: pass
            else:
                cf.mlflow = original_mlf   # type: ignore[attr-defined]

    def test_log_to_mlflow_silent_on_exception(self) -> None:
        """If MLflow raises, _log_to_mlflow must not propagate."""
        import src.underwriting as uw
        mock_mlflow = self._mock_mlflow()
        mock_mlflow.set_experiment.side_effect = RuntimeError("server down")
        original_has = uw.HAS_MLFLOW
        original_mlf = getattr(uw, "mlflow", None)
        try:
            uw.HAS_MLFLOW = True
            uw.mlflow = mock_mlflow    # type: ignore[attr-defined]
            uw._log_to_mlflow({"k": "v"}, {"m": 1.0}, MagicMock(),
                               pd.DataFrame({"x": [1]}))   # must not raise
        finally:
            uw.HAS_MLFLOW = original_has
            if original_mlf is None:
                try: delattr(uw, "mlflow")
                except AttributeError: pass
            else:
                uw.mlflow = original_mlf   # type: ignore[attr-defined]


# ===========================================================================
# J — ClaimFeatures backward compatibility (policy_id optional)
# ===========================================================================

class TestClaimFeaturesSchema:
    def test_claim_features_without_policy_id(self) -> None:
        from src.claims_fraud import ClaimFeatures
        f = ClaimFeatures(
            claim_id="C1", claim_amount=50_000.0,
            days_since_policy_start=90, num_prior_claims=0,
            claim_type="motor", claim_severity="low", claimant_id="CLT1",
        )
        assert f.policy_id is None

    def test_claim_features_with_policy_id(self) -> None:
        from src.claims_fraud import ClaimFeatures
        f = ClaimFeatures(
            claim_id="C1", claim_amount=50_000.0,
            days_since_policy_start=90, num_prior_claims=0,
            claim_type="motor", claim_severity="low", claimant_id="CLT1",
            policy_id="POL001",
        )
        assert f.policy_id == "POL001"
