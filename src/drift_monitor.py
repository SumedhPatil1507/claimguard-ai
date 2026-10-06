"""
src/drift_monitor.py
====================
Data Drift and Concept Drift monitoring for ClaimGuard AI.

Uses Evidently AI to compare incoming feature distributions against a saved
baseline, then exports the resulting drift scores as Prometheus gauges so
Grafana dashboards stay current in real time.

Architecture
------------
                 ┌──────────────┐
  Training data  │  save_        │
  ─────────────► │  baseline()  │ ──► data/baselines/<model>.parquet
                 └──────────────┘

  Inference      ┌──────────────────────────────────────────┐
  batch          │  DriftMonitor.detect()                   │
  ─────────────► │  ├── Data Drift   (feature distributions)│
                 │  └── Concept Drift(prediction vs actuals) │
                 └─────────────┬────────────────────────────┘
                               │ DriftReport
                               ▼
                 ┌──────────────────────────────────────────┐
                 │  _push_to_prometheus()                   │
                 │  claimguard_drift_* gauges               │
                 └──────────────────────────────────────────┘

Drift report structure
----------------------
DriftReport is a plain dataclass so it serialises to JSON without any Evidently
dependency on the caller side.

Fields:
  model_name        – 'underwriting' | 'fraud'
  timestamp         – UTC ISO-8601 string
  data_drift_score  – share of drifted features ∈ [0, 1]
  drifted_features  – list of column names that drifted
  concept_drift_detected – bool; True when prediction distribution shifted
  concept_drift_score    – PSI of predictions ∈ [0, ∞) (None when no labels)
  feature_drift_details  – {column: {"drift_score": float, "drifted": bool}}
  n_reference       – number of rows in the reference (baseline) DataFrame
  n_current         – number of rows in the current DataFrame
  evidently_available    – whether Evidently was used (vs. fallback stats)

Graceful degradation
--------------------
* Evidently absent  → lightweight scipy/numpy fallback (KS test for numeric,
  chi-squared for categorical, PSI for concept drift).
* prometheus_client absent → Prometheus push is a no-op.
* scipy absent      → fallback uses simple mean-shift heuristic.
* Baseline file missing → detect() returns an empty DriftReport with a warning.

Prometheus metrics
------------------
All metrics live under the ``claimguard_drift_*`` namespace.

  claimguard_drift_data_score{model}      Gauge – share of drifted features
  claimguard_drift_feature{model,feature} Gauge – per-feature drift score
  claimguard_drift_concept_score{model}   Gauge – PSI of predictions
  claimguard_drift_concept_detected{model} Gauge – 1.0 / 0.0 boolean

Public API
----------
    from src.drift_monitor import DriftMonitor, save_baseline

    # During training — save the training data as baseline
    save_baseline(train_df, model_name="fraud", prediction_col="fraud_label")

    # During serving — detect drift on a batch of incoming features
    monitor = DriftMonitor(model_name="fraud")
    report  = monitor.detect(current_df, predictions=scores_array)
    if report.data_drift_score > 0.3:
        logger.warning("Significant data drift detected!")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR      = Path(__file__).parent.parent / "data"
_BASELINE_DIR  = _DATA_DIR / "baselines"

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from evidently.report import Report                              # type: ignore
    from evidently.metric_preset import (                            # type: ignore
        DataDriftPreset,
        TargetDriftPreset,
    )
    from evidently.metrics import DatasetDriftMetric                 # type: ignore
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

try:
    from scipy import stats as _scipy_stats                          # type: ignore
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from prometheus_client import Gauge as _PGauge                  # type: ignore
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# ---------------------------------------------------------------------------
# Prometheus metrics (created once at module load)
# ---------------------------------------------------------------------------

_DRIFT_DATA_SCORE: Any    = None
_DRIFT_FEATURE: Any       = None
_DRIFT_CONCEPT_SCORE: Any = None
_DRIFT_CONCEPT_FLAG: Any  = None

if HAS_PROMETHEUS:
    try:
        _DRIFT_DATA_SCORE = _PGauge(
            "claimguard_drift_data_score",
            "Share of features with detected data drift (0–1)",
            ["model"],
        )
        _DRIFT_FEATURE = _PGauge(
            "claimguard_drift_feature_score",
            "Per-feature drift score (KS statistic or PSI)",
            ["model", "feature"],
        )
        _DRIFT_CONCEPT_SCORE = _PGauge(
            "claimguard_drift_concept_score",
            "Concept drift score — PSI of prediction distribution",
            ["model"],
        )
        _DRIFT_CONCEPT_FLAG = _PGauge(
            "claimguard_drift_concept_detected",
            "1.0 if concept drift detected, 0.0 otherwise",
            ["model"],
        )
    except Exception as _prom_exc:
        # Duplicate metric registration during tests — use no-ops
        logger.debug("drift_monitor: Prometheus metric registration: %s", _prom_exc)
        HAS_PROMETHEUS = False


# ---------------------------------------------------------------------------
# DriftReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Plain-Python result of a drift detection run."""

    model_name:               str
    timestamp:                str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data_drift_score:         float = 0.0      # share of features drifted ∈ [0,1]
    drifted_features:         List[str] = field(default_factory=list)
    concept_drift_detected:   bool = False
    concept_drift_score:      Optional[float] = None   # PSI; None when no labels
    feature_drift_details:    Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_reference:              int = 0
    n_current:                int = 0
    evidently_available:      bool = False
    warning:                  Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_name":             self.model_name,
            "timestamp":              self.timestamp,
            "data_drift_score":       round(self.data_drift_score, 4),
            "drifted_features":       self.drifted_features,
            "concept_drift_detected": self.concept_drift_detected,
            "concept_drift_score":    round(self.concept_drift_score, 4) if self.concept_drift_score else None,
            "feature_drift_details":  self.feature_drift_details,
            "n_reference":            self.n_reference,
            "n_current":              self.n_current,
            "evidently_available":    self.evidently_available,
            "warning":                self.warning,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------

def save_baseline(
    df: pd.DataFrame,
    model_name: str,
    prediction_col: Optional[str] = None,
) -> Path:
    """
    Persist *df* as the reference distribution for *model_name*.

    Saved as a Parquet file at data/baselines/<model_name>.parquet.

    Parameters
    ----------
    df             : training or held-out reference DataFrame
    model_name     : 'underwriting' | 'fraud' (or any string key)
    prediction_col : column to use as prediction proxy for concept drift.
                     Stored as 'target' in the baseline.

    Returns
    -------
    Path to the saved Parquet file.
    """
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    baseline = df.copy()
    if prediction_col and prediction_col in baseline.columns:
        baseline = baseline.rename(columns={prediction_col: "target"})
    path = _BASELINE_DIR / f"{model_name}.parquet"
    baseline.to_parquet(path, index=False)
    logger.info("drift_monitor: baseline saved → %s (%d rows)", path, len(baseline))
    return path


def load_baseline(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load the reference baseline DataFrame for *model_name*.

    Returns None when the baseline file does not exist.
    """
    path = _BASELINE_DIR / f"{model_name}.parquet"
    if not path.exists():
        logger.warning(
            "drift_monitor: no baseline found at %s — call save_baseline() first.",
            path,
        )
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.error("drift_monitor: failed to load baseline from %s — %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Drift thresholds
# ---------------------------------------------------------------------------

# A feature is considered drifted when its KS statistic or PSI exceeds this.
DATA_DRIFT_THRESHOLD    = float(os.getenv("CLAIMGUARD_DRIFT_DATA_THRESHOLD",    "0.10"))
# Concept drift fired when the prediction-PSI exceeds this.
CONCEPT_DRIFT_THRESHOLD = float(os.getenv("CLAIMGUARD_DRIFT_CONCEPT_THRESHOLD", "0.25"))


# ---------------------------------------------------------------------------
# Fallback drift computation (no Evidently)
# ---------------------------------------------------------------------------

def _ks_score(ref: pd.Series, cur: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic for two numeric samples."""
    if HAS_SCIPY:
        try:
            stat, _ = _scipy_stats.ks_2samp(ref.dropna(), cur.dropna())
            return float(stat)
        except Exception:
            pass
    # Ultra-lightweight fallback: normalised mean-shift
    mn = float(ref.std()) or 1.0
    return min(abs(float(ref.mean()) - float(cur.mean())) / mn, 1.0)


def _psi_score(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions."""
    try:
        ref_arr = ref.dropna().to_numpy(dtype=float)
        cur_arr = cur.dropna().to_numpy(dtype=float)
        if len(ref_arr) == 0 or len(cur_arr) == 0:
            return 0.0
        # Build bins on combined range
        combined = np.concatenate([ref_arr, cur_arr])
        bin_edges = np.percentile(combined, np.linspace(0, 100, bins + 1))
        bin_edges = np.unique(bin_edges)   # remove duplicates
        if len(bin_edges) < 2:
            return 0.0
        ref_counts = np.histogram(ref_arr, bins=bin_edges)[0] + 1e-6
        cur_counts = np.histogram(cur_arr, bins=bin_edges)[0] + 1e-6
        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(psi, 0.0)
    except Exception:
        return 0.0


def _chi2_score(ref: pd.Series, cur: pd.Series) -> float:
    """Normalised chi-squared distance for categorical columns."""
    try:
        all_cats = set(ref.dropna().unique()) | set(cur.dropna().unique())
        if not all_cats:
            return 0.0
        ref_counts = {c: 0 for c in all_cats}
        cur_counts = {c: 0 for c in all_cats}
        for v in ref.dropna():
            ref_counts[v] += 1
        for v in cur.dropna():
            cur_counts[v] += 1
        ref_total = max(sum(ref_counts.values()), 1)
        cur_total = max(sum(cur_counts.values()), 1)
        chi2 = 0.0
        for cat in all_cats:
            r = ref_counts[cat] / ref_total
            c = cur_counts[cat] / cur_total
            if r > 0:
                chi2 += (c - r) ** 2 / r
        return min(chi2, 1.0)
    except Exception:
        return 0.0


def _fallback_drift(
    reference: pd.DataFrame,
    current:   pd.DataFrame,
    prediction_col: Optional[str] = None,
) -> dict:
    """
    Compute per-column drift scores without Evidently.

    Returns a dict matching the structure expected by DriftMonitor.
    """
    feature_details: Dict[str, Dict[str, Any]] = {}
    numeric_cols   = reference.select_dtypes(include=[np.number]).columns.tolist()
    category_cols  = reference.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col in current.columns and col not in ("target",):
            score = _ks_score(reference[col], current[col])
            feature_details[col] = {
                "drift_score": round(score, 4),
                "drifted":     score > DATA_DRIFT_THRESHOLD,
                "method":      "ks",
            }
    for col in category_cols:
        if col in current.columns and col not in ("target",):
            score = _chi2_score(reference[col], current[col])
            feature_details[col] = {
                "drift_score": round(score, 4),
                "drifted":     score > DATA_DRIFT_THRESHOLD,
                "method":      "chi2",
            }

    drifted = [c for c, d in feature_details.items() if d["drifted"]]
    n_total = len(feature_details)
    data_score = len(drifted) / max(n_total, 1)

    # Concept drift: PSI on prediction column
    concept_score: Optional[float] = None
    concept_flag = False
    if prediction_col and prediction_col in reference.columns and prediction_col in current.columns:
        concept_score = _psi_score(reference[prediction_col], current[prediction_col])
        concept_flag  = concept_score > CONCEPT_DRIFT_THRESHOLD

    return {
        "feature_details": feature_details,
        "drifted_features": drifted,
        "data_drift_score": round(data_score, 4),
        "concept_score":    concept_score,
        "concept_flag":     concept_flag,
    }


# ---------------------------------------------------------------------------
# Evidently-based drift computation
# ---------------------------------------------------------------------------

def _evidently_drift(
    reference: pd.DataFrame,
    current:   pd.DataFrame,
    prediction_col: Optional[str] = None,
) -> dict:
    """
    Compute drift using Evidently AI report.

    Falls back to _fallback_drift on any Evidently error.
    """
    try:
        # ── Data drift report ────────────────────────────────────────────
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        result_dict = report.as_dict()

        feature_details: Dict[str, Dict[str, Any]] = {}
        drifted: List[str] = []

        # Navigate Evidently's nested result structure
        metrics = result_dict.get("metrics", [])
        for metric in metrics:
            result = metric.get("result", {})
            # DatasetDriftMetric result
            if "drift_by_columns" in result:
                for col, col_result in result["drift_by_columns"].items():
                    score = float(col_result.get("drift_score", 0.0))
                    col_drifted = bool(col_result.get("drift_detected", False))
                    feature_details[col] = {
                        "drift_score": round(score, 4),
                        "drifted":     col_drifted,
                        "method":      col_result.get("stattest_name", "evidently"),
                    }
                    if col_drifted:
                        drifted.append(col)

        # dataset-level share_of_drifted_columns
        dataset_drift_share = 0.0
        for metric in metrics:
            result = metric.get("result", {})
            if "share_of_drifted_columns" in result:
                dataset_drift_share = float(result["share_of_drifted_columns"])
                break
        if dataset_drift_share == 0.0 and feature_details:
            dataset_drift_share = len(drifted) / len(feature_details)

        # ── Concept drift (target / prediction drift) ─────────────────────
        concept_score: Optional[float] = None
        concept_flag = False
        if prediction_col:
            ref_pred = reference.rename(columns={prediction_col: "target"}) if prediction_col != "target" else reference
            cur_pred = current.rename(columns={prediction_col: "target"}) if prediction_col != "target" else current
            if "target" in ref_pred.columns and "target" in cur_pred.columns:
                target_report = Report(metrics=[TargetDriftPreset()])
                target_report.run(reference_data=ref_pred, current_data=cur_pred)
                t_dict = target_report.as_dict()
                for m in t_dict.get("metrics", []):
                    r = m.get("result", {})
                    if "drift_score" in r:
                        concept_score = float(r["drift_score"])
                        concept_flag  = bool(r.get("drift_detected", concept_score > CONCEPT_DRIFT_THRESHOLD))
                        break
                # Fallback: compute PSI manually if Evidently didn't yield a score
                if concept_score is None:
                    ref_tgt = ref_pred["target"] if "target" in ref_pred else pd.Series(dtype=float)
                    cur_tgt = cur_pred["target"] if "target" in cur_pred else pd.Series(dtype=float)
                    concept_score = _psi_score(ref_tgt, cur_tgt)
                    concept_flag  = concept_score > CONCEPT_DRIFT_THRESHOLD

        return {
            "feature_details": feature_details,
            "drifted_features": drifted,
            "data_drift_score": round(dataset_drift_share, 4),
            "concept_score":    concept_score,
            "concept_flag":     concept_flag,
        }

    except Exception as exc:
        logger.warning(
            "drift_monitor: Evidently failed (%s); using statistical fallback.", exc
        )
        return _fallback_drift(reference, current, prediction_col)


# ---------------------------------------------------------------------------
# Prometheus push
# ---------------------------------------------------------------------------

def _push_to_prometheus(model_name: str, report: DriftReport) -> None:
    """
    Update Prometheus gauges from a DriftReport.

    No-op when prometheus_client is absent or gauge creation failed.
    """
    if not HAS_PROMETHEUS:
        return
    try:
        if _DRIFT_DATA_SCORE is not None:
            _DRIFT_DATA_SCORE.labels(model=model_name).set(report.data_drift_score)
        if _DRIFT_CONCEPT_SCORE is not None and report.concept_drift_score is not None:
            _DRIFT_CONCEPT_SCORE.labels(model=model_name).set(report.concept_drift_score)
        if _DRIFT_CONCEPT_FLAG is not None:
            _DRIFT_CONCEPT_FLAG.labels(model=model_name).set(
                1.0 if report.concept_drift_detected else 0.0
            )
        if _DRIFT_FEATURE is not None:
            for feat, detail in report.feature_drift_details.items():
                _DRIFT_FEATURE.labels(
                    model=model_name, feature=feat
                ).set(float(detail.get("drift_score", 0.0)))
    except Exception as exc:
        logger.debug("drift_monitor: Prometheus push failed — %s", exc)


# ---------------------------------------------------------------------------
# DriftMonitor — main public class
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Detects data drift and concept drift for a named ClaimGuard model.

    Parameters
    ----------
    model_name : str
        'underwriting' | 'fraud' | any string matching a saved baseline.
    baseline_dir : Path, optional
        Override the default data/baselines/ directory.  Useful in tests.
    push_prometheus : bool
        Whether to update Prometheus gauges after every detect() call.
        Default True.

    Usage
    -----
        monitor = DriftMonitor("fraud")
        report  = monitor.detect(incoming_df, predictions=prob_array)
    """

    def __init__(
        self,
        model_name:       str,
        baseline_dir:     Optional[Path] = None,
        push_prometheus:  bool = True,
    ) -> None:
        self.model_name     = model_name
        self._baseline_dir  = baseline_dir or _BASELINE_DIR
        self._push_prom     = push_prometheus
        self._reference:    Optional[pd.DataFrame] = None
        self._load_baseline()

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def _load_baseline(self) -> None:
        path = self._baseline_dir / f"{self.model_name}.parquet"
        if path.exists():
            try:
                self._reference = pd.read_parquet(path)
                logger.info(
                    "drift_monitor[%s]: baseline loaded (%d rows).",
                    self.model_name, len(self._reference),
                )
            except Exception as exc:
                logger.error("drift_monitor[%s]: baseline load failed — %s", self.model_name, exc)
        else:
            logger.debug(
                "drift_monitor[%s]: no baseline at %s.", self.model_name, path
            )

    def reload_baseline(self) -> None:
        """Re-read the baseline from disk.  Call after save_baseline()."""
        self._load_baseline()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        current:        pd.DataFrame,
        predictions:    Optional[np.ndarray] = None,
        prediction_col: Optional[str]        = None,
    ) -> DriftReport:
        """
        Compute data drift and concept drift for *current* vs the baseline.

        Parameters
        ----------
        current        : DataFrame of incoming features (same schema as baseline).
        predictions    : 1-D array of model output probabilities / predictions.
                         When provided it is appended to *current* as '__pred__'
                         and used for concept-drift detection.
        prediction_col : column name in *current* that holds predictions
                         (alternative to the *predictions* array).

        Returns
        -------
        DriftReport  — always returned, never raises.
        """
        report = DriftReport(model_name=self.model_name)

        # ── Guard: baseline required ───────────────────────────────────
        if self._reference is None:
            report.warning = (
                f"No baseline for model '{self.model_name}'. "
                "Call save_baseline() during training first."
            )
            logger.warning("drift_monitor[%s]: %s", self.model_name, report.warning)
            return report

        # ── Align schemas ──────────────────────────────────────────────
        reference = self._reference.copy()
        current   = current.copy()

        # Attach predictions array as a column for concept drift
        _pred_col = prediction_col
        if predictions is not None:
            current["__pred__"] = np.asarray(predictions).ravel()
            reference["__pred__"] = np.nan   # fill with NaN; PSI uses current only
            _pred_col = "__pred__"

        report.n_reference = len(reference)
        report.n_current   = len(current)

        # ── Compute drift ──────────────────────────────────────────────
        try:
            if HAS_EVIDENTLY:
                result = _evidently_drift(reference, current, _pred_col)
                report.evidently_available = True
            else:
                result = _fallback_drift(reference, current, _pred_col)
                report.evidently_available = False

            report.feature_drift_details  = result["feature_details"]
            report.drifted_features        = result["drifted_features"]
            report.data_drift_score        = result["data_drift_score"]
            report.concept_drift_score     = result["concept_score"]
            report.concept_drift_detected  = result["concept_flag"]

        except Exception as exc:
            report.warning = f"Drift computation failed: {exc}"
            logger.error("drift_monitor[%s]: %s", self.model_name, exc, exc_info=True)

        # ── Push to Prometheus ─────────────────────────────────────────
        if self._push_prom:
            _push_to_prometheus(self.model_name, report)

        logger.info(
            "drift_monitor[%s]: data_drift=%.3f drifted_cols=%d concept_psi=%s",
            self.model_name,
            report.data_drift_score,
            len(report.drifted_features),
            f"{report.concept_drift_score:.3f}" if report.concept_drift_score is not None else "—",
        )
        return report

    # ------------------------------------------------------------------
    # Convenience: batch-score a list of feature dicts
    # ------------------------------------------------------------------

    def detect_from_dicts(
        self,
        feature_dicts:  List[Dict[str, Any]],
        predictions:    Optional[np.ndarray] = None,
    ) -> DriftReport:
        """Thin wrapper: build a DataFrame from *feature_dicts* then call detect()."""
        current = pd.DataFrame(feature_dicts)
        return self.detect(current, predictions=predictions)
