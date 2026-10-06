"""
src/claims_fraud.py
===================
Claims fraud-detection engine for ClaimGuard AI.

Scores claim features at claim-filing time and returns a fraud probability,
binary flag, confidence tier, and SHAP-based explanation.

Fallback chain (model training)
--------------------------------
  1. XGBoost + LightGBM ensemble   (preferred)
  2. sklearn RandomForestClassifier (if xgboost/lightgbm absent)
  3. Rule-based heuristic           (claim_amount + num_prior_claims)

Observability additions
-----------------------
* MLflow — logs hyperparameters, cross-val ROC-AUC + F1, pickled model
  artifact, and a feature-importance bar chart on every _train() call.
  Gracefully skipped when mlflow is not installed or the tracking server
  is unreachable (HAS_MLFLOW guard + try/except throughout).

* Evidently / Drift Monitor — detect_drift() runs data-drift and
  concept-drift detection against the training-time baseline and exports
  results to Prometheus gauges (claimguard_drift_*).
  Gracefully skipped when evidently or the baseline file is absent.
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.shap_utils import explain_prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy-dep imports
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb   # type: ignore
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb  # type: ignore
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna            # type: ignore
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import mlflow            # type: ignore
    import mlflow.sklearn    # type: ignore
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
_CLAIMS_CSV = Path(__file__).parent.parent / "data" / "sample_claims.csv"
_MODEL_PATH = _MODELS_DIR / "fraud_model.pkl"

# Categorical columns that get label-encoded.
_CATEGORICAL_COLS = ["claim_type", "claim_severity"]

# Numeric feature columns.
_NUMERIC_COLS = [
    "claim_amount",
    "days_since_policy_start",
    "num_prior_claims",
]

_ALL_FEATURE_COLS = _NUMERIC_COLS + _CATEGORICAL_COLS

MODEL_VERSION  = "1.0.0"
MLFLOW_EXP     = "claimguard_fraud_detection"   # MLflow experiment name

# Fraud score thresholds.
_HIGH_THRESHOLD   = 0.7
_MEDIUM_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Pydantic v2 schemas
# ---------------------------------------------------------------------------

class ClaimFeatures(BaseModel):
    """Input features collected at claim-filing time."""
    model_config = ConfigDict(str_strip_whitespace=True)

    claim_id:                str
    claim_amount:            float
    days_since_policy_start: int
    num_prior_claims:        int
    claim_type:              str
    claim_severity:          str
    repair_shop_id:          Optional[str] = None
    medical_provider_id:     Optional[str] = None
    claimant_id:             str
    policy_id:               Optional[str] = None


class FraudScoringResult(BaseModel):
    """Scored output returned to the caller."""
    model_config = ConfigDict(str_strip_whitespace=True, protected_namespaces=())

    claim_id:         str
    fraud_score:      float             # 0.0 – 1.0
    fraud_flag:       bool              # True when fraud_score >= 0.5
    confidence_tier:  str               # 'low' | 'medium' | 'high'
    shap_drivers:     List[dict] = Field(default_factory=list)
    model_version:    str = MODEL_VERSION
    timestamp:        datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Helper: encode a ClaimFeatures row to a 1-row DataFrame
# ---------------------------------------------------------------------------

def _encode_row(
    features: ClaimFeatures,
    encoders: Dict[str, LabelEncoder],
) -> pd.DataFrame:
    row: Dict[str, Any] = {
        "claim_amount":            features.claim_amount,
        "days_since_policy_start": features.days_since_policy_start,
        "num_prior_claims":        features.num_prior_claims,
    }
    for col in _CATEGORICAL_COLS:
        val = getattr(features, col)
        enc: LabelEncoder = encoders[col]
        row[col] = int(enc.transform([val])[0]) if val in enc.classes_ else 0
    return pd.DataFrame([row], columns=_ALL_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Rule-based fallback scorer
# ---------------------------------------------------------------------------

class _RuleBasedFraudScorer:
    """Minimal heuristic when no ML library is available."""

    def predict_proba(self, X: pd.DataFrame):   # noqa: N802
        scores = []
        for _, row in X.iterrows():
            score  = 0.3
            amount = float(row.get("claim_amount", 50_000))
            claims = float(row.get("num_prior_claims", 0))
            days   = float(row.get("days_since_policy_start", 365))

            if amount > 400_000:
                score += 0.25
            elif amount > 200_000:
                score += 0.10

            score += min(claims * 0.07, 0.25)

            if days < 30:
                score += 0.20

            scores.append([1 - min(max(score, 0.0), 1.0), min(max(score, 0.0), 1.0)])
        return np.array(scores)


# ---------------------------------------------------------------------------
# MLflow logging helper
# ---------------------------------------------------------------------------

def _log_to_mlflow(
    run_params: dict,
    metrics:    dict,
    model:      Any,
    X_train:    pd.DataFrame,
    model_name: str = "fraud_model",
) -> None:
    """
    Log training metadata to MLflow.

    Called from FraudDetectionEngine._train().
    Silently skipped when MLflow is not installed or the server is unreachable.

    Logs
    ----
    * Parameters : all entries in *run_params*
    * Metrics    : roc_auc_cv, f1_cv, n_features, n_samples
    * Artifacts  : pickled model binary, feature-importance PNG
    * Tags       : model_type, framework, fraud_threshold
    """
    if not HAS_MLFLOW:
        return
    try:
        mlflow.set_experiment(MLFLOW_EXP)
        with mlflow.start_run(
            run_name=f"{model_name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        ):
            # Parameters
            mlflow.log_params({k: str(v) for k, v in run_params.items()})

            # Metrics
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})

            # Tags
            mlflow.set_tags({
                "model_type":       "ensemble" if hasattr(model, "xgb_model") else type(model).__name__,
                "framework":        "xgboost+lightgbm" if hasattr(model, "xgb_model") else "sklearn",
                "model_version":    MODEL_VERSION,
                "fraud_threshold":  str(0.5),
            })

            # Model artifact (pickled)
            with tempfile.TemporaryDirectory() as tmp:
                model_path = Path(tmp) / f"{model_name}.pkl"
                with open(model_path, "wb") as fh:
                    pickle.dump(model, fh)
                mlflow.log_artifact(str(model_path), artifact_path="model")

            # Feature importance plot (best-effort)
            _log_feature_importance(model, X_train)

    except Exception as exc:
        logger.warning("claims_fraud: MLflow logging failed — %s", exc)


def _log_feature_importance(model: Any, X: pd.DataFrame) -> None:
    """Log a feature-importance bar chart to the active MLflow run."""
    try:
        import matplotlib                       # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt         # type: ignore

        primary = getattr(model, "primary_model", model)
        importances: Optional[np.ndarray] = None
        if hasattr(primary, "feature_importances_"):
            importances = primary.feature_importances_
        elif hasattr(primary, "coef_"):
            importances = np.abs(primary.coef_[0])

        if importances is None or len(importances) != len(X.columns):
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        sorted_idx = np.argsort(importances)
        ax.barh(
            np.array(X.columns)[sorted_idx],
            importances[sorted_idx],
            color="#FF6B6B",
        )
        ax.set_title("Feature Importance — Fraud Detection Model")
        ax.set_xlabel("Importance")
        fig.tight_layout()

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "feature_importance.png"
            fig.savefig(img_path, dpi=100)
            mlflow.log_artifact(str(img_path), artifact_path="plots")
        plt.close(fig)
    except Exception as exc:
        logger.debug("claims_fraud: feature importance plot failed — %s", exc)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class FraudDetectionEngine:
    """
    Orchestrates training and inference for the claims fraud-detection model.

    Training data: data/sample_claims.csv  (fraud_label as target).
    Model persisted to data/models/fraud_model.pkl.

    MLflow
    ------
    Training logs: experiment='claimguard_fraud_detection', run per _train().
    Set MLFLOW_TRACKING_URI env var to point at a remote server.

    Drift monitoring
    ----------------
    detect_drift(df) uses DriftMonitor against the training-time baseline.
    """

    def __init__(self, force_retrain: bool = False) -> None:
        self._encoders:      Dict[str, LabelEncoder] = {}
        self._model:         Any = None
        self._feature_names: List[str] = list(_ALL_FEATURE_COLS)
        self._train_df:      Optional[pd.DataFrame] = None

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not force_retrain and _MODEL_PATH.exists():
            self._load()
        else:
            self._train()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        payload = {
            "model":         self._model,
            "encoders":      self._encoders,
            "feature_names": self._feature_names,
        }
        with open(_MODEL_PATH, "wb") as fh:
            pickle.dump(payload, fh)

    def _load(self) -> None:
        try:
            with open(_MODEL_PATH, "rb") as fh:
                payload = pickle.load(fh)
            self._model         = payload["model"]
            self._encoders      = payload["encoders"]
            self._feature_names = payload.get("feature_names", list(_ALL_FEATURE_COLS))
        except Exception:
            self._train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self) -> None:
        """
        Train on data/sample_claims.csv with fraud_label as target.

        Saves the baseline for drift monitoring and logs run to MLflow.
        """
        try:
            df = pd.read_csv(_CLAIMS_CSV)
        except Exception:
            self._model    = _RuleBasedFraudScorer()
            self._encoders = {c: LabelEncoder().fit(["unknown"]) for c in _CATEGORICAL_COLS}
            return

        # ── Encode categoricals ──────────────────────────────────────────
        self._encoders = {}
        for col in _CATEGORICAL_COLS:
            enc = LabelEncoder()
            if col in df.columns:
                df[col] = enc.fit_transform(df[col].astype(str))
            else:
                df[col] = 0
                enc.fit(["unknown"])
            self._encoders[col] = enc

        for col in _NUMERIC_COLS:
            if col not in df.columns:
                df[col] = 0

        X = df[_ALL_FEATURE_COLS].copy()

        if "fraud_label" in df.columns:
            y = df["fraud_label"].astype(int)
        else:
            y = (df["claim_amount"] > df["claim_amount"].median()).astype(int)

        self._feature_names = list(X.columns)
        self._train_df      = X.copy()
        self._train_df["target"] = y.values

        # ── Build model ──────────────────────────────────────────────────
        best_params = self._get_best_xgb_params(X, y)
        self._model = self._build_model(X, y)
        self._save()

        # ── Evaluate ─────────────────────────────────────────────────────
        metrics = self._evaluate(X, y)

        # ── Save baseline for drift monitor ──────────────────────────────
        try:
            from src.drift_monitor import save_baseline
            save_baseline(self._train_df, model_name="fraud", prediction_col="target")
        except Exception as exc:
            logger.warning("claims_fraud: baseline save failed — %s", exc)

        # ── Log to MLflow ────────────────────────────────────────────────
        run_params: dict = {
            "n_features":    len(self._feature_names),
            "n_samples":     len(X),
            "optuna_trials": 10 if HAS_OPTUNA else 0,
            "fallback_chain": (
                "xgb+lgbm" if HAS_XGBOOST and HAS_LIGHTGBM
                else "xgb" if HAS_XGBOOST
                else "lgbm" if HAS_LIGHTGBM
                else "sklearn_rf"
            ),
            "fraud_pos_rate": round(float(y.mean()), 4),
        }
        run_params.update(best_params)
        _log_to_mlflow(run_params, metrics, self._model, X)

        logger.info(
            "claims_fraud: trained — ROC-AUC=%.3f  F1=%.3f  fraud_rate=%.1f%%",
            metrics.get("roc_auc_cv") or 0,
            metrics.get("f1_cv") or 0,
            float(y.mean()) * 100,
        )

    def _evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Compute cross-val ROC-AUC and F1 for logging; never raises."""
        try:
            if isinstance(self._model, _RuleBasedFraudScorer):
                return {"roc_auc_cv": None, "f1_cv": None, "n_samples": len(y)}
            _m = (
                self._model.xgb_model
                if hasattr(self._model, "xgb_model")
                else self._model
            )
            roc_scores = cross_val_score(_m, X, y, cv=min(3, len(y)), scoring="roc_auc")
            f1_scores  = cross_val_score(_m, X, y, cv=min(3, len(y)), scoring="f1")
            return {
                "roc_auc_cv": round(float(roc_scores.mean()), 4),
                "f1_cv":      round(float(f1_scores.mean()), 4),
                "n_samples":  len(y),
                "n_features": len(X.columns),
            }
        except Exception as exc:
            logger.debug("claims_fraud: evaluation failed — %s", exc)
            return {"roc_auc_cv": None, "f1_cv": None}

    def _build_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        if HAS_XGBOOST and HAS_LIGHTGBM:
            return self._train_ensemble(X, y)
        if HAS_XGBOOST or HAS_LIGHTGBM:
            return self._train_single_tree(X, y)
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            return rf
        except Exception:
            return _RuleBasedFraudScorer()

    def _get_best_xgb_params(self, X: pd.DataFrame, y: pd.Series) -> dict:
        if not HAS_OPTUNA or not HAS_XGBOOST:
            return {
                "n_estimators":  100,
                "max_depth":     4,
                "learning_rate": 0.1,
                "random_state":  42,
            }

        def objective(trial: "optuna.Trial") -> float:   # type: ignore[name-defined]
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 50, 200),
                "max_depth":     trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "random_state":  42,
            }
            model = xgb.XGBClassifier(**params)
            score = cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()
            return -score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        best = study.best_params
        best["random_state"] = 42
        return best

    def _train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> "_FraudEnsembleWrapper":
        params    = self._get_best_xgb_params(X, y)
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X, y)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1
        )
        lgb_model.fit(X, y)
        return _FraudEnsembleWrapper(xgb_model, lgb_model)

    def _train_single_tree(self, X: pd.DataFrame, y: pd.Series) -> Any:
        if HAS_XGBOOST:
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1
            )
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: ClaimFeatures) -> FraudScoringResult:
        """Score a claim and return a structured FraudScoringResult."""
        row_df = _encode_row(features, self._encoders)

        raw_proba   = self._model.predict_proba(row_df)
        fraud_score = float(raw_proba[0][1])
        fraud_flag  = fraud_score >= 0.5

        confidence_tier = _score_to_confidence(fraud_score)

        shap_model = (
            self._model.primary_model
            if isinstance(self._model, _FraudEnsembleWrapper)
            else self._model
        )
        shap_drivers = explain_prediction(shap_model, row_df, self._feature_names)

        return FraudScoringResult(
            claim_id        = features.claim_id,
            fraud_score     = round(fraud_score, 6),
            fraud_flag      = fraud_flag,
            confidence_tier = confidence_tier,
            shap_drivers    = shap_drivers,
            model_version   = MODEL_VERSION,
        )

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        current_df:  pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
    ) -> "DriftReport":   # type: ignore[name-defined]
        """
        Detect data drift and concept drift for *current_df* vs the
        training-time baseline.

        Parameters
        ----------
        current_df  : DataFrame with the same claim-feature columns as
                      the training data (raw, before label-encoding).
        predictions : 1-D array of fraud probabilities for rows in
                      *current_df*.  Used for concept-drift detection.

        Returns
        -------
        DriftReport — see src.drift_monitor.DriftReport for fields.
        Always returned; never raises.
        """
        try:
            from src.drift_monitor import DriftMonitor, DriftReport
            monitor = DriftMonitor(model_name="fraud")
            return monitor.detect(current_df, predictions=predictions)
        except Exception as exc:
            logger.error("claims_fraud.detect_drift failed — %s", exc)
            try:
                from src.drift_monitor import DriftReport
                r = DriftReport(model_name="fraud")
                r.warning = str(exc)
                return r
            except Exception:
                return {"model_name": "fraud", "warning": str(exc)}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

class _FraudEnsembleWrapper:
    """Averages probabilities from XGBoost and LightGBM."""

    def __init__(self, xgb_model: Any, lgb_model: Any) -> None:
        self.xgb_model    = xgb_model
        self.lgb_model    = lgb_model
        self.primary_model = xgb_model

    def predict_proba(self, X: pd.DataFrame):   # noqa: N802
        p_xgb = self.xgb_model.predict_proba(X)
        p_lgb = self.lgb_model.predict_proba(X)
        return (p_xgb + p_lgb) / 2.0

    def get_params(self, deep: bool = True) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _score_to_confidence(score: float) -> str:
    if score >= _HIGH_THRESHOLD:
        return "high"
    if score >= _MEDIUM_THRESHOLD:
        return "medium"
    return "low"
