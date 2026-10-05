"""
Underwriting risk-scoring engine for ClaimGuard AI.

Scores applicant/policy features at policy-issuance time and returns a
risk tier, premium adjustment recommendation, and SHAP-based explanation.

Fallback chain:
  1. XGBoost + LightGBM ensemble (preferred)
  2. sklearn RandomForestClassifier (if xgboost/lightgbm absent)
  3. Rule-based heuristic (credit_score + prior_claims_count)
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.shap_utils import explain_prediction

# ---------------------------------------------------------------------------
# Optional heavy-dep imports
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb  # type: ignore

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb  # type: ignore

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
_POLICIES_CSV = Path(__file__).parent.parent / "data" / "sample_policies.csv"
_MODEL_PATH = _MODELS_DIR / "underwriting_model.pkl"

# Categorical columns that get label-encoded.
_CATEGORICAL_COLS = ["coverage_type", "region", "occupation"]

# Numeric feature columns (after encoding).
_NUMERIC_COLS = [
    "age",
    "annual_income",
    "credit_score",
    "sum_insured",
    "num_dependents",
    "prior_claims_count",
]

_ALL_FEATURE_COLS = _NUMERIC_COLS + _CATEGORICAL_COLS

MODEL_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Pydantic v2 schemas
# ---------------------------------------------------------------------------


class UnderwritingFeatures(BaseModel):
    """Input features collected at policy-issuance time."""

    model_config = ConfigDict(str_strip_whitespace=True)

    age: int
    annual_income: float
    credit_score: int
    sum_insured: float
    coverage_type: str
    num_dependents: int
    prior_claims_count: int
    region: str
    occupation: str


class UnderwritingResult(BaseModel):
    """Scored output returned to the caller."""

    model_config = ConfigDict(str_strip_whitespace=True, protected_namespaces=())

    risk_tier: str  # 'low' | 'medium' | 'high'
    risk_score: float  # 0.0 – 1.0
    premium_adjustment: float  # multiplier, e.g. 1.15
    shap_drivers: List[dict] = Field(default_factory=list)
    model_version: str = MODEL_VERSION
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Helper: label-encode a features dict to a 1-row DataFrame
# ---------------------------------------------------------------------------


def _encode_row(features: UnderwritingFeatures, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    row: Dict[str, Any] = {
        "age": features.age,
        "annual_income": features.annual_income,
        "credit_score": features.credit_score,
        "sum_insured": features.sum_insured,
        "num_dependents": features.num_dependents,
        "prior_claims_count": features.prior_claims_count,
    }
    for col in _CATEGORICAL_COLS:
        val = getattr(features, col)
        enc: LabelEncoder = encoders[col]
        # Handle unseen categories gracefully.
        if val in enc.classes_:
            row[col] = int(enc.transform([val])[0])
        else:
            row[col] = 0
    return pd.DataFrame([row], columns=_ALL_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Rule-based fallback scorer
# ---------------------------------------------------------------------------


class _RuleBasedScorer:
    """Minimal heuristic when no ML library is available."""

    def predict_proba(self, X: pd.DataFrame):  # noqa: N802
        scores = []
        for _, row in X.iterrows():
            score = 0.5
            credit = float(row.get("credit_score", 600))
            claims = float(row.get("prior_claims_count", 0))
            # Low credit → higher risk.
            if credit < 450:
                score += 0.25
            elif credit < 600:
                score += 0.10
            else:
                score -= 0.10
            # Prior claims → higher risk.
            score += min(claims * 0.06, 0.30)
            scores.append([1 - min(max(score, 0.0), 1.0), min(max(score, 0.0), 1.0)])
        return np.array(scores)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class UnderwritingEngine:
    """
    Orchestrates training and inference for the underwriting risk model.

    On first instantiation the model is trained from the policies CSV and
    persisted to data/models/underwriting_model.pkl. Subsequent instantiations
    load from disk unless *force_retrain=True* is passed.
    """

    def __init__(self, force_retrain: bool = False) -> None:
        self._encoders: Dict[str, LabelEncoder] = {}
        self._model: Any = None
        self._feature_names: List[str] = list(_ALL_FEATURE_COLS)

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
            "model": self._model,
            "encoders": self._encoders,
            "feature_names": self._feature_names,
        }
        with open(_MODEL_PATH, "wb") as fh:
            pickle.dump(payload, fh)

    def _load(self) -> None:
        try:
            with open(_MODEL_PATH, "rb") as fh:
                payload = pickle.load(fh)
            self._model = payload["model"]
            self._encoders = payload["encoders"]
            self._feature_names = payload.get("feature_names", list(_ALL_FEATURE_COLS))
        except Exception:
            # Corrupt/stale pkl — retrain from scratch.
            self._train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self) -> None:
        """
        Train on data/sample_policies.csv.

        Tries XGBoost+LightGBM ensemble → sklearn RF → rule-based heuristic.
        """
        try:
            df = pd.read_csv(_POLICIES_CSV)
        except Exception:
            # CSV missing — initialise rule-based fallback directly.
            self._model = _RuleBasedScorer()
            self._encoders = {col: LabelEncoder().fit(["unknown"]) for col in _CATEGORICAL_COLS}
            return

        # --- encode categoricals ---
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
        # Target: binarise risk_tier (high → 1, else → 0)
        if "risk_tier" in df.columns:
            y = (df["risk_tier"] == "high").astype(int)
        elif "fraud_label" in df.columns:
            y = df["fraud_label"].astype(int)
        else:
            y = (df["prior_claims_count"] >= 3).astype(int)

        self._feature_names = list(X.columns)
        self._model = self._build_model(X, y)
        self._save()

    def _build_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Try ensemble → RF → rule-based, return the best available."""
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
            return _RuleBasedScorer()

    def _get_best_xgb_params(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Return tuned XGB params if Optuna available, else defaults."""
        if not HAS_OPTUNA:
            return {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "random_state": 42}

        def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }
            model = xgb.XGBClassifier(**params)
            score = cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()
            return -score  # minimise negative AUC

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10, show_progress_bar=False)
        best = study.best_params
        best["random_state"] = 42
        return best

    def _train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> "_EnsembleWrapper":
        xgb_params = self._get_best_xgb_params(X, y)
        xgb_params.pop("use_label_encoder", None)
        xgb_params.pop("eval_metric", None)

        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X, y)

        lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
        lgb_model.fit(X, y)

        return _EnsembleWrapper(xgb_model, lgb_model)

    def _train_single_tree(self, X: pd.DataFrame, y: pd.Series) -> Any:
        if HAS_XGBOOST:
            model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        else:
            model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
        model.fit(X, y)
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: UnderwritingFeatures) -> UnderwritingResult:
        """Score an applicant and return a structured UnderwritingResult."""
        row_df = _encode_row(features, self._encoders)

        raw_proba = self._model.predict_proba(row_df)
        risk_score = float(raw_proba[0][1])

        risk_tier = _score_to_tier(risk_score)
        premium_adjustment = round(0.8 + risk_score * 0.7, 4)

        # Explainability (best-effort, never raises).
        shap_model = (
            self._model.primary_model
            if isinstance(self._model, _EnsembleWrapper)
            else self._model
        )
        shap_drivers = explain_prediction(shap_model, row_df, self._feature_names)

        return UnderwritingResult(
            risk_tier=risk_tier,
            risk_score=round(risk_score, 6),
            premium_adjustment=premium_adjustment,
            shap_drivers=shap_drivers,
            model_version=MODEL_VERSION,
        )


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------


class _EnsembleWrapper:
    """Averages probabilities from XGBoost and LightGBM."""

    def __init__(self, xgb_model, lgb_model) -> None:
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        # Expose primary model for SHAP (XGBoost has richer TreeExplainer support).
        self.primary_model = xgb_model

    def predict_proba(self, X: pd.DataFrame):  # noqa: N802
        p_xgb = self.xgb_model.predict_proba(X)
        p_lgb = self.lgb_model.predict_proba(X)
        return (p_xgb + p_lgb) / 2.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _score_to_tier(score: float) -> str:
    if score <= 0.33:
        return "low"
    if score <= 0.66:
        return "medium"
    return "high"
