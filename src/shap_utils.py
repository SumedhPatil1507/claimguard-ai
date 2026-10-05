"""
SHAP explainability utilities for ClaimGuard AI.

Provides model-agnostic SHAP explanations for tree-based ensembles used
in both underwriting risk scoring and claims fraud detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

try:
    import shap  # type: ignore

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

if TYPE_CHECKING:
    import pandas as pd


def explain_prediction(
    model,
    features_df: "pd.DataFrame",
    feature_names: List[str],
    top_n: int = 5,
) -> List[dict]:
    """
    Compute SHAP values for a single-row prediction and return the top-N
    most influential features.

    Parameters
    ----------
    model : fitted sklearn/xgboost/lightgbm estimator
    features_df : pd.DataFrame with exactly one row
    feature_names : list of feature column names
    top_n : number of top drivers to return (default 5)

    Returns
    -------
    list of dicts with keys:
        feature     – column name
        shap_value  – raw SHAP value (float)
        direction   – 'increases_risk' if shap_value > 0 else 'decreases_risk'

    Returns an empty list when SHAP is not installed or explanation fails.
    """
    if not HAS_SHAP:
        return []

    try:
        # Prefer TreeExplainer for tree-based models (faster + exact).
        # Fall back to generic Explainer for other model types.
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df)
        except Exception:
            explainer = shap.Explainer(model, features_df)
            shap_values = explainer(features_df).values

        # Handle multi-output / binary classifier output shapes.
        # For binary classifiers, TreeExplainer may return a list [neg, pos].
        if isinstance(shap_values, list):
            # Take the positive-class (index 1) values for binary classification.
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            values = np.array(shap_values).flatten()
            if len(values) != len(feature_names):
                # 2-D array: take the first (only) row.
                values = np.array(shap_values)[0]

        values = np.array(values, dtype=float).flatten()

        # Pair with names and sort by absolute magnitude.
        paired = sorted(
            zip(feature_names, values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        drivers = []
        for feature, sv in paired[:top_n]:
            drivers.append(
                {
                    "feature": str(feature),
                    "shap_value": float(sv),
                    "direction": "increases_risk" if sv > 0 else "decreases_risk",
                }
            )
        return drivers

    except Exception:
        # Never let explainability failures crash the scoring pipeline.
        return []
