"""
SHAP-based user-level churn explanations for User Engagement Intelligence (Day 4).

SHAP values (plain language): starting from the model's average prediction,
each feature's SHAP value is how much that feature pushed THIS user's
prediction up (positive) or down (negative).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def rebuild_feature_matrix(
    df: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Rebuild the Day-3 feature matrix: drop ids/target/cluster id, one-hot
    encode cluster_name + device_type, then reindex to the saved column order.
    """
    X = df.drop(columns=[c for c in ["user_id", "churn", "cluster"] if c in df.columns])
    X = pd.get_dummies(X, columns=["cluster_name", "device_type"], drop_first=False)
    X = X.astype(float)
    # reindex guarantees the exact column order from feature_names.joblib
    # (missing one-hot cols become 0; extra cols are dropped)
    X = X.reindex(columns=list(feature_names), fill_value=0.0)
    return X


def _factor_phrase(name: str, feature_val: float, ref_median: float) -> str:
    """
    Turn a feature into a short phrase like 'low sessions_last_30_days'
    or 'device_type_Web' (for active one-hot categories).
    """
    # One-hot / binary columns: phrase depends on whether the flag is on
    if feature_val in (0.0, 1.0) and (
        name.startswith("cluster_name_") or name.startswith("device_type_")
    ):
        if feature_val >= 0.5:
            return name  # e.g. cluster_name_At-Risk/Dormant
        # Absence of a category — still mention it when SHAP |value| is large
        return f"not {name}"

    level = "high" if feature_val >= ref_median else "low"
    return f"{level} {name}"


def _top_factors(
    shap_row: np.ndarray,
    feature_vals: np.ndarray,
    feature_names: Sequence[str],
    medians: np.ndarray,
    k: int = 3,
):
    """Split features into top risk-increasing vs risk-reducing by SHAP value."""
    pairs = []
    for name, shap_v, feat_v, med in zip(
        feature_names, shap_row, feature_vals, medians
    ):
        # Skip cluster one-hots in the sentence — they are derived from the same
        # behaviors and often look contradictory once sessions/recency are included.
        # (They still appear in SHAP beeswarm / waterfall plots.)
        if name.startswith("cluster_name_"):
            continue
        # Skip inactive device one-hots ("not device_type_X")
        if name.startswith("device_type_") and feat_v < 0.5:
            continue
        pairs.append((name, shap_v, feat_v, med))

    increasing = sorted(
        [p for p in pairs if p[1] > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:k]
    decreasing = sorted(
        [p for p in pairs if p[1] < 0],
        key=lambda x: x[1],
    )[:k]

    inc_phrases = [_factor_phrase(n, v, m) for n, _, v, m in increasing]
    dec_phrases = [_factor_phrase(n, v, m) for n, _, v, m in decreasing]
    return inc_phrases, dec_phrases


def explain_user(
    user_id: str,
    df: pd.DataFrame,
    model,
    explainer,
    feature_names: Sequence[str],
    top_k: int = 3,
    feature_medians: np.ndarray | None = None,
) -> str:
    """
    Print (and return) a plain-English SHAP explanation for one user.

    Parameters
    ----------
    user_id : str
        e.g. \"U00042\"
    df : pd.DataFrame
        Full users_with_clusters-style table (must include user_id + feature cols).
    model : fitted classifier
        Champion churn model (LogisticRegression).
    explainer : shap.Explainer / LinearExplainer
        Pre-fit SHAP explainer.
    feature_names : sequence of str
        Exact Day-3 feature column order.
    top_k : int
        How many push-up / push-down factors to mention.
    feature_medians : optional array
        Per-feature medians for high/low wording. Computed from df if omitted.
    """
    matches = df[df["user_id"] == user_id]
    if matches.empty:
        msg = f"User {user_id} not found."
        print(msg)
        return msg

    row_df = matches.iloc[[0]]
    X_row = rebuild_feature_matrix(row_df, feature_names)

    # Predicted churn probability for the positive class
    proba = float(model.predict_proba(X_row)[0, 1])

    # SHAP values for this single row
    # Newer SHAP API returns an Explanation object; .values is (1, n_features)
    explanation = explainer(X_row)
    shap_vals = np.asarray(explanation.values)
    if shap_vals.ndim == 3:
        # some versions return (n, features, classes) — take positive class
        shap_vals = shap_vals[:, :, 1]
    shap_row = shap_vals[0]
    feature_vals = X_row.to_numpy()[0]

    if feature_medians is None:
        X_all = rebuild_feature_matrix(df, feature_names)
        feature_medians = X_all.median().to_numpy()

    increasing, decreasing = _top_factors(
        shap_row, feature_vals, feature_names, feature_medians, k=top_k
    )

    inc_txt = ", ".join(increasing) if increasing else "none notable"
    dec_txt = ", ".join(decreasing) if decreasing else "none notable"

    text = (
        f"User {user_id} has a {proba:.0%} predicted churn probability. "
        f"Main factors increasing risk: {inc_txt}. "
        f"Main factors reducing risk: {dec_txt}."
    )
    print(text)
    return text
