"""
Reusable K-Means clustering for User Engagement Intelligence (Day 2).

Fits/loads a StandardScaler + KMeans (k=4) so Day 3+ can add cluster
features without re-running the notebook.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Behavioral features only — never include churn (label leakage) or user_id
BEHAVIORAL_FEATURES = [
    "sessions_last_30_days",
    "avg_session_duration_minutes",
    "content_views_last_30_days",
    "likes_last_30_days",
    "shares_last_30_days",
    "days_since_last_login",
    "notifications_clicked_last_30_days",
]

# Chosen in notebooks/02_clustering.ipynb via elbow + silhouette
N_CLUSTERS = 4
RANDOM_STATE = 42

# Human-readable names mapped to KMeans labels (random_state=42 on users.csv).
# Re-run fit_and_save() if you regenerate the data and labels shift.
CLUSTER_NAMES = {
    0: "Power Users",
    1: "Casual Browsers",
    2: "Steady Engagers",
    3: "At-Risk/Dormant",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans.joblib"


def _validate_features(df: pd.DataFrame) -> None:
    missing = [c for c in BEHAVIORAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def fit_and_save(df: pd.DataFrame) -> tuple[StandardScaler, KMeans]:
    """
    Fit StandardScaler + KMeans on behavioral features and persist to models/.

    Returns
    -------
    scaler, kmeans
    """
    _validate_features(df)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # StandardScaler: subtract mean, divide by std so each feature is ~N(0,1).
    # K-Means uses Euclidean distance — without scaling, large-range features
    # (e.g. content_views) would dominate small-range ones (e.g. shares).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[BEHAVIORAL_FEATURES])

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,  # run 10 initializations; keep the lowest-inertia result
    )
    kmeans.fit(X_scaled)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, KMEANS_PATH)
    return scaler, kmeans


def load_models() -> tuple[StandardScaler, KMeans]:
    """Load the fitted scaler and KMeans from models/."""
    if not SCALER_PATH.exists() or not KMEANS_PATH.exists():
        raise FileNotFoundError(
            f"Missing model files in {MODELS_DIR}. "
            "Run fit_and_save() or notebooks/02_clustering.ipynb first."
        )
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    return scaler, kmeans


def assign_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `cluster` (int) and `cluster_name` (str) columns using the saved models.

    Parameters
    ----------
    df : pd.DataFrame
        Raw user data including the behavioral feature columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with cluster + cluster_name appended.
    """
    _validate_features(df)
    scaler, kmeans = load_models()

    # Use transform (not fit_transform) so we apply the SAME scaling as training
    X_scaled = scaler.transform(df[BEHAVIORAL_FEATURES])
    labels = kmeans.predict(X_scaled)

    out = df.copy()
    out["cluster"] = labels
    # map() replaces each cluster id with its human-readable name
    out["cluster_name"] = out["cluster"].map(CLUSTER_NAMES)
    return out


if __name__ == "__main__":
    # Convenience: fit on users.csv, assign clusters, write users_with_clusters.csv
    data_path = PROJECT_ROOT / "data" / "users.csv"
    out_path = PROJECT_ROOT / "data" / "users_with_clusters.csv"

    raw = pd.read_csv(data_path)
    fit_and_save(raw)
    clustered = assign_clusters(raw)
    clustered.to_csv(out_path, index=False)

    print(f"Saved models -> {MODELS_DIR}")
    print(f"Saved clustered data -> {out_path}")
    print("\nCluster sizes:")
    print(clustered["cluster_name"].value_counts())
    print("\nChurn rate by cluster:")
    print(
        clustered.groupby("cluster_name")["churn"]
        .mean()
        .sort_values(ascending=False)
        .round(3)
    )
