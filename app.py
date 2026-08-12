"""
User Engagement Intelligence — Day 5 Streamlit Dashboard

End-to-end portfolio view:
  Data → pandas/SQL features → K-Means clusters → churn models → SHAP → this UI

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Allow `from src.explain import ...` when launched via `streamlit run app.py`
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import explain_user, rebuild_feature_matrix  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "users_with_clusters.csv"
MODELS_DIR = PROJECT_ROOT / "models"
CHURN_MODEL_PATH = MODELS_DIR / "churn_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"
SHAP_VALUES_PATH = MODELS_DIR / "shap_values.joblib"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

# Same split used in notebooks/03_churn_prediction.ipynb
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Display order for clusters (At-Risk first so the finding is obvious)
CLUSTER_ORDER = [
    "At-Risk/Dormant",
    "Casual Browsers",
    "Steady Engagers",
    "Power Users",
]

# Stats shown in the User Lookup table
USER_STAT_COLS = [
    "user_id",
    "cluster_name",
    "device_type",
    "sessions_last_30_days",
    "days_since_last_login",
    "content_views_last_30_days",
    "likes_last_30_days",
    "shares_last_30_days",
    "avg_session_duration_minutes",
    "notifications_clicked_last_30_days",
    "churn",
]


# ---------------------------------------------------------------------------
# Caching
# @st.cache_data  → serializable / data-like objects (DataFrames, arrays, dicts).
#                   Streamlit hashes inputs and can pickle the return value.
# @st.cache_resource → non-data objects that should stay in memory (fitted models,
#                      SHAP explainers). These are not cheaply pickleable and must
#                      not be rebuilt on every widget interaction.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading user data…")
def load_users() -> pd.DataFrame:
    """Load the clustered user table (CSV is data → cache_data)."""
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner="Loading model comparison…")
def load_comparison() -> pd.DataFrame:
    """Day 3 precision/recall/F1/AUC table saved under models/."""
    return pd.read_csv(COMPARISON_PATH, index_col="model")


@st.cache_data(show_spinner="Loading SHAP values…")
def load_shap_bundle() -> dict:
    """
    Pre-computed SHAP matrix + alignment metadata from Day 4.
    Stored as a dict of arrays/lists → cache_data (not cache_resource).
    """
    return joblib.load(SHAP_VALUES_PATH)


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """
    Fitted Logistic Regression + SHAP LinearExplainer.
    These are heavy sklearn/shap objects → cache_resource so they stay resident.
    """
    model = joblib.load(CHURN_MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    explainer = joblib.load(SHAP_EXPLAINER_PATH)
    return model, feature_names, explainer


@st.cache_data(show_spinner="Computing champion metrics…")
def compute_champion_metrics(_model, feature_names: list[str]) -> dict[str, float]:
    """
    Re-evaluate the *saved* champion on the Day-3 holdout split.
    Numbers come from the artifact + data, not hardcoded notebook memory.
    (Leading underscore on _model tells Streamlit not to hash the model object.)
    """
    df = load_users()
    X = rebuild_feature_matrix(df, feature_names)
    y = df["churn"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    y_pred = _model.predict(X_test)
    y_proba = _model.predict_proba(X_test)[:, 1]
    return {
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def churn_rate_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster size + churn rate for the overview chart/table."""
    summary = (
        df.groupby("cluster_name", as_index=False)
        .agg(n_users=("user_id", "count"), churn_rate=("churn", "mean"))
        .sort_values("churn_rate", ascending=False)
    )
    # Stable categorical order for the bar chart
    summary["cluster_name"] = pd.Categorical(
        summary["cluster_name"], categories=CLUSTER_ORDER, ordered=True
    )
    return summary.sort_values("cluster_name")


def risk_style(proba: float) -> tuple[str, str, str]:
    """Return (hex color, label, emoji-free status) for churn probability."""
    if proba > 0.70:
        return "#c62828", "High risk", "> 70%"
    if proba >= 0.40:
        return "#f9a825", "Medium risk", "40–70%"
    return "#2e7d32", "Low risk", "< 40%"


def render_proba_indicator(proba: float) -> None:
    """Large colored probability callout for the selected user."""
    color, label, band = risk_style(proba)
    st.markdown(
        f"""
        <div style="
            background:{color};
            color:white;
            padding:1.25rem 1.5rem;
            border-radius:8px;
            text-align:center;
            margin:0.5rem 0 1rem 0;
        ">
            <div style="font-size:0.95rem;opacity:0.9;">{label} ({band})</div>
            <div style="font-size:2.4rem;font-weight:700;line-height:1.2;">
                {proba:.0%} churn probability
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_cluster_churn_bar(summary: pd.DataFrame):
    """Plotly bar chart — At-Risk/Dormant should visually dominate."""
    plot_df = summary.copy()
    plot_df["churn_pct"] = plot_df["churn_rate"] * 100
    # Highlight At-Risk/Dormant in a stronger red; others muted
    colors = {
        "At-Risk/Dormant": "#c62828",
        "Casual Browsers": "#90a4ae",
        "Steady Engagers": "#78909c",
        "Power Users": "#546e7a",
    }
    fig = px.bar(
        plot_df,
        x="cluster_name",
        y="churn_pct",
        text=plot_df["churn_pct"].map(lambda v: f"{v:.1f}%"),
        color="cluster_name",
        color_discrete_map=colors,
        labels={
            "cluster_name": "Cluster",
            "churn_pct": "Churn rate (%)",
        },
        title="Churn rate by behavioral cluster",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        showlegend=False,
        yaxis_range=[0, max(55, plot_df["churn_pct"].max() * 1.25)],
        margin=dict(t=50, b=40),
        height=420,
    )
    return fig


def plot_shap_beeswarm(shap_values: np.ndarray, X: pd.DataFrame) -> plt.Figure:
    """Global SHAP summary (beeswarm) from pre-computed values — no recompute."""
    # summary_plot draws into the current matplotlib figure
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=list(X.columns),
        show=False,
        max_display=15,
    )
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_shap_waterfall(explanation) -> plt.Figure:
    """Local SHAP waterfall for one Explanation row."""
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page config + load artifacts
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="User Engagement Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("User Engagement Intelligence")
st.caption(
    "Behavioral clustering → churn prediction → SHAP explanations — "
    "interactive portfolio dashboard (Day 5)."
)

df = load_users()
model, feature_names, explainer = load_models()
shap_bundle = load_shap_bundle()
comparison = load_comparison()
champion_metrics = compute_champion_metrics(model, list(feature_names))

# Feature matrix in the same row order as Day-4 SHAP values (full CSV order)
X_all = rebuild_feature_matrix(df, feature_names)
shap_values = np.asarray(shap_bundle["shap_values"])


# ===========================================================================
# a) KPI ROW
# ===========================================================================
total_users = len(df)
overall_churn = float(df["churn"].mean())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Users", f"{total_users:,}")
k2.metric("Overall Churn Rate", f"{overall_churn:.1%}")
k3.metric("Champion Model F1", f"{champion_metrics['f1']:.3f}")
k4.metric("Champion Model ROC-AUC", f"{champion_metrics['roc_auc']:.3f}")

st.divider()


# ===========================================================================
# b) CLUSTER OVERVIEW
# ===========================================================================
st.header("Cluster Overview")
st.write(
    "K-Means (k=4) segments users by engagement behavior. "
    "**At-Risk/Dormant** dominates churn — the clearest Day-2 finding."
)

cluster_summary = churn_rate_by_cluster(df)
c_chart, c_table = st.columns([2, 1])
with c_chart:
    st.plotly_chart(plot_cluster_churn_bar(cluster_summary), use_container_width=True)
with c_table:
    st.subheader("Cluster sizes")
    table_view = cluster_summary.copy()
    table_view["churn_rate"] = table_view["churn_rate"].map(lambda x: f"{x:.1%}")
    table_view = table_view.rename(
        columns={
            "cluster_name": "Cluster",
            "n_users": "Users",
            "churn_rate": "Churn rate",
        }
    )
    st.dataframe(table_view, hide_index=True, use_container_width=True)

st.divider()


# ===========================================================================
# c) MODEL COMPARISON
# ===========================================================================
st.header("Model Comparison")
st.write(
    "Day 3 holdout comparison (same stratified 80/20 split, `random_state=42`). "
    "Champion: **Logistic Regression**."
)
st.dataframe(comparison.round(3), use_container_width=True)
st.caption(
    "Logistic Regression was chosen despite lower precision because **recall** "
    "matters more for churn: missing a true churner (false negative) usually "
    "costs more than messaging a user who would have stayed. LR catches ~84% of "
    "churners with the best F1 and ROC-AUC; Random Forest is more precise but "
    "misses most churners."
)

st.divider()


# ===========================================================================
# d) GLOBAL EXPLAINABILITY
# ===========================================================================
st.header("Global Explainability (SHAP)")
st.write(
    "Beeswarm summary from **pre-computed** `shap_values.joblib` (Day 4). "
    "Red = higher feature value; position on the x-axis = push toward/away from churn."
)

with st.spinner("Rendering SHAP beeswarm…"):
    beeswarm_fig = plot_shap_beeswarm(shap_values, X_all)
    st.pyplot(beeswarm_fig, clear_figure=True)
    plt.close("all")

st.divider()


# ===========================================================================
# f) SIDEBAR FILTERS  (defined before user lookup so the selectbox is filtered)
# ===========================================================================
st.sidebar.header("User Lookup Filters")
st.sidebar.write("Narrow the dropdown — the full table has 10,000 user IDs.")

all_clusters = sorted(df["cluster_name"].dropna().unique().tolist())
all_devices = sorted(df["device_type"].dropna().unique().tolist())

selected_clusters = st.sidebar.multiselect(
    "cluster_name",
    options=all_clusters,
    default=all_clusters,
)
selected_devices = st.sidebar.multiselect(
    "device_type",
    options=all_devices,
    default=all_devices,
)

filtered = df[
    df["cluster_name"].isin(selected_clusters) & df["device_type"].isin(selected_devices)
]
user_ids = filtered["user_id"].sort_values().tolist()
st.sidebar.caption(f"{len(user_ids):,} users match the current filters.")


# ===========================================================================
# e) USER LOOKUP
# ===========================================================================
st.header("User Lookup")
st.write(
    "Pick a user to see engagement stats, predicted churn risk, a SHAP waterfall, "
    "and a plain-English explanation from `explain_user()`."
)

if not user_ids:
    st.warning("No users match the sidebar filters. Broaden cluster_name / device_type.")
else:
    # Default to first filtered ID; keep selection stable when possible
    default_idx = 0
    user_id = st.selectbox("user_id", options=user_ids, index=default_idx)

    user_row = df.loc[df["user_id"] == user_id].iloc[0]
    X_row = rebuild_feature_matrix(df.loc[df["user_id"] == user_id], feature_names)
    proba = float(model.predict_proba(X_row)[0, 1])

    left, right = st.columns([1, 1])
    with left:
        st.subheader("User stats")
        stats = user_row[USER_STAT_COLS].to_frame(name="value")
        st.dataframe(stats, use_container_width=True)
    with right:
        st.subheader("Predicted churn risk")
        render_proba_indicator(proba)

    st.subheader("SHAP waterfall (this user)")
    # Graceful failure: waterfall / Explanation API quirks should not kill the app
    try:
        explanation = explainer(X_row)
        waterfall_fig = plot_shap_waterfall(explanation)
        st.pyplot(waterfall_fig, clear_figure=True)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001 — show friendly UI message
        st.error(
            "Could not render the SHAP waterfall for this user. "
            "The probability and plain-English explanation below still work."
        )
        st.caption(f"Details: {exc}")

    st.subheader("Plain-English explanation")
    try:
        # explain_user prints to the terminal and returns the sentence
        sentence = explain_user(
            user_id,
            df,
            model,
            explainer,
            feature_names,
        )
        st.info(sentence)
    except Exception as exc:  # noqa: BLE001
        st.error("Could not generate the plain-English explanation for this user.")
        st.caption(f"Details: {exc}")

st.divider()
st.caption(
    "Artifacts: `data/users_with_clusters.csv` · `models/churn_model.joblib` · "
    "`models/shap_*.joblib` · `models/model_comparison.csv` · `src/explain.py`"
)
