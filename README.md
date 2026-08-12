# User Engagement Intelligence

A portfolio project that segments users by behavior, predicts churn, and explains model predictions for a content/engagement platform — then surfaces everything in an interactive Streamlit dashboard.

**Problem:** Content platforms lose revenue when engaged users quietly stop returning. This project builds an end-to-end pipeline that (1) clusters users into behavioral tiers, (2) predicts who is likely to churn using classical ML and a lightweight neural net, and (3) explains *why* with SHAP — so product and retention teams can act on the drivers, not just a score.

> **Screenshot:** After you run the dashboard (`streamlit run app.py`), add a screenshot of the Streamlit UI here (e.g. `docs/dashboard.png` or drop an image below this note).

## Architecture

```
Data (synthetic users + SQLite sessions/events)
    → pandas / SQL feature engineering
    → K-Means clustering (behavioral tiers)
    → Churn models (Logistic Regression · Random Forest · MLP)
    → SHAP (global beeswarm + local waterfalls)
    → Streamlit dashboard (app.py)
```

## How to run

```bash
# from the project root, with dependencies installed
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens a local browser tab (typically **http://localhost:8501**).

## Key Findings

- **At-Risk/Dormant** users churn at **~46.3%**, versus **~0.1%** for Power Users and **~0.3%** for Steady Engagers — cluster membership is the strongest behavioral signal.
- Champion model is **Logistic Regression** (test F1 ≈ 0.58, ROC-AUC ≈ 0.90, recall ≈ 0.84), chosen for high recall so retention teams catch churners even when precision is lower than Random Forest.
- SHAP highlights **days_since_last_login**, low **sessions_last_30_days**, and related engagement drops as the main drivers pushing individual predictions toward churn.
- SQL aggregations over raw session/event logs in SQLite can rebuild the same engagement features used by the models (`src/build_features_sql.py`).

## Tech stack

| Layer | Tools |
| --- | --- |
| Data | pandas, NumPy, SQLite |
| Visualization (EDA / notebooks) | matplotlib, seaborn |
| ML | scikit-learn (K-Means, Logistic Regression, Random Forest, MLPClassifier) |
| Explainability | SHAP |
| Persistence | joblib |
| Dashboard | Streamlit, Plotly |
| Notebooks | Jupyter |

## Progress

- **Day 1 — Data + EDA + Cleaning:** Generated a realistic 10,000-user synthetic dataset and completed exploratory analysis (churn rates, engagement breakdowns, and feature correlations).
- **Day 2 — User Segmentation (K-Means):** Clustered users into 4 behavioral tiers (Power Users, Steady Engagers, Casual Browsers, At-Risk/Dormant); At-Risk/Dormant shows ~46% churn vs ~0% for Power/Steady users.
- **Day 3 — Churn Prediction:** Compared Logistic Regression, Random Forest, and MLPClassifier; champion is **Logistic Regression** (test F1 ≈ 0.58, ROC-AUC ≈ 0.90, recall ≈ 0.84).
- **Day 4 — SHAP + SQL Features:** Global/local SHAP explanations for the champion model (`explain_user`), plus SQLite event tables and SQL aggregations that rebuild engagement features from raw logs.
- **Day 5 — Streamlit Dashboard:** Interactive multi-section app (`app.py`) with KPIs, cluster churn chart, model comparison, global SHAP beeswarm, and filtered user lookup with local SHAP waterfalls.
