# User Engagement Intelligence

A portfolio project that segments users by behavior, predicts churn, and explains model predictions for a content/engagement platform.

**Problem:** Content platforms lose revenue when engaged users quietly stop returning. This project builds an end-to-end pipeline that (1) clusters users into behavioral tiers, (2) predicts who is likely to churn using classical ML and a lightweight neural net, and (3) explains *why* with SHAP — so product and retention teams can act on the drivers, not just a score.

## Progress

- **Day 1 — Data + EDA + Cleaning:** Generated a realistic 10,000-user synthetic dataset and completed exploratory analysis (churn rates, engagement breakdowns, and feature correlations).
- **Day 2 — User Segmentation (K-Means):** Clustered users into 4 behavioral tiers (Power Users, Steady Engagers, Casual Browsers, At-Risk/Dormant); At-Risk/Dormant shows ~46% churn vs ~0% for Power/Steady users.
- **Day 3 — Churn Prediction:** Compared Logistic Regression, Random Forest, and MLPClassifier; champion is **Logistic Regression** (test F1 ≈ 0.58, ROC-AUC ≈ 0.90, recall ≈ 0.84).
- **Day 4 — SHAP + SQL Features:** Global/local SHAP explanations for the champion model (`explain_user`), plus SQLite event tables and SQL aggregations that rebuild engagement features from raw logs.
