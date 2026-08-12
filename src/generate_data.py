"""
Generate a realistic synthetic user engagement dataset for the
User Engagement Intelligence portfolio project.

Churn is constructed from engagement signals (not random noise) so
downstream models can actually learn the relationship.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Reproducibility — same seed = same dataset every run
RNG = np.random.default_rng(42)
N_USERS = 10_000

# ---------------------------------------------------------------------------
# 1. Demographics & device
# ---------------------------------------------------------------------------
user_id = [f"U{str(i).zfill(5)}" for i in range(1, N_USERS + 1)]
age = RNG.integers(18, 66, size=N_USERS)  # inclusive 18–65

# Device mix roughly matches consumer apps: Android > iOS > Web
device_type = RNG.choice(
    ["Android", "iOS", "Web"],
    size=N_USERS,
    p=[0.45, 0.35, 0.20],
)

# ---------------------------------------------------------------------------
# 2. Latent "engagement propensity" (0–1)
#    High propensity → more sessions, views, likes, shares, notifications
#    Low propensity  → infrequent use and longer gaps since last login
# ---------------------------------------------------------------------------
engagement_propensity = RNG.beta(a=2.0, b=2.5, size=N_USERS)  # mild left skew

# Small device effect: Web users tend to engage a bit less than mobile
device_engagement_boost = np.where(
    device_type == "Web", -0.08,
    np.where(device_type == "iOS", 0.03, 0.0),
)
engagement_propensity = np.clip(engagement_propensity + device_engagement_boost, 0.02, 0.98)

# ---------------------------------------------------------------------------
# 3. Behavioral features derived from propensity (+ noise)
# ---------------------------------------------------------------------------
# Sessions: low engagers ~0–5, high engagers ~20–45
sessions_last_30_days = np.clip(
    RNG.poisson(lam=3 + engagement_propensity * 35, size=N_USERS),
    0,
    60,
).astype(int)

# Avg session duration (minutes): correlated with sessions
avg_session_duration_minutes = np.round(
    np.clip(
        1.5 + engagement_propensity * 18 + RNG.normal(0, 2.5, N_USERS),
        0.5,
        45.0,
    ),
    1,
)

# Content views scale with sessions
content_views_last_30_days = np.clip(
    RNG.poisson(lam=sessions_last_30_days * (1.5 + engagement_propensity * 2.5) + 1),
    0,
    300,
).astype(int)

# Likes ~ fraction of views
likes_last_30_days = np.clip(
    RNG.binomial(n=np.maximum(content_views_last_30_days, 1), p=0.08 + engagement_propensity * 0.25),
    0,
    150,
).astype(int)

# Shares are rarer than likes
shares_last_30_days = np.clip(
    RNG.binomial(n=np.maximum(likes_last_30_days, 1), p=0.05 + engagement_propensity * 0.15),
    0,
    40,
).astype(int)

# Days since last login: INVERSELY related to engagement
# High engagers → recent login (0–7 days); low engagers → 20–60+ days
days_since_last_login = np.clip(
    np.round(
        (1 - engagement_propensity) * 45
        + RNG.exponential(scale=5, size=N_USERS)
        - sessions_last_30_days * 0.3
    ).astype(int),
    0,
    90,
)

# Notification clicks: more engaged users click more
notifications_clicked_last_30_days = np.clip(
    RNG.poisson(lam=0.5 + engagement_propensity * 12 + sessions_last_30_days * 0.1),
    0,
    40,
).astype(int)

# ---------------------------------------------------------------------------
# 4. Churn label — learnable from features (logistic-style score + noise)
# ---------------------------------------------------------------------------
# Higher score → higher churn probability
churn_logit = (
    -2.2
    - 0.08 * sessions_last_30_days
    - 0.06 * avg_session_duration_minutes
    - 0.015 * content_views_last_30_days
    - 0.04 * likes_last_30_days
    - 0.08 * shares_last_30_days
    - 0.05 * notifications_clicked_last_30_days
    + 0.09 * days_since_last_login
    + np.where(device_type == "Web", 0.35, 0.0)  # Web slightly riskier
    + RNG.normal(0, 0.6, N_USERS)  # noise so it's not perfectly separable
)

churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = (RNG.random(N_USERS) < churn_prob).astype(int)

# ---------------------------------------------------------------------------
# 5. Assemble & save
# ---------------------------------------------------------------------------
df = pd.DataFrame(
    {
        "user_id": user_id,
        "age": age,
        "device_type": device_type,
        "sessions_last_30_days": sessions_last_30_days,
        "avg_session_duration_minutes": avg_session_duration_minutes,
        "content_views_last_30_days": content_views_last_30_days,
        "likes_last_30_days": likes_last_30_days,
        "shares_last_30_days": shares_last_30_days,
        "days_since_last_login": days_since_last_login,
        "notifications_clicked_last_30_days": notifications_clicked_last_30_days,
        "churn": churn,
    }
)

out_path = Path(__file__).resolve().parent.parent / "data" / "users.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Saved {len(df):,} users -> {out_path}")
print(f"Overall churn rate: {df['churn'].mean():.1%}")
print(df.head())
print("\nChurn rate by device:")
print(df.groupby("device_type")["churn"].mean().round(3))