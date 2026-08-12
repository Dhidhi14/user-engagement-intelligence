"""
SQL feature engineering for User Engagement Intelligence (Day 4).

Simulates the real-world path: raw event tables in SQLite -> aggregated
features ready for modeling (the step that normally happens BEFORE the
clean CSV used in Days 1-3).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "users_with_clusters.csv"
DEFAULT_DB = PROJECT_ROOT / "data" / "engagement.db"

RNG = np.random.default_rng(42)


def build_engagement_db(
    csv_path: str | Path = DEFAULT_CSV,
    db_path: str | Path = DEFAULT_DB,
) -> Path:
    """
    Build data/engagement.db with three tables from users_with_clusters.csv:
      - users
      - sessions          (synthetic rows consistent with session counts/durations)
      - engagement_events (synthetic rows consistent with views/likes/shares/clicks)
    """
    csv_path = Path(csv_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    # Anchor "today" so days_since_last_login maps to a concrete last session date
    today = pd.Timestamp("2026-08-12")

    users = df[["user_id", "age", "device_type"]].copy()

    session_rows = []
    event_rows = []

    for row in df.itertuples(index=False):
        user_id = row.user_id
        n_sessions = int(row.sessions_last_30_days)
        avg_dur = float(row.avg_session_duration_minutes)
        days_since = int(row.days_since_last_login)

        last_login = today - pd.Timedelta(days=days_since)

        # Create one row per session counted in the CSV (0 sessions => no rows)
        if n_sessions > 0:
            # Spread sessions across the 30 days before last_login
            day_offsets = RNG.integers(0, 30, size=n_sessions)
            # Force the most recent session onto last_login so days-since matches
            day_offsets[0] = 0
            for offset in day_offsets:
                session_date = last_login - pd.Timedelta(days=int(offset))
                # Duration centered on the user's avg, with small noise
                duration = float(
                    np.clip(avg_dur + RNG.normal(0, max(avg_dur * 0.15, 0.3)), 0.5, 60.0)
                )
                session_rows.append(
                    {
                        "user_id": user_id,
                        "session_date": session_date.strftime("%Y-%m-%d"),
                        "session_duration_minutes": round(duration, 2),
                    }
                )

        # Engagement events — counts match the CSV columns
        event_specs = [
            ("view", int(row.content_views_last_30_days)),
            ("like", int(row.likes_last_30_days)),
            ("share", int(row.shares_last_30_days)),
            ("notification_click", int(row.notifications_clicked_last_30_days)),
        ]
        for event_type, count in event_specs:
            if count <= 0:
                continue
            offsets = RNG.integers(0, 30, size=count)
            for offset in offsets:
                event_date = last_login - pd.Timedelta(days=int(offset))
                event_rows.append(
                    {
                        "user_id": user_id,
                        "event_type": event_type,
                        "event_date": event_date.strftime("%Y-%m-%d"),
                    }
                )

    sessions = pd.DataFrame(session_rows)
    events = pd.DataFrame(event_rows)

    # Replace DB if it already exists
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        users.to_sql("users", conn, index=False, if_exists="replace")
        sessions.to_sql("sessions", conn, index=False, if_exists="replace")
        events.to_sql("engagement_events", conn, index=False, if_exists="replace")

        # Helpful indexes for JOIN / GROUP BY queries
        conn.execute("CREATE INDEX idx_sessions_user ON sessions(user_id)")
        conn.execute("CREATE INDEX idx_events_user ON engagement_events(user_id)")
        conn.execute(
            "CREATE INDEX idx_events_user_type ON engagement_events(user_id, event_type)"
        )

    print(f"Built {db_path}")
    print(f"  users:             {len(users):,}")
    print(f"  sessions:          {len(sessions):,}")
    print(f"  engagement_events: {len(events):,}")
    return db_path


def compute_features_from_db(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """
    Run SQL aggregations against engagement.db and return a clean feature table.

    Columns returned (aligned with the modeling CSV where possible):
      user_id, age, device_type,
      sessions_last_30_days, avg_session_duration_minutes, days_since_last_login,
      content_views_last_30_days, likes_last_30_days, shares_last_30_days,
      notifications_clicked_last_30_days
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Run build_engagement_db() first."
        )

    # Reference "today" must match the date used when generating the DB
    as_of_date = "2026-08-12"

    # --- Query 1: sessions per user (COUNT + AVG duration) ---
    # GROUP BY user_id compresses many session rows into one summary row.
    q_sessions = """
        SELECT
            user_id,
            COUNT(*) AS sessions_last_30_days,
            AVG(session_duration_minutes) AS avg_session_duration_minutes,
            -- julianday difference = days between as-of date and last session
            CAST(julianday(?) - julianday(MAX(session_date)) AS INTEGER)
                AS days_since_last_login
        FROM sessions
        GROUP BY user_id
    """

    # --- Query 2: event counts pivoted by event_type ---
    # SUM(CASE WHEN ...) is a classic SQL pivot: one column per event type.
    q_events = """
        SELECT
            user_id,
            SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END)
                AS content_views_last_30_days,
            SUM(CASE WHEN event_type = 'like' THEN 1 ELSE 0 END)
                AS likes_last_30_days,
            SUM(CASE WHEN event_type = 'share' THEN 1 ELSE 0 END)
                AS shares_last_30_days,
            SUM(CASE WHEN event_type = 'notification_click' THEN 1 ELSE 0 END)
                AS notifications_clicked_last_30_days
        FROM engagement_events
        GROUP BY user_id
    """

    # --- Query 3: join users with session aggregates ---
    # LEFT JOIN keeps users who have zero sessions (no matching session rows).
    q_user_sessions = """
        SELECT
            u.user_id,
            u.age,
            u.device_type,
            COALESCE(s.sessions_last_30_days, 0) AS sessions_last_30_days,
            s.avg_session_duration_minutes,
            s.days_since_last_login
        FROM users u
        LEFT JOIN (
            SELECT
                user_id,
                COUNT(*) AS sessions_last_30_days,
                AVG(session_duration_minutes) AS avg_session_duration_minutes,
                CAST(julianday(?) - julianday(MAX(session_date)) AS INTEGER)
                    AS days_since_last_login
            FROM sessions
            GROUP BY user_id
        ) s ON u.user_id = s.user_id
    """

    # --- Query 4: full feature table (users + sessions + events) ---
    q_full = """
        SELECT
            u.user_id,
            u.age,
            u.device_type,
            COALESCE(s.sessions_last_30_days, 0) AS sessions_last_30_days,
            ROUND(COALESCE(s.avg_session_duration_minutes, 0), 2)
                AS avg_session_duration_minutes,
            COALESCE(s.days_since_last_login, NULL) AS days_since_last_login,
            COALESCE(e.content_views_last_30_days, 0) AS content_views_last_30_days,
            COALESCE(e.likes_last_30_days, 0) AS likes_last_30_days,
            COALESCE(e.shares_last_30_days, 0) AS shares_last_30_days,
            COALESCE(e.notifications_clicked_last_30_days, 0)
                AS notifications_clicked_last_30_days
        FROM users u
        LEFT JOIN (
            SELECT
                user_id,
                COUNT(*) AS sessions_last_30_days,
                AVG(session_duration_minutes) AS avg_session_duration_minutes,
                CAST(julianday(?) - julianday(MAX(session_date)) AS INTEGER)
                    AS days_since_last_login
            FROM sessions
            GROUP BY user_id
        ) s ON u.user_id = s.user_id
        LEFT JOIN (
            SELECT
                user_id,
                SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END)
                    AS content_views_last_30_days,
                SUM(CASE WHEN event_type = 'like' THEN 1 ELSE 0 END)
                    AS likes_last_30_days,
                SUM(CASE WHEN event_type = 'share' THEN 1 ELSE 0 END)
                    AS shares_last_30_days,
                SUM(CASE WHEN event_type = 'notification_click' THEN 1 ELSE 0 END)
                    AS notifications_clicked_last_30_days
            FROM engagement_events
            GROUP BY user_id
        ) e ON u.user_id = e.user_id
        ORDER BY u.user_id
    """

    with sqlite3.connect(db_path) as conn:
        # Expose the individual queries for the notebook / debugging
        _ = pd.read_sql_query(q_sessions, conn, params=(as_of_date,))
        _ = pd.read_sql_query(q_events, conn)
        _ = pd.read_sql_query(q_user_sessions, conn, params=(as_of_date,))
        features = pd.read_sql_query(q_full, conn, params=(as_of_date,))

    return features


if __name__ == "__main__":
    build_engagement_db()
    feats = compute_features_from_db()
    print("\nSQL-derived features (head):")
    print(feats.head())
    print(f"\nShape: {feats.shape}")
