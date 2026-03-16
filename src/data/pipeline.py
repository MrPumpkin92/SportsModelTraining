"""
NBA Data Ingestion Pipeline
============================
Pulls the last 60 days of per-game box scores for all active NBA players
via the nba_api package and stores results locally (gitignored).

Directory layout produced:
    data/local/raw/YYYY-MM-DD/box_scores.parquet   ← raw, date-partitioned
    data/local/features/player_features.parquet     ← rolling-average features
    logs/data_pipeline.log                          ← append-only run log
"""

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLogs

# ---------------------------------------------------------------------------
# Paths (relative to repo root — run from there or set CWD accordingly)
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/local")
RAW_DIR  = DATA_DIR / "raw"
FEAT_DIR = DATA_DIR / "features"
LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "data_pipeline.log"

# Stats columns required in the output
STAT_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_DATE",
    "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT", "MIN", "PLUS_MINUS",
    "GAME_ID", "MATCHUP", "WL",
]

# Numeric stats for which rolling averages are computed
NUMERIC_STATS = [
    "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS",
]

ROLLING_WINDOWS = [5, 10, 20]

# nba_api recommends ≥1 s between requests to avoid rate-limiting
API_SLEEP = 1.0


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create required directories if they don't already exist."""
    for d in (RAW_DIR, FEAT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    """
    Configure root logger to write to both file and stdout.
    Called after ensure_dirs() so LOG_DIR is guaranteed to exist.
    """
    logger = logging.getLogger("nba_pipeline")
    if logger.handlers:
        # Avoid duplicate handlers on repeated in-process calls
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # File handler — append mode so every run is preserved
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _season_str(dt: datetime) -> str:
    """
    Convert a date to the NBA season string the date falls in.
    E.g. 2025-11-01 → '2025-26', 2026-02-01 → '2025-26'.
    NBA seasons start in October of year Y and end in June of year Y+1.
    """
    year = dt.year if dt.month >= 10 else dt.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def fetch_all_players_last_60_days(logger: logging.Logger) -> pd.DataFrame:
    """
    Fetch box scores for all active NBA players for the last 60 days.

    Uses the PlayerGameLogs bulk endpoint, which returns every player's
    game for a given date range in a single API call — far more efficient
    than looping per player.

    Returns a DataFrame with the columns defined in STAT_COLS.
    If the 60-day window straddles two seasons, both are fetched and merged.
    """
    today      = datetime.now()
    date_from  = today - timedelta(days=60)
    date_from_s = date_from.strftime("%m/%d/%Y")
    date_to_s   = today.strftime("%m/%d/%Y")

    logger.info(
        "Fetching player game logs | date range: %s → %s",
        date_from_s, date_to_s,
    )

    # Determine which season(s) overlap the 60-day window
    seasons_needed = {_season_str(date_from), _season_str(today)}
    frames: list[pd.DataFrame] = []

    for season in sorted(seasons_needed):
        logger.info("  Querying PlayerGameLogs for season %s ...", season)
        try:
            endpoint = PlayerGameLogs(
                season_nullable=season,
                date_from_nullable=date_from_s,
                date_to_nullable=date_to_s,
                league_id_nullable="00",  # NBA
            )
            time.sleep(API_SLEEP)  # respect rate-limit
            df_season = endpoint.get_data_frames()[0]
        except Exception as exc:
            logger.error("  Failed to fetch season %s: %s", season, exc)
            continue

        logger.info("  Season %s → %d rows returned", season, len(df_season))
        frames.append(df_season)

    if not frames:
        logger.warning("No data returned from API — returning empty DataFrame.")
        return pd.DataFrame(columns=STAT_COLS)

    df = pd.concat(frames, ignore_index=True)

    # Keep only the columns we care about (subset intersection to be safe)
    available = [c for c in STAT_COLS if c in df.columns]
    missing   = set(STAT_COLS) - set(available)
    if missing:
        logger.warning("Columns missing from API response: %s", missing)

    df = df[available].copy()

    # Normalise GAME_DATE to date string YYYY-MM-DD
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    # Deduplicate (shouldn't happen, but guard against overlapping season pulls)
    df = df.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])

    logger.info("Total rows after deduplication: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Raw storage
# ---------------------------------------------------------------------------

def save_raw_by_date(df: pd.DataFrame, logger: logging.Logger) -> None:
    """
    Partition the raw box-score DataFrame by GAME_DATE and write each
    partition to data/local/raw/YYYY-MM-DD/box_scores.parquet.
    Existing files for a date are overwritten (idempotent re-runs).
    """
    if df.empty:
        logger.warning("save_raw_by_date: received empty DataFrame, skipping.")
        return

    for game_date, group in df.groupby("GAME_DATE"):
        out_dir  = RAW_DIR / game_date
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "box_scores.parquet"
        group.reset_index(drop=True).to_parquet(out_path, index=False)
        logger.info("  Saved %d rows → %s", len(group), out_path)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling 5-, 10-, and 20-game averages per player for all
    numeric stats listed in NUMERIC_STATS.

    Returns a new DataFrame with additional columns:
        <STAT>_ROLL_<WINDOW>   e.g. PTS_ROLL_5, REB_ROLL_10, AST_ROLL_20
    """
    if df.empty:
        return df

    # Sort chronologically per player so rolling windows look backward
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # Cast numeric columns (API sometimes returns strings)
    for col in NUMERIC_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute rolling features grouped by player
    roll_frames: list[pd.DataFrame] = []

    for player_id, player_df in df.groupby("PLAYER_ID", sort=False):
        player_df = player_df.copy()
        for stat in NUMERIC_STATS:
            if stat not in player_df.columns:
                continue
            for window in ROLLING_WINDOWS:
                col_name = f"{stat}_ROLL_{window}"
                # shift(1) ensures we use only PAST games (no data leakage)
                player_df[col_name] = (
                    player_df[stat]
                    .shift(1)
                    .rolling(window=window, min_periods=1)
                    .mean()
                )
        roll_frames.append(player_df)

    if not roll_frames:
        return df

    return pd.concat(roll_frames, ignore_index=True)


def save_features(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Write the enriched feature DataFrame to a single Parquet file."""
    if df.empty:
        logger.warning("save_features: received empty DataFrame, skipping.")
        return

    out_path = FEAT_DIR / "player_features.parquet"
    df.reset_index(drop=True).to_parquet(out_path, index=False)
    logger.info("Feature store saved → %s  (%d rows, %d cols)",
                out_path, len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    Orchestrates the full ingestion pipeline:
        1. Bootstrap directories & logging
        2. Fetch last-60-day box scores from nba_api
        3. Save raw data partitioned by date
        4. Compute rolling feature windows
        5. Save feature store
        6. Log summary
    """
    ensure_dirs()
    logger = setup_logging()

    run_start = datetime.now()
    logger.info("=" * 60)
    logger.info("NBA Pipeline RUN START  %s", run_start.isoformat())
    logger.info("=" * 60)

    # 1. Fetch
    raw_df = fetch_all_players_last_60_days(logger)
    logger.info("Fetch complete | total rows: %d", len(raw_df))

    # 2. Save raw, date-partitioned
    logger.info("Saving raw partitions ...")
    save_raw_by_date(raw_df, logger)

    # 3. Feature engineering
    logger.info("Computing rolling features ...")
    feat_df = compute_rolling_features(raw_df)
    logger.info(
        "Features computed | rows: %d | new cols: %d",
        len(feat_df),
        len(feat_df.columns) - len(raw_df.columns),
    )

    # 4. Save feature store
    save_features(feat_df, logger)

    elapsed = (datetime.now() - run_start).total_seconds()
    logger.info("=" * 60)
    logger.info("NBA Pipeline RUN COMPLETE  elapsed=%.1fs", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
