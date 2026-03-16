"""
NBA Data Ingestion Pipeline
Pulls last 60 days of player game logs from nba_api.
Respects out_player_ids.txt written by preflight.py.
All data stored locally in data/local/ (gitignored).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from src.data.preflight import load_out_player_ids

DATA_DIR = Path("data/local")
RAW_DIR = DATA_DIR / "raw"
LOG_PATH = Path("logs/data_pipeline.log")

# Retry settings for the single bulk NBA API request.
REQUEST_DELAY = 3
MAX_RETRIES = 3


def _current_nba_season() -> str:
    """Return season string like '2025-26' based on current date."""
    today = datetime.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    end_year_short = (start_year + 1) % 100
    return f"{start_year}-{end_year_short:02d}"


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def fetch_all_players_last_60_days() -> pd.DataFrame:
    """Fetch box scores for active NBA players, last 60 days.
    Players in out_player_ids.txt are skipped entirely."""
    out_ids = load_out_player_ids()
    if out_ids:
        logging.info("Skipping %s Out/Inactive players from preflight", len(out_ids))
    else:
        logging.warning("No out_player_ids found. Continuing with all players.")

    date_from = (datetime.today() - timedelta(days=60)).strftime("%m/%d/%Y")
    date_to = datetime.today().strftime("%m/%d/%Y")
    season = _current_nba_season()

    logs_df = pd.DataFrame()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = leaguegamelog.LeagueGameLog(
                player_or_team_abbreviation="P",
                season=season,
                season_type_all_star="Regular Season",
                date_from_nullable=date_from,
                date_to_nullable=date_to,
            )
            logs_df = response.get_data_frames()[0]
            break
        except Exception as exc:
            backoff = REQUEST_DELAY * (2 ** attempt)
            if attempt < MAX_RETRIES:
                logging.warning(
                    "[pipeline] Bulk pull attempt %d/%d failed (%s) -- retrying in %ds",
                    attempt, MAX_RETRIES, exc, backoff,
                )
                time.sleep(backoff)
            else:
                logging.error(
                    "[pipeline] Bulk pull failed after %d attempts: %s",
                    MAX_RETRIES, exc,
                )
                return pd.DataFrame()

    if logs_df.empty:
        logging.warning("[pipeline] Bulk pull returned no rows")
        return logs_df

    logs_df["PLAYER_ID"] = pd.to_numeric(logs_df["PLAYER_ID"], errors="coerce")
    logs_df = logs_df.dropna(subset=["PLAYER_ID"]).copy()
    logs_df["PLAYER_ID"] = logs_df["PLAYER_ID"].astype(int)

    before = len(logs_df)
    if out_ids:
        logs_df = logs_df[~logs_df["PLAYER_ID"].isin(out_ids)].copy()
    removed = before - len(logs_df)
    logging.info(
        "[pipeline] Pulled %s rows in one request; filtered out %s rows for Out/Inactive players",
        before,
        removed,
    )

    return logs_df


def run_pipeline() -> None:
    """Entry point: fetch and write partitioned raw files by game date."""
    _setup_logging()
    logging.info("Starting data pipeline run")

    df = fetch_all_players_last_60_days()
    if df.empty:
        logging.warning("No rows fetched from nba_api")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
        for game_date, group in df.groupby("GAME_DATE"):
            partition_dir = RAW_DIR / str(game_date)
            partition_dir.mkdir(parents=True, exist_ok=True)
            out_file = partition_dir / "box_scores.parquet"
            group.to_parquet(out_file, index=False)
        logging.info("Saved %s rows across %s date partitions", len(df), df["GAME_DATE"].nunique())
    else:
        out_file = RAW_DIR / "box_scores.parquet"
        df.to_parquet(out_file, index=False)
        logging.info("Saved %s rows to %s", len(df), out_file)


if __name__ == "__main__":
    run_pipeline()