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
from nba_api.stats.endpoints import commonallplayers, playergamelogs

from src.data.preflight import load_out_player_ids

DATA_DIR = Path("data/local")
RAW_DIR = DATA_DIR / "raw"
LOG_PATH = Path("logs/data_pipeline.log")


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

    all_players = commonallplayers.CommonAllPlayers(is_only_current_season=1)
    time.sleep(1)
    players_df = all_players.get_data_frames()[0]

    frames: list[pd.DataFrame] = []
    for _, player in players_df.iterrows():
        pid = int(player["PERSON_ID"])

        if pid in out_ids:
            logging.info("[pipeline] Skipping %s (Out/Inactive today)", player["DISPLAY_FIRST_LAST"])
            continue

        logs = playergamelogs.PlayerGameLogs(
            player_id_nullable=pid,
            date_from_nullable=(datetime.today() - timedelta(days=60)).strftime("%m/%d/%Y"),
            date_to_nullable=datetime.today().strftime("%m/%d/%Y"),
        )
        time.sleep(1)
        df = logs.get_data_frames()[0]
        if not df.empty:
            frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
