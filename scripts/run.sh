#!/bin/bash
set -e

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[0/6] Pre-flight: fetching today's roster and injury report..."
python -m src.data.preflight

echo "[0b/6] Scraping real-time injury report via nbainjuries..."
python -m src.data.injury_scraper

echo "[1/6] Fetching NBA player data (last 60 days)..."
python -m src.data.pipeline

# Remaining steps are intentionally left unchanged in future prompts.
