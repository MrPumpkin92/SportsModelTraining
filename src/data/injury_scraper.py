# src/data/injury_scraper.py
"""
nbainjuries-based real-time injury scraper for today's NBA games.

Uses the nbainjuries package (pip install nbainjuries) which parses
official NBA injury report PDFs stored on the NBA's server.

Requires: Java 8+ in system PATH (for tabula-py PDF parsing)

Outputs (all gitignored):
  data/local/injury_report_nbainjuries.json  — raw nbainjuries output
  data/local/today_roster.json               — UPDATED with enriched injury data
  data/local/out_player_ids.txt              — UPDATED with newly discovered Out players
"""

import json
import os
import shutil
import sys
import time
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR          = Path("data/local")
RAW_INJURY_PATH   = DATA_DIR / "injury_report_nbainjuries.json"
TODAY_ROSTER_PATH = DATA_DIR / "today_roster.json"
OUT_PLAYERS_PATH  = DATA_DIR / "out_player_ids.txt"

# ── Status normalization: nbainjuries → our internal labels ─────────────────
STATUS_MAP = {
    "Out":                "Out",
    "Doubtful":           "Doubtful",
    "Questionable":       "Questionable",
    "Probable":           "Probable",
    "Available":          "Active",
    "Game Time Decision": "GTD",
    "GTD":                "GTD",
}

PERFORMANCE_MODIFIERS = {
    "Active":       1.00,
    "Probable":     0.97,
    "Questionable": 0.88,
    "Doubtful":     0.65,
    "GTD":          0.80,
    "Out":          0.00,
    "Inactive":     0.00,
}

SKIP_STATUSES = {"Out", "Inactive"}


def _find_java_executable() -> Optional[str]:
    """Find java executable via PATH, JAVA_HOME, and common Windows locations."""
    java_on_path = shutil.which("java")
    if java_on_path:
        return java_on_path

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / "java.exe"
        if candidate.exists():
            return str(candidate)

    common_windows_candidates = [
        Path("C:/Program Files/Java/latest/bin/java.exe"),
        Path("C:/Program Files/Java/jre1.8.0_481/bin/java.exe"),
    ]
    for candidate in common_windows_candidates:
        if candidate.exists():
            return str(candidate)

    java_roots = [Path("C:/Program Files/Java"), Path("C:/Program Files (x86)/Java")]
    for root in java_roots:
        if not root.exists():
            continue
        for exe in root.glob("*/bin/java.exe"):
            if exe.exists():
                return str(exe)

    return None


def _check_java() -> bool:
    """Returns True if Java is available; auto-adds discovered Java bin to PATH."""
    import subprocess

    java_executable = _find_java_executable()
    if not java_executable:
        return False

    java_bin = str(Path(java_executable).parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if java_bin not in path_parts:
        os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")

    try:
        result = subprocess.run(
            [java_executable, "-version"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def fetch_todays_injury_report() -> list:
    """
    Fetches the most recent NBA injury report for today using nbainjuries.

    Strategy:
      1. Try fetching at the most recent 15-minute boundary
      2. If that fails, walk back in 15-minute steps up to 4 hours
      3. If all fail, try yesterday's final report as a fallback
      4. If all fail, return [] (caller handles gracefully)
    """
    try:
        from nbainjuries import injury
    except ImportError:
        if sys.version_info < (3, 10):
            logger.error(
                "[injury_scraper] nbainjuries requires Python >=3.10. "
                "Current Python is %d.%d. "
                "Run this module with a newer interpreter.",
                sys.version_info.major,
                sys.version_info.minor,
            )
            return []
        logger.error(
            "[injury_scraper] nbainjuries not installed. "
            "Run: pip install nbainjuries"
        )
        return []

    now = datetime.now()

    # Build candidate timestamps: today's 15-min boundaries from now backwards
    candidates = []
    current = now.replace(second=0, microsecond=0)
    current = current.replace(minute=(current.minute // 15) * 15)

    for _ in range(16):   # up to 4 hours back in 15-min steps
        candidates.append(current)
        current = current - timedelta(minutes=15)

    # Also try yesterday's 11pm as final fallback
    yesterday_final = (now - timedelta(days=1)).replace(
        hour=23, minute=0, second=0, microsecond=0
    )
    candidates.append(yesterday_final)

    last_error = None
    for candidate_dt in candidates:
        try:
            logger.info(
                "[injury_scraper] Trying nbainjuries at %s",
                candidate_dt.strftime("%Y-%m-%d %H:%M"),
            )
            result = injury.get_reportdata(candidate_dt)

            if result is not None and len(result) > 0:
                logger.info(
                    "[injury_scraper] Got %d injury records from %s",
                    len(result),
                    candidate_dt.strftime("%Y-%m-%d %H:%M"),
                )
                if isinstance(result, list):
                    return result
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, list):
                            return parsed
                        if isinstance(parsed, dict):
                            return [parsed]
                    except json.JSONDecodeError:
                        logger.warning("[injury_scraper] nbainjuries returned non-JSON string payload")
                    return []
                if hasattr(result, "to_dict"):
                    return result.to_dict(orient="records")
                if isinstance(result, dict):
                    return [result]
                return []

        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
            continue

    logger.warning(
        "[injury_scraper] All timestamp attempts failed. Last error: %s", last_error
    )
    return []


def normalize_status(raw_status: str) -> str:
    """Normalizes nbainjuries status strings to our internal labels."""
    return STATUS_MAP.get(raw_status.strip(), "Active")


def normalize_player_name(name_last_first: str) -> str:
    """Converts 'James, LeBron' → 'LeBron James'."""
    parts = [p.strip() for p in name_last_first.split(",")]
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return name_last_first


def build_injury_map(records: list) -> dict:
    """
    Converts raw nbainjuries records into a lookup dict keyed by normalized
    player name.
    """
    injury_map = {}
    for record in records:
        raw_name   = record.get("Player Name", "")
        raw_status = record.get("Current Status", "Available")
        raw_reason = record.get("Reason", "")
        matchup    = record.get("Matchup", "")
        game_time  = record.get("Game Time", "")
        team       = record.get("Team", "")

        if not raw_name:
            continue

        norm_name   = normalize_player_name(raw_name)
        norm_status = normalize_status(raw_status)

        # Strip "Injury/Illness - " prefix for cleaner display
        clean_reason = raw_reason
        if " - " in raw_reason:
            clean_reason = raw_reason.split(" - ", 1)[1].strip()

        injury_map[norm_name] = {
            "status":    norm_status,
            "reason":    clean_reason,
            "matchup":   matchup,
            "game_time": game_time,
            "team":      team,
            "modifier":  PERFORMANCE_MODIFIERS.get(norm_status, 1.0),
        }

    return injury_map


def merge_into_today_roster(injury_map: dict) -> tuple:
    """
    Loads today_roster.json, merges enriched injury data from nbainjuries,
    and writes it back in-place.

    Returns: (n_updated, n_newly_out)
    """
    if not TODAY_ROSTER_PATH.exists():
        logger.error(
            "[injury_scraper] today_roster.json not found. "
            "Run Prompt 00 (preflight.py) first."
        )
        return 0, 0

    roster  = json.loads(TODAY_ROSTER_PATH.read_text(encoding="utf-8"))
    players = roster.get("players", [])

    n_updated   = 0
    n_newly_out = 0

    for player in players:
        name  = player["player_name"]
        match = injury_map.get(name)

        if match:
            old_status = player.get("injury_status", "Active")
            new_status = match["status"]

            player["injury_status"]        = new_status
            player["injury_description"]   = match["reason"]
            player["performance_modifier"] = match["modifier"]
            player["include_in_pipeline"]  = new_status not in SKIP_STATUSES
            player["injury_source"]        = "nbainjuries"

            if new_status != old_status:
                n_updated += 1
                logger.info(
                    "[injury_scraper] %s: %s → %s (%s)",
                    name, old_status, new_status, match["reason"],
                )
                if new_status in SKIP_STATUSES and old_status not in SKIP_STATUSES:
                    n_newly_out += 1
                    logger.warning(
                        "[injury_scraper] %s newly discovered as %s — "
                        "will be excluded from pipeline",
                        name, new_status,
                    )
        else:
            # Not on injury report → assumed Active / healthy
            if player.get("injury_source") != "nbainjuries":
                player.setdefault("injury_status", "Active")
                player.setdefault("injury_source", "assumed_active")

    TODAY_ROSTER_PATH.write_text(
        json.dumps(roster, indent=2), encoding="utf-8"
    )
    logger.info(
        "[injury_scraper] today_roster.json updated: %d status changes, %d newly Out",
        n_updated, n_newly_out,
    )
    return n_updated, n_newly_out


def refresh_out_player_ids() -> int:
    """
    Re-reads today_roster.json and rewrites out_player_ids.txt to reflect any
    newly discovered Out/Inactive players.

    Returns: total number of Out/Inactive players written to file.
    """
    if not TODAY_ROSTER_PATH.exists():
        return 0

    roster  = json.loads(TODAY_ROSTER_PATH.read_text(encoding="utf-8"))
    out_ids = [
        str(p["player_id"])
        for p in roster.get("players", [])
        if not p.get("include_in_pipeline", True)
    ]

    OUT_PLAYERS_PATH.write_text("\n".join(out_ids), encoding="utf-8")
    logger.info(
        "[injury_scraper] out_player_ids.txt: %d players excluded", len(out_ids)
    )
    return len(out_ids)


def print_injury_summary(injury_map: dict) -> None:
    """Prints a formatted per-matchup injury summary to stdout."""
    if not injury_map:
        print("[injury_scraper] No injury data retrieved — all players assumed Active.")
        return

    by_matchup: dict = {}
    for name, info in injury_map.items():
        matchup = info.get("matchup", "Unknown")
        by_matchup.setdefault(matchup, []).append((name, info))

    print(f"\n[injury_scraper] NBA Injury Report — {date.today().isoformat()}")
    print("=" * 65)
    for matchup, players in sorted(by_matchup.items()):
        print(f"\n  {matchup}  ({players[0][1].get('game_time', '')})")
        for name, info in sorted(players, key=lambda x: x[1]["status"]):
            status   = info["status"]
            reason   = info["reason"]
            badge = {
                "Out":          "[OUT]",
                "Doubtful":     "[DOUBT]",
                "Questionable": "[QUEST]",
                "GTD":          "[GTD]",
                "Probable":     "[PROB]",
                "Active":       "[OK]",
            }.get(status, "[?]")
            print(f"    {badge} {name:<25s}  {status:<14s}  {reason}")
    print()


def run_injury_scraper() -> None:
    """
    Entry point. Called after preflight.py (Prompt 00).

    1. Check Java availability
    2. Fetch today's injury report via nbainjuries
    3. Save raw output
    4. Build injury map and merge into today_roster.json
    5. Refresh out_player_ids.txt
    6. Print summary
    """
    print(
        f"[injury_scraper] Running nbainjuries scraper for "
        f"{date.today().isoformat()}"
    )

    # Step 1: Java check
    if not _check_java():
        print(
            "[injury_scraper] Java not found in PATH. "
            "Skipping nbainjuries scraper.\n"
            "   Install Java 8+ to enable real-time PDF-based injury reports.\n"
            "   Falling back to nba_api injury data from Prompt 00."
        )
        return

    # Step 2: Fetch
    print("[injury_scraper] Fetching official NBA injury report (nbainjuries)...")
    records = fetch_todays_injury_report()

    if not records:
        print(
            "[injury_scraper] No injury records returned. "
            "This may be an off-day or the report is not yet available.\n"
            "   Retaining injury data from Prompt 00."
        )
        return

    # Step 3: Save raw output
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_INJURY_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[injury_scraper] Saved {len(records)} raw records → {RAW_INJURY_PATH}")

    # Step 4: Build map and merge
    injury_map = build_injury_map(records)
    print_injury_summary(injury_map)

    n_updated, n_newly_out = merge_into_today_roster(injury_map)

    # Step 5: Refresh exclusion list
    total_out = refresh_out_player_ids()

    # Step 6: Summary
    print(
        f"[injury_scraper] Complete.\n"
        f"   {len(injury_map)} players on injury report\n"
        f"   {n_updated} status updates merged into today_roster.json\n"
        f"   {n_newly_out} players newly discovered Out\n"
        f"   {total_out} total players excluded from pipeline (out_player_ids.txt)\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_injury_scraper()
