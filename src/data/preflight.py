"""
Pre-flight roster and injury status check.
Runs before the main data pipeline.

Outputs (all gitignored):
  data/local/today_roster.json     - Active players for today's games
  data/local/injury_report.json    - Full league injury report
  data/local/out_player_ids.txt    - Newline-separated player IDs to SKIP in pipeline
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from nba_api.stats.endpoints import commonteamroster

try:
    from nba_api.stats.endpoints import scoreboard as scoreboard_endpoint
except ImportError:
    try:
        from nba_api.stats.endpoints import scoreboardv2 as scoreboard_endpoint
    except ImportError:
        scoreboard_endpoint = None

try:
    from nba_api.stats.endpoints import injuryreport
except ImportError:
    injuryreport = None

# Output paths
DATA_DIR = Path("data/local")
TODAY_ROSTER_PATH = DATA_DIR / "today_roster.json"
INJURY_REPORT_PATH = DATA_DIR / "injury_report.json"
OUT_PLAYERS_PATH = DATA_DIR / "out_player_ids.txt"

# Injury designation -> pipeline behavior
SKIP_DESIGNATIONS = {"Out", "Inactive"}  # Fully excluded from pipeline
WARN_DESIGNATIONS = {
    "Doubtful",
    "GTD",
    "Questionable",
    "Probable",
}  # Included but flagged

PERFORMANCE_MODIFIERS = {
    "Active": 1.00,
    "Probable": 0.97,
    "Questionable": 0.88,
    "Doubtful": 0.65,
    "GTD": 0.80,
    "Out": 0.00,
    "Inactive": 0.00,
}


@dataclass
class PlayerRosterEntry:
    player_id: int
    player_name: str
    team_id: int
    team_abbreviation: str
    position: str
    jersey_number: str
    injury_status: str
    injury_description: str
    performance_modifier: float
    include_in_pipeline: bool


def _normalize_status(status: str) -> str:
    """Map non-standard injury status values to supported status labels."""
    if not status:
        return "Active"
    text = str(status).strip()
    lowered = text.lower()

    mapping = {
        "game time decision": "GTD",
        "gtd": "GTD",
        "questionable": "Questionable",
        "probable": "Probable",
        "doubtful": "Doubtful",
        "inactive": "Inactive",
        "out": "Out",
        "active": "Active",
    }
    return mapping.get(lowered, text)


def fetch_todays_games() -> list[dict]:
    """
    Returns list of today's games:
    [{"game_id": "...", "home_team_id": "...", "away_team_id": "...",
      "home_abbr": "LAL", "away_abbr": "GSW", "tip_off": "7:30 ET"}, ...]
    """
    if scoreboard_endpoint is None:
        print("[preflight] scoreboard endpoint not available in installed nba_api.")
        return []

    game_date = date.today().isoformat()
    if hasattr(scoreboard_endpoint, "Scoreboard"):
        board = scoreboard_endpoint.Scoreboard(game_date=game_date)
    elif hasattr(scoreboard_endpoint, "ScoreboardV2"):
        board = scoreboard_endpoint.ScoreboardV2(game_date=game_date, day_offset=0, league_id="00")
    else:
        print("[preflight] No supported scoreboard class found in installed nba_api.")
        return []

    time.sleep(1)

    games_df = board.game_header.get_data_frame()
    line_df = board.line_score.get_data_frame() if hasattr(board, "line_score") else None
    team_abbr_by_id: dict[str, str] = {}
    if line_df is not None and not line_df.empty:
        for _, row in line_df.iterrows():
            team_id = str(row.get("TEAM_ID", ""))
            abbr = str(row.get("TEAM_ABBREVIATION", ""))
            if team_id and abbr:
                team_abbr_by_id[team_id] = abbr

    games: list[dict] = []
    for _, row in games_df.iterrows():
        home_team_id = str(row["HOME_TEAM_ID"])
        away_team_id = str(row["VISITOR_TEAM_ID"])
        games.append(
            {
                "game_id": str(row["GAME_ID"]),
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_abbr": team_abbr_by_id.get(home_team_id, ""),
                "away_abbr": team_abbr_by_id.get(away_team_id, ""),
                "tip_off": str(row.get("GAME_STATUS_TEXT", "")),
            }
        )

    return games


def fetch_injury_report() -> dict[int, dict]:
    """
    Fetches the official NBA league-wide injury report.
    Returns: {player_id: {"status": "Out", "description": "..."}}

    Falls back to empty dict if endpoint unavailable (pre-game days, off-days).
    """
    if injuryreport is None:
        print("[preflight] injuryreport endpoint not available in installed nba_api. Defaulting all to Active.")
        return {}

    try:
        report = injuryreport.InjuryReport()
        time.sleep(1)
        df = report.get_data_frames()[0]

        result: dict[int, dict] = {}
        for _, row in df.iterrows():
            pid = int(row.get("Player_ID", 0))
            if not pid:
                continue
            status = _normalize_status(row.get("Status", "Active"))
            result[pid] = {
                "status": status,
                "description": str(row.get("Reason", "")),
            }
        return result
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[preflight] Injury report unavailable: {exc}. Defaulting all to Active.")
        return {}


def fetch_team_roster(team_id: str) -> list[dict]:
    """
    Returns list of players on a team roster with basic info.
    """
    roster = commonteamroster.CommonTeamRoster(team_id=team_id)
    time.sleep(1)
    df = roster.common_team_roster.get_data_frame()

    players: list[dict] = []
    for _, row in df.iterrows():
        players.append(
            {
                "player_id": int(row["PLAYER_ID"]),
                "player_name": str(row["PLAYER"]),
                "team_id": int(team_id),
                "position": str(row.get("POSITION", "")),
                "jersey": str(row.get("NUM", "")),
            }
        )
    return players


def build_today_roster(games: list[dict], injury_map: dict[int, dict]) -> list[PlayerRosterEntry]:
    """
    For each game today:
      1. Fetch both teams' rosters
      2. Cross-reference injury report
      3. Build PlayerRosterEntry list

    Returns all players for today's games, fully annotated.
    """
    team_abbr_by_id: dict[str, str] = {}
    for game in games:
        team_abbr_by_id[str(game["home_team_id"])] = str(game.get("home_abbr", ""))
        team_abbr_by_id[str(game["away_team_id"])] = str(game.get("away_abbr", ""))

    seen_team_ids: set[str] = set()
    entries: list[PlayerRosterEntry] = []

    for game in games:
        for team_id in [game["home_team_id"], game["away_team_id"]]:
            if team_id in seen_team_ids:
                continue
            seen_team_ids.add(team_id)

            roster_players = fetch_team_roster(team_id)
            for player in roster_players:
                pid = player["player_id"]
                injury_info = injury_map.get(pid, {})
                status = _normalize_status(injury_info.get("status", "Active"))
                description = str(injury_info.get("description", ""))
                modifier = PERFORMANCE_MODIFIERS.get(status, 1.0)
                include = status not in SKIP_DESIGNATIONS

                entries.append(
                    PlayerRosterEntry(
                        player_id=pid,
                        player_name=player["player_name"],
                        team_id=int(team_id),
                        team_abbreviation=team_abbr_by_id.get(str(team_id), ""),
                        position=player["position"],
                        jersey_number=player["jersey"],
                        injury_status=status,
                        injury_description=description,
                        performance_modifier=modifier,
                        include_in_pipeline=include,
                    )
                )

    return entries


def save_outputs(entries: list[PlayerRosterEntry], injury_map: dict, games: list[dict]) -> None:
    """
    Writes three local files (all gitignored):
      - today_roster.json        Full annotated roster
      - injury_report.json       Raw injury map
      - out_player_ids.txt       IDs to skip, one per line
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "game_date": date.today().isoformat(),
        "games": games,
        "players": [asdict(entry) for entry in entries],
    }
    TODAY_ROSTER_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[preflight] Saved today_roster.json  ({len(entries)} players)")

    INJURY_REPORT_PATH.write_text(json.dumps(injury_map, indent=2), encoding="utf-8")
    print(f"[preflight] Saved injury_report.json ({len(injury_map)} entries)")

    out_ids = [str(entry.player_id) for entry in entries if not entry.include_in_pipeline]
    OUT_PLAYERS_PATH.write_text("\n".join(out_ids), encoding="utf-8")
    print(
        f"[preflight] {len(out_ids)} players marked Out/Inactive "
        "- will be skipped in pipeline"
    )

    by_status: dict[str, int] = {}
    for entry in entries:
        by_status[entry.injury_status] = by_status.get(entry.injury_status, 0) + 1

    print("\n[preflight] Injury status summary for today's games:")
    for status, count in sorted(by_status.items()):
        modifier = PERFORMANCE_MODIFIERS.get(status, 1.0)
        flagged = " (warn)" if status in WARN_DESIGNATIONS else ""
        print(f"  {status:15s} {count:3d} players  (modifier: {modifier:.2f}){flagged}")


def load_out_player_ids() -> set[int]:
    """
    Utility function used by pipeline.py and predictor.py.
    Reads out_player_ids.txt -> returns set of player IDs to skip.
    Returns empty set if file doesn't exist (safe default).
    """
    if not OUT_PLAYERS_PATH.exists():
        return set()
    return {int(pid) for pid in OUT_PLAYERS_PATH.read_text(encoding="utf-8").splitlines() if pid.strip()}


def load_today_roster() -> dict:
    """
    Utility function used by predictor.py and UI.
    Returns the full today_roster.json payload as a dict.
    """
    if not TODAY_ROSTER_PATH.exists():
        raise FileNotFoundError(
            "today_roster.json not found. Run preflight.py first: "
            "python -m src.data.preflight"
        )
    return json.loads(TODAY_ROSTER_PATH.read_text(encoding="utf-8"))


def run_preflight() -> None:
    """
    Entry point. Call this before any other pipeline step.
    """
    print(f"[preflight] Running pre-flight check for {date.today().isoformat()}")

    print("[preflight] Fetching today's games...")
    games = fetch_todays_games()
    if not games:
        print("[preflight] No games scheduled today. Exiting.")
        return
    print(f"[preflight] {len(games)} game(s) today")

    print("[preflight] Fetching league injury report...")
    injury_map = fetch_injury_report()

    print("[preflight] Building today's roster with injury status...")
    entries = build_today_roster(games, injury_map)

    save_outputs(entries, injury_map, games)
    print("\n[preflight] Pre-flight complete. Pipeline may now run.\n")


if __name__ == "__main__":
    run_preflight()
