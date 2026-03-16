"""
Automatic injury-status sync watcher.

Monitors data/local/injury_report_nbainjuries.json for file-system changes.
Whenever the file is written (by the scraper, a manual edit, or any other
process), this watcher re-applies the new data to today_roster.json and
refreshes out_player_ids.txt automatically — no manual re-run required.

Usage:
    python -m src.data.injury_watcher

Run this in a separate terminal before (or alongside) any process that may
update injury_report_nbainjuries.json.  Press Ctrl+C to stop.

Requires:
    pip install watchdog
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

WATCH_FILE = Path("data/local/injury_report_nbainjuries.json")
DEBOUNCE_SECONDS = 2.0  # suppress duplicate events within this window


class _InjuryReportHandler:
    """watchdog event handler that triggers a roster sync on file writes."""

    def __init__(self) -> None:
        self._last_trigger: float = 0.0

    # watchdog calls dispatch() → we handle the three relevant event types
    def dispatch(self, event) -> None:  # type: ignore[override]
        event_type = getattr(event, "event_type", "")
        src = Path(getattr(event, "src_path", ""))
        if src.resolve() != WATCH_FILE.resolve():
            return
        if event_type not in ("modified", "created", "closed"):
            return

        now = time.monotonic()
        if now - self._last_trigger < DEBOUNCE_SECONDS:
            return
        self._last_trigger = now

        print(
            f"\n[injury_watcher] '{WATCH_FILE.name}' updated — syncing today_roster.json..."
        )
        logger.info("[injury_watcher] File change detected: %s", src)

        try:
            from src.data.injury_scraper import sync_from_saved_file

            n_updated, n_newly_out = sync_from_saved_file()
            print(
                f"[injury_watcher] Sync complete — "
                f"{n_updated} status change(s), {n_newly_out} newly Out.\n"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[injury_watcher] Sync failed: %s", exc, exc_info=True)
            print(f"[injury_watcher] ERROR during sync: {exc}")


def start_watcher() -> None:
    """Start the file-system observer and block until Ctrl+C."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print(
            "[injury_watcher] 'watchdog' package not found.\n"
            "   Install it with:  pip install watchdog\n"
            "   Then re-run:      python -m src.data.injury_watcher"
        )
        sys.exit(1)

    # Ensure the watch directory exists so the observer can attach
    watch_dir = WATCH_FILE.parent
    watch_dir.mkdir(parents=True, exist_ok=True)

    # Wrap our handler so watchdog can call it via its normal dispatch path
    handler_instance = _InjuryReportHandler()

    class _WatchdogAdapter(FileSystemEventHandler):
        def dispatch(self, event):  # type: ignore[override]
            handler_instance.dispatch(event)

    observer = Observer()
    observer.schedule(_WatchdogAdapter(), str(watch_dir.resolve()), recursive=False)
    observer.start()

    print(f"[injury_watcher] Watching  {WATCH_FILE}")
    print(f"[injury_watcher] Will auto-sync → {Path('data/local/today_roster.json')}")
    print("[injury_watcher] Press Ctrl+C to stop.\n")

    # Do an immediate sync on start-up so the files are consistent right away
    if WATCH_FILE.exists():
        print("[injury_watcher] Running initial sync from existing file...")
        try:
            from src.data.injury_scraper import sync_from_saved_file

            n_updated, n_newly_out = sync_from_saved_file()
            print(
                f"[injury_watcher] Initial sync complete — "
                f"{n_updated} status change(s), {n_newly_out} newly Out.\n"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[injury_watcher] Initial sync failed: {exc}")
    else:
        print(
            f"[injury_watcher] '{WATCH_FILE.name}' not found yet — "
            "will sync once it appears.\n"
        )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        print("\n[injury_watcher] Stopped.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    start_watcher()
