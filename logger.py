"""Structured CSV event logger for the driver monitoring system.

Appends one row per alert event to ``logs/fatigue_log.csv`` (relative to the
project root, not the working directory).  A header row is written if the file
does not yet exist.

Columns
-------
iso_utc_time    ISO-8601 UTC timestamp of the alert event.
session_id      Identifier passed by the caller (default "unknown").
state           Alert state: DROWSY | LEFT | RIGHT | DOWN.
ear             Smoothed EAR value at the time of the alert (float, 4 d.p.).
ear_threshold   EAR threshold in use at the time of the alert (float, 4 d.p.).
fatigue_score   Fatigue score 0–100 at time of alert (float, 1 d.p.).
fps             Frames per second at time of alert (float, 1 d.p.).

Usage::

    from logger import EventLogger
    log = EventLogger(session_id="s01")
    log.log_event("DROWSY", ear=0.22, ear_threshold=0.26, fatigue=87.5, fps=29.1)
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

# Project root = directory containing this file.
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_FILE = _LOG_DIR / "fatigue_log.csv"

_HEADER = [
    "iso_utc_time",
    "session_id",
    "state",
    "ear",
    "ear_threshold",
    "fatigue_score",
    "fps",
]


class EventLogger:
    """Append-mode CSV event logger with a header row.

    Args:
        session_id: String identifier for this monitoring session.
        log_path:   Override the default log path (useful for tests).
    """

    def __init__(
        self,
        session_id: str = "unknown",
        log_path: Path | None = None,
    ) -> None:
        self._session_id = session_id
        self._path = log_path or _LOG_FILE
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Create the log directory and write the header row if the file is new."""
        os.makedirs(self._path.parent, exist_ok=True)
        if not self._path.exists():
            with open(self._path, "w", newline="") as f:
                csv.writer(f).writerow(_HEADER)

    def log_event(
        self,
        state: str,
        ear: float = float("nan"),
        ear_threshold: float = float("nan"),
        fatigue: float = float("nan"),
        fps: float = float("nan"),
    ) -> None:
        """Append one alert row to the log file.

        Args:
            state:          Alert state string (e.g. ``"DROWSY"``).
            ear:            Current smoothed EAR value (dimensionless).
            ear_threshold:  Current EAR alert threshold (dimensionless).
            fatigue:        Current fatigue score 0–100 (points).
            fps:            Current frame rate (frames per second).
        """
        row = [
            datetime.now(tz=timezone.utc).isoformat(),
            self._session_id,
            state,
            f"{ear:.4f}",
            f"{ear_threshold:.4f}",
            f"{fatigue:.1f}",
            f"{fps:.1f}",
        ]
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# Module-level convenience (backward-compatible with old log_event() callers)
# ---------------------------------------------------------------------------

_default_logger: EventLogger | None = None


def log_event(event: str) -> None:
    """Module-level log_event() kept for backward compatibility.

    Logs with ear/threshold/fatigue/fps as NaN (not yet threaded through).
    New code should create an ``EventLogger`` instance directly.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = EventLogger()
    _default_logger.log_event(event)
