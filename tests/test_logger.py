"""Tests for EventLogger (logger.py).

Separated from test_state.py per A5-5d.
No cv2, mediapipe or pygame imported.
"""

from __future__ import annotations

from logger import EventLogger


class TestEventLogger:
    def test_header_written_on_creation(self, tmp_path) -> None:
        EventLogger(session_id="s01", log_path=tmp_path / "test.csv")
        content = (tmp_path / "test.csv").read_text()
        assert content.startswith("iso_utc_time"), "Header missing"

    def test_row_contains_all_columns(self, tmp_path) -> None:
        log = EventLogger(session_id="ses1", log_path=tmp_path / "test.csv")
        log.log_event("DROWSY", ear=0.22, ear_threshold=0.26, fatigue=87.5, fps=29.1)
        rows = (tmp_path / "test.csv").read_text().splitlines()
        assert len(rows) == 2, f"Expected header + 1 data row, got {len(rows)}"
        cols = rows[1].split(",")
        assert cols[1] == "ses1"
        assert cols[2] == "DROWSY"
        assert cols[3] == "0.2200"
        assert cols[6] == "29.1"

    def test_idempotent_header(self, tmp_path) -> None:
        """Opening the same file twice does not duplicate the header."""
        p = tmp_path / "test.csv"
        EventLogger(session_id="s1", log_path=p).log_event("LEFT")
        EventLogger(session_id="s2", log_path=p).log_event("RIGHT")
        rows = p.read_text().splitlines()
        assert rows[0].startswith("iso_utc_time"), "First row must be header"
        assert not rows[1].startswith("iso_utc_time"), "Second row must be data"
        assert len(rows) == 3, f"Expected header + 2 data rows, got {len(rows)}"

    def test_iso_utc_timestamp(self, tmp_path) -> None:
        """Timestamps must be ISO-8601 UTC (end with +00:00)."""
        log = EventLogger(session_id="s1", log_path=tmp_path / "test.csv")
        log.log_event("DOWN")
        rows = (tmp_path / "test.csv").read_text().splitlines()
        ts = rows[1].split(",")[0]
        assert ts.endswith("+00:00"), f"Timestamp {ts!r} is not UTC"

    def test_nan_placeholders_for_missing_values(self, tmp_path) -> None:
        """log_event() with no numeric args writes nan placeholders."""
        log = EventLogger(session_id="s1", log_path=tmp_path / "test.csv")
        log.log_event("LEFT")
        cols = (tmp_path / "test.csv").read_text().splitlines()[1].split(",")
        assert "nan" in cols[3].lower(), f"EAR column {cols[3]!r} expected nan"
