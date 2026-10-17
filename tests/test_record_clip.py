"""Unit tests for scripts/record_clip.py — schedule builder and CSV writer.

Only pure functions are tested here (build_schedule, write_labels_csv).
No webcam, no cv2, no mediapipe.  All times are in seconds (never frames).
"""

from __future__ import annotations

import csv

# Add scripts/ to sys.path so we can import record_clip without installing.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from record_clip import (  # noqa: E402
    _EVENT_TYPES,
    NEUTRAL_S,
    REST_MAX_S,
    REST_MIN_S,
    CueEvent,
    build_schedule,
    write_labels_csv,
)

# ---------------------------------------------------------------------------
# build_schedule — pure function tests
# ---------------------------------------------------------------------------


class TestBuildSchedule:
    def test_returns_list_of_cue_events(self) -> None:
        events = build_schedule(seed=0, events_per_type=2)
        assert isinstance(events, list)
        assert all(isinstance(e, CueEvent) for e in events)

    def test_count(self) -> None:
        """Total events = len(_EVENT_TYPES) × events_per_type."""
        n = 3
        events = build_schedule(seed=0, events_per_type=n)
        assert len(events) == len(_EVENT_TYPES) * n

    def test_each_type_appears_events_per_type_times(self) -> None:
        n = 4
        events = build_schedule(seed=0, events_per_type=n)
        from collections import Counter
        counts = Counter((e.label, e.variant) for e in events)
        for label, variant, _ in _EVENT_TYPES:
            assert counts[(label, variant)] == n, (
                f"Expected {n} occurrences of ({label},{variant}), got {counts[(label,variant)]}"
            )

    def test_times_are_seconds_not_frames(self) -> None:
        """start_s and end_s must be floats (seconds), never frame counts."""
        events = build_schedule(seed=0, events_per_type=2)
        for ev in events:
            assert isinstance(ev.start_s, float), f"start_s is not float: {ev}"
            assert isinstance(ev.end_s, float), f"end_s is not float: {ev}"

    def test_non_overlapping_and_ordered(self) -> None:
        """Events must be strictly ordered and non-overlapping."""
        events = build_schedule(seed=0, events_per_type=4)
        for i in range(len(events) - 1):
            assert events[i].end_s < events[i + 1].start_s, (
                f"Events overlap or out of order: {events[i]} then {events[i+1]}"
            )

    def test_rest_periods_in_range(self) -> None:
        """Gaps between end_s[i] and start_s[i+1] must be in [REST_MIN_S, REST_MAX_S]."""
        events = build_schedule(seed=0, events_per_type=4)
        for i in range(len(events) - 1):
            gap = events[i + 1].start_s - events[i].end_s
            assert REST_MIN_S <= gap <= REST_MAX_S + 1e-9, (
                f"Rest gap {gap:.3f}s outside [{REST_MIN_S}, {REST_MAX_S}]"
            )

    def test_first_event_starts_at_zero(self) -> None:
        """t=0 is immediately after the calibration window."""
        events = build_schedule(seed=0, events_per_type=2)
        assert events[0].start_s == pytest.approx(0.0)

    def test_end_after_start(self) -> None:
        events = build_schedule(seed=0, events_per_type=2)
        for ev in events:
            assert ev.end_s > ev.start_s, f"end_s <= start_s: {ev}"

    def test_seeded_reproducible(self) -> None:
        """Same seed must produce identical schedules."""
        a = build_schedule(seed=99, events_per_type=3)
        b = build_schedule(seed=99, events_per_type=3)
        assert [(e.label, e.variant, e.start_s, e.end_s) for e in a] == \
               [(e.label, e.variant, e.start_s, e.end_s) for e in b]

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce different orderings (statistically)."""
        a = build_schedule(seed=1, events_per_type=8)
        b = build_schedule(seed=2, events_per_type=8)
        assert [(e.label, e.variant) for e in a] != [(e.label, e.variant) for e in b], (
            "Seeds 1 and 2 produced the same order — very unlikely, check RNG"
        )

    def test_labels_are_known_values(self) -> None:
        valid_labels = {"DROWSY", "LEFT", "RIGHT", "DOWN"}
        events = build_schedule(seed=0, events_per_type=2)
        for ev in events:
            assert ev.label in valid_labels, f"Unknown label: {ev.label}"

    def test_variants_are_known_values(self) -> None:
        valid_variants = {"N/A", "HEAD", "IRIS"}
        events = build_schedule(seed=0, events_per_type=2)
        for ev in events:
            assert ev.variant in valid_variants, f"Unknown variant: {ev.variant}"

    def test_default_events_per_type_is_8(self) -> None:
        events = build_schedule(seed=0)
        assert len(events) == len(_EVENT_TYPES) * 8

    def test_events_per_type_1(self) -> None:
        events = build_schedule(seed=0, events_per_type=1)
        assert len(events) == len(_EVENT_TYPES)


# ---------------------------------------------------------------------------
# write_labels_csv
# ---------------------------------------------------------------------------


class TestWriteLabelsCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        events = build_schedule(seed=0, events_per_type=1)
        path = tmp_path / "labels.csv"
        write_labels_csv(path, "S01", "1", "test", events)
        assert path.exists()

    def test_header_row(self, tmp_path: Path) -> None:
        events = build_schedule(seed=0, events_per_type=1)
        path = tmp_path / "labels.csv"
        write_labels_csv(path, "S01", "1", "test", events)
        with path.open() as fh:
            header = next(csv.reader(fh))
        assert header == ["subject_id", "session", "condition", "variant", "label", "start_s", "end_s"]

    def test_row_count(self, tmp_path: Path) -> None:
        n = 2
        events = build_schedule(seed=0, events_per_type=n)
        path = tmp_path / "labels.csv"
        write_labels_csv(path, "S01", "1", "test", events)
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(_EVENT_TYPES) * n

    def test_times_are_seconds_strings(self, tmp_path: Path) -> None:
        """start_s and end_s must be numeric strings, not frame counts."""
        events = build_schedule(seed=0, events_per_type=1)
        path = tmp_path / "labels.csv"
        write_labels_csv(path, "S01", "1", "test", events)
        with path.open() as fh:
            for row in csv.DictReader(fh):
                float(row["start_s"])  # must parse as float
                float(row["end_s"])

    def test_subject_session_condition_written(self, tmp_path: Path) -> None:
        events = build_schedule(seed=0, events_per_type=1)
        path = tmp_path / "labels.csv"
        write_labels_csv(path, "S99", "3", "lowlight", events)
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert all(r["subject_id"] == "S99" for r in rows)
        assert all(r["session"] == "3" for r in rows)
        assert all(r["condition"] == "lowlight" for r in rows)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        events = build_schedule(seed=0, events_per_type=1)
        path = tmp_path / "subdir" / "labels.csv"
        write_labels_csv(path, "S01", "1", "test", events)
        assert path.exists()


# ---------------------------------------------------------------------------
# NEUTRAL_S sanity
# ---------------------------------------------------------------------------


def test_neutral_s_is_10() -> None:
    """Calibration window must be 10 s as required by the spec."""
    assert NEUTRAL_S == pytest.approx(10.0)
