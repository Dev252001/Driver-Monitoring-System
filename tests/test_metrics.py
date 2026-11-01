"""Tests for evaluation/metrics.py.

All tests use hand-made event/alert lists with known TP/FN/FP/delay values.
No video, no mediapipe, no webcam.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))

from metrics import (  # noqa: E402
    ALERT_LABELS,
    Alert,
    FrameRecord,
    GroundTruthEvent,
    add_bootstrap_cis,
    event_metrics,
    false_alarm_rate,
    frame_confusion_matrix,
    neutral_seconds,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def ev(label: str, start: float, end: float) -> GroundTruthEvent:
    return GroundTruthEvent(label=label, start_s=start, end_s=end)


def al(label: str, t: float) -> Alert:
    return Alert(label=label, time_s=t)


# ---------------------------------------------------------------------------
# ALERT_LABELS constant
# ---------------------------------------------------------------------------


def test_alert_labels() -> None:
    assert ALERT_LABELS == {"DROWSY", "LEFT", "RIGHT", "DOWN"}


# ---------------------------------------------------------------------------
# event_metrics — TP / FN / FP
# ---------------------------------------------------------------------------


class TestEventMetricsBasic:
    def test_single_tp(self) -> None:
        """Alert inside window → TP, delay = alert_time - start_s."""
        events = [ev("DROWSY", 10.0, 13.0)]
        alerts = [al("DROWSY", 11.0)]   # 10 + 0.5 = 10.5 ≤ 11.0 ≤ 13 + 1.0 = 14.0 ✓
        m = event_metrics(events, alerts, "DROWSY")
        assert m.tp == 1
        assert m.fn == 0
        assert m.fp == 0
        assert m.delays == pytest.approx([1.0])   # 11.0 - 10.0

    def test_single_fn_alert_too_early(self) -> None:
        """Alert before margin_s → FN (alert arrives before driver reacted)."""
        events = [ev("DROWSY", 10.0, 13.0)]
        alerts = [al("DROWSY", 10.3)]   # < 10.0 + 0.5 = 10.5 → outside window
        m = event_metrics(events, alerts, "DROWSY")
        assert m.tp == 0
        assert m.fn == 1
        assert m.fp == 1   # alert matches no event → FP

    def test_single_fn_alert_too_late(self) -> None:
        """Alert after end + tol_s → FP and FN."""
        events = [ev("DROWSY", 10.0, 13.0)]
        alerts = [al("DROWSY", 14.5)]   # > 13.0 + 1.0 = 14.0 → outside window
        m = event_metrics(events, alerts, "DROWSY")
        assert m.tp == 0
        assert m.fn == 1
        assert m.fp == 1

    def test_no_alerts_no_events(self) -> None:
        m = event_metrics([], [], "DROWSY")
        assert m.tp == 0
        assert m.fn == 0
        assert m.fp == 0
        assert math.isnan(m.recall)
        assert math.isnan(m.precision)

    def test_no_events_alert_is_fp(self) -> None:
        alerts = [al("DROWSY", 5.0)]
        m = event_metrics([], alerts, "DROWSY")
        assert m.fp == 1
        assert m.fn == 0
        assert m.tp == 0

    def test_no_alerts_event_is_fn(self) -> None:
        events = [ev("DROWSY", 5.0, 8.0)]
        m = event_metrics(events, [], "DROWSY")
        assert m.fn == 1
        assert m.tp == 0
        assert m.fp == 0

    def test_alert_at_boundary_start_plus_margin(self) -> None:
        """Alert exactly at start_s + margin_s is a TP."""
        events = [ev("LEFT", 5.0, 8.0)]
        alerts = [al("LEFT", 5.5)]   # exactly 5.0 + 0.5
        m = event_metrics(events, alerts, "LEFT", margin_s=0.5, tol_s=1.0)
        assert m.tp == 1

    def test_alert_at_boundary_end_plus_tol(self) -> None:
        """Alert exactly at end_s + tol_s is a TP."""
        events = [ev("LEFT", 5.0, 8.0)]
        alerts = [al("LEFT", 9.0)]   # exactly 8.0 + 1.0
        m = event_metrics(events, alerts, "LEFT", margin_s=0.5, tol_s=1.0)
        assert m.tp == 1

    def test_multiple_events_all_tp(self) -> None:
        events = [ev("DOWN", 5.0, 7.5), ev("DOWN", 15.0, 17.5)]
        alerts = [al("DOWN", 6.0), al("DOWN", 16.0)]
        m = event_metrics(events, alerts, "DOWN")
        assert m.tp == 2
        assert m.fn == 0
        assert m.fp == 0

    def test_multiple_events_mixed(self) -> None:
        """2 events, 1 hit, 1 miss, 1 extra alert."""
        events = [ev("DROWSY", 5.0, 8.0), ev("DROWSY", 20.0, 23.0)]
        alerts = [al("DROWSY", 6.0), al("DROWSY", 30.0)]   # 2nd alert too late
        m = event_metrics(events, alerts, "DROWSY")
        assert m.tp == 1
        assert m.fn == 1
        assert m.fp == 1

    def test_each_alert_consumed_at_most_once(self) -> None:
        """Two overlapping events; one alert should match only the first."""
        events = [ev("DROWSY", 5.0, 8.0), ev("DROWSY", 7.0, 10.0)]
        alerts = [al("DROWSY", 6.0)]
        m = event_metrics(events, alerts, "DROWSY")
        assert m.tp == 1
        assert m.fn == 1   # second event has no alert

    def test_wrong_label_alert_not_counted(self) -> None:
        """Alert with wrong label must not match a DROWSY event."""
        events = [ev("DROWSY", 5.0, 8.0)]
        alerts = [al("LEFT", 6.0)]
        m = event_metrics(events, alerts, "DROWSY")
        assert m.tp == 0
        assert m.fn == 1
        assert m.fp == 0   # LEFT alert evaluated separately

    def test_recall_precision_f1_correct(self) -> None:
        """2 TP, 1 FN, 1 FP → recall=2/3, precision=2/3, f1=2/3."""
        events = [ev("DOWN", 5.0, 7.5), ev("DOWN", 15.0, 17.5), ev("DOWN", 25.0, 27.5)]
        alerts = [al("DOWN", 6.0), al("DOWN", 16.0), al("DOWN", 35.0)]
        m = event_metrics(events, alerts, "DOWN")
        assert m.tp == 2
        assert m.fn == 1
        assert m.fp == 1
        assert m.recall   == pytest.approx(2 / 3)
        assert m.precision == pytest.approx(2 / 3)
        assert m.f1       == pytest.approx(2 / 3)

    def test_mean_delay_correct(self) -> None:
        events = [ev("DROWSY", 10.0, 13.0), ev("DROWSY", 20.0, 23.0)]
        alerts = [al("DROWSY", 11.5), al("DROWSY", 21.0)]   # delays 1.5, 1.0
        m = event_metrics(events, alerts, "DROWSY")
        assert m.mean_delay_s == pytest.approx(1.25)

    def test_mean_delay_none_when_no_tp(self) -> None:
        events = [ev("DROWSY", 10.0, 13.0)]
        m = event_metrics(events, [], "DROWSY")
        assert m.mean_delay_s is None


# ---------------------------------------------------------------------------
# false_alarm_rate
# ---------------------------------------------------------------------------


class TestFalseAlarmRate:
    def test_nonzero_fp(self) -> None:
        """3 FP in 6 minutes → rate = 0.5 FP/min."""
        far = false_alarm_rate(fp_count=3, neutral_seconds=360.0)
        assert far.rate == pytest.approx(0.5)
        assert far.upper_bound_95 is None

    def test_zero_fp_gives_upper_bound(self) -> None:
        """0 FP in 2 minutes → upper_bound_95 = 3/2 = 1.5."""
        far = false_alarm_rate(fp_count=0, neutral_seconds=120.0)
        assert far.rate == pytest.approx(0.0)
        assert far.upper_bound_95 == pytest.approx(1.5)

    def test_zero_neutral_time(self) -> None:
        far = false_alarm_rate(fp_count=0, neutral_seconds=0.0)
        assert math.isnan(far.rate)
        assert far.upper_bound_95 is None


# ---------------------------------------------------------------------------
# frame_confusion_matrix
# ---------------------------------------------------------------------------


class TestFrameConfusionMatrix:
    def test_basic_counts(self) -> None:
        frames = [
            FrameRecord(time_s=5.0, gt_label="DROWSY", pred_label="DROWSY"),
            FrameRecord(time_s=5.1, gt_label="DROWSY", pred_label="ACTIVE"),
            FrameRecord(time_s=5.2, gt_label="ACTIVE", pred_label="ACTIVE"),
        ]
        events = [ev("DROWSY", 3.0, 6.0)]
        mat = frame_confusion_matrix(frames, events, boundary_s=0.0)
        assert mat[("DROWSY", "DROWSY")] == 1
        assert mat[("DROWSY", "ACTIVE")] == 1
        assert mat[("ACTIVE", "ACTIVE")] == 1

    def test_boundary_exclusion(self) -> None:
        """Frames within 0.5s of a boundary are excluded."""
        frames = [
            FrameRecord(time_s=9.7, gt_label="ACTIVE", pred_label="ACTIVE"),   # 0.3s before start→ excluded
            FrameRecord(time_s=10.6, gt_label="DROWSY", pred_label="DROWSY"),  # 0.6s after start → included
            FrameRecord(time_s=12.9, gt_label="DROWSY", pred_label="ACTIVE"),  # 0.1s before end → excluded
        ]
        events = [ev("DROWSY", 10.0, 13.0)]
        mat = frame_confusion_matrix(frames, events, boundary_s=0.5)
        assert mat.get(("ACTIVE", "ACTIVE"), 0) == 0       # excluded
        assert mat.get(("DROWSY", "DROWSY"), 0) == 1       # included
        assert mat.get(("DROWSY", "ACTIVE"), 0) == 0       # excluded

    def test_no_frames(self) -> None:
        mat = frame_confusion_matrix([], [], boundary_s=0.5)
        assert mat == {}


# ---------------------------------------------------------------------------
# neutral_seconds
# ---------------------------------------------------------------------------


class TestNeutralSeconds:
    def test_basic(self) -> None:
        """60s total, 1×3s event, 10s calib → 60 - 3 - 10 = 47s neutral."""
        events = [ev("DROWSY", 0.0, 3.0)]
        ns = neutral_seconds(total_duration_s=60.0, events=events, calibration_s=10.0)
        assert ns == pytest.approx(47.0)

    def test_no_events(self) -> None:
        ns = neutral_seconds(total_duration_s=60.0, events=[], calibration_s=10.0)
        assert ns == pytest.approx(50.0)

    def test_clamps_to_zero(self) -> None:
        """More event time than recording → clamp to 0."""
        events = [ev("DROWSY", 0.0, 100.0)]
        ns = neutral_seconds(total_duration_s=30.0, events=events, calibration_s=0.0)
        assert ns == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# add_bootstrap_cis
# ---------------------------------------------------------------------------


class TestBootstrapCIs:
    def test_ci_within_0_1(self) -> None:
        """Bootstrap recall CI must be in [0, 1]."""
        events = [ev("DROWSY", 10.0 + i * 8, 13.0 + i * 8) for i in range(10)]
        alerts = [al("DROWSY", 11.0 + i * 8) for i in range(8)]   # 8 TP, 2 FN
        m = event_metrics(events, alerts, "DROWSY")
        m2 = add_bootstrap_cis(m, events, alerts, n_boot=200, seed=42)
        lo, hi = m2.recall_ci
        assert 0.0 <= lo <= hi <= 1.0, f"Recall CI out of [0,1]: ({lo:.3f}, {hi:.3f})"

    def test_ci_ordered(self) -> None:
        """Low bound must be ≤ high bound."""
        events = [ev("LEFT", 5.0 + i * 10, 7.0 + i * 10) for i in range(6)]
        alerts = [al("LEFT", 6.0 + i * 10) for i in range(4)]
        m = event_metrics(events, alerts, "LEFT")
        m2 = add_bootstrap_cis(m, events, alerts, n_boot=200, seed=0)
        r_lo, r_hi = m2.recall_ci
        p_lo, p_hi = m2.precision_ci
        assert r_lo <= r_hi
        assert p_lo <= p_hi

    def test_seeded_reproducible(self) -> None:
        events = [ev("DOWN", 5.0 + i * 8, 7.5 + i * 8) for i in range(8)]
        alerts = [al("DOWN", 6.0 + i * 8) for i in range(6)]
        m = event_metrics(events, alerts, "DOWN")
        a = add_bootstrap_cis(m, events, alerts, n_boot=100, seed=7)
        b = add_bootstrap_cis(m, events, alerts, n_boot=100, seed=7)
        assert a.recall_ci == b.recall_ci

    def test_no_events_returns_unchanged(self) -> None:
        m = event_metrics([], [], "DROWSY")
        m2 = add_bootstrap_cis(m, [], [], n_boot=100, seed=0)
        assert m2.recall_ci is None
        assert m2.precision_ci is None
