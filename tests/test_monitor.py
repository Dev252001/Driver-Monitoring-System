"""Tests for DriverMonitor and FrameResult (monitor.py).

Tests are written test-first (Stage B1) and must FAIL before Stage B2 is
implemented.  After Stage B2 all tests here must pass.

Design constraints (from docs/PLAN.md architectural decisions):
  - monitor.py must be importable without cv2, mediapipe or pygame.
  - DriverMonitor.process_landmarks(lm, t) → FrameResult
  - FrameResult is a dataclass with fields:
      ear, ema_ear, threshold, gaze, head_down, state,
      fatigue_score, blink_count, alert_fired
  - Time injection: t is passed by caller; DriverMonitor never calls
    time.time() or time.monotonic() internally.
  - AudioPlayer("null") and EventLogger(log_path=tmp) used in tests.

No cv2, mediapipe or pygame may be imported at module collection time.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_lm(n: int = 478) -> list[Any]:
    """Return n landmarks all at (0.5, 0.5)."""
    return [SimpleNamespace(x=0.5, y=0.5) for _ in range(n)]


def set_lm(lm: list[Any], i: int, x: float, y: float) -> None:
    lm[i] = SimpleNamespace(x=x, y=y)


def _open_eye_lm() -> list[Any]:
    """Landmark list for an open eye at 640×480 normalised coords."""
    lm = make_lm()
    # Right eye horizontal corners and vertical lid landmarks
    set_lm(lm, 33,  0.300, 0.500)
    set_lm(lm, 133, 0.390, 0.500)
    set_lm(lm, 160, 0.315, 0.479)
    set_lm(lm, 144, 0.315, 0.521)
    set_lm(lm, 158, 0.375, 0.479)
    set_lm(lm, 153, 0.375, 0.521)
    # Left eye (mirror)
    set_lm(lm, 362, 0.610, 0.500)
    set_lm(lm, 263, 0.700, 0.500)
    set_lm(lm, 385, 0.615, 0.479)
    set_lm(lm, 380, 0.615, 0.521)
    set_lm(lm, 387, 0.675, 0.479)
    set_lm(lm, 373, 0.675, 0.521)
    # Centred irises
    for i in [469, 470, 471, 472]:
        set_lm(lm, i, 0.345, 0.500)
    for i in [474, 475, 476, 477]:
        set_lm(lm, i, 0.655, 0.500)
    # Nose tip (for head-pose heuristic)
    set_lm(lm, 1, 0.5, 0.55)
    return lm


def _closed_eye_lm() -> list[Any]:
    """Landmark list for closed eyes (vertical gap ≈ 0)."""
    lm = _open_eye_lm()
    for idx in [160, 144, 158, 153]:
        set_lm(lm, idx, lm[idx].x, 0.500)
    for idx in [385, 380, 387, 373]:
        set_lm(lm, idx, lm[idx].x, 0.500)
    return lm


def _make_monitor(tmp_path: Path):
    """Return a calibrated DriverMonitor with NullAudio and a temp log."""
    from audio import NullAudio  # noqa: PLC0415
    from config import CFG  # noqa: PLC0415
    from logger import EventLogger  # noqa: PLC0415
    from monitor import DriverMonitor  # noqa: PLC0415

    audio = NullAudio()
    log = EventLogger(log_path=tmp_path / "test_log.csv")
    mon = DriverMonitor(audio_backend=audio, event_logger=log)

    lm = _open_eye_lm()
    for i in range(CFG.calib_frames):
        t = i * (1 / 30)
        mon.process_landmarks(lm, t)

    return mon, audio, log


# ---------------------------------------------------------------------------
# Stage B1 / C3 — FrameResult and DriverMonitor tests
# ---------------------------------------------------------------------------


class TestFrameResult:
    """FrameResult must be a dataclass with the required fields (incl. C3 additions)."""

    def test_frameresult_is_dataclass(self) -> None:
        from monitor import FrameResult  # noqa: PLC0415

        assert dataclasses.is_dataclass(FrameResult)

    def test_frameresult_has_required_fields(self) -> None:
        """FrameResult must include pitch_deg and yaw_deg (added in Stage C3)."""
        from monitor import FrameResult  # noqa: PLC0415

        fields = {f.name for f in dataclasses.fields(FrameResult)}
        required = {
            "ear", "ema_ear", "threshold", "gaze", "head_down",
            "pitch_deg", "yaw_deg",
            "state", "fatigue_score", "blink_count", "alert_fired",
        }
        missing = required - fields
        assert not missing, f"FrameResult missing fields: {missing}"

    def test_frameresult_instantiates(self) -> None:
        from monitor import FrameResult  # noqa: PLC0415

        r = FrameResult(
            ear=0.30, ema_ear=0.28, threshold=0.22, gaze="CENTER",
            head_down=False, pitch_deg=0.0, yaw_deg=0.0,
            state="ACTIVE", fatigue_score=100.0,
            blink_count=0, alert_fired=False,
        )
        assert r.state == "ACTIVE"


class TestDriverMonitorInit:
    """DriverMonitor must accept injectable audio and logger backends."""

    def test_instantiates_without_cv2_or_mediapipe(self, tmp_path: Path) -> None:
        """Importing and instantiating DriverMonitor must not touch cv2/mediapipe."""
        from audio import NullAudio  # noqa: PLC0415
        from logger import EventLogger  # noqa: PLC0415
        from monitor import DriverMonitor  # noqa: PLC0415

        audio = NullAudio()
        log = EventLogger(log_path=tmp_path / "init_log.csv")
        mon = DriverMonitor(audio_backend=audio, event_logger=log)
        assert mon is not None

    def test_not_calibrated_at_init(self, tmp_path: Path) -> None:
        from audio import NullAudio  # noqa: PLC0415
        from logger import EventLogger  # noqa: PLC0415
        from monitor import DriverMonitor  # noqa: PLC0415

        audio = NullAudio()
        log = EventLogger(log_path=tmp_path / "log.csv")
        mon = DriverMonitor(audio_backend=audio, event_logger=log)
        assert not mon.calibrated


class TestProcessLandmarks:
    """DriverMonitor.process_landmarks(lm, t) runs the full detection pipeline."""

    def test_returns_frameresult(self, tmp_path: Path) -> None:
        from monitor import FrameResult  # noqa: PLC0415

        mon, _, _ = _make_monitor(tmp_path)
        lm = _open_eye_lm()
        result = mon.process_landmarks(lm, 5.0)
        assert isinstance(result, FrameResult)

    def test_calibrating_state_during_calib(self, tmp_path: Path) -> None:
        from audio import NullAudio  # noqa: PLC0415
        from logger import EventLogger  # noqa: PLC0415
        from monitor import DriverMonitor  # noqa: PLC0415

        audio = NullAudio()
        log = EventLogger(log_path=tmp_path / "log.csv")
        mon = DriverMonitor(audio_backend=audio, event_logger=log)

        lm = _open_eye_lm()
        result = mon.process_landmarks(lm, 0.0)
        assert result.state == "CALIBRATING"
        assert not mon.calibrated

    def test_active_after_calibration(self, tmp_path: Path) -> None:
        mon, _, _ = _make_monitor(tmp_path)
        lm = _open_eye_lm()
        result = mon.process_landmarks(lm, 10.0)
        assert result.state == "ACTIVE", (
            f"Expected ACTIVE after calibration, got {result.state}"
        )
        assert mon.calibrated

    def test_ear_and_ema_ear_populated(self, tmp_path: Path) -> None:
        mon, _, _ = _make_monitor(tmp_path)
        lm = _open_eye_lm()
        result = mon.process_landmarks(lm, 10.0)
        assert 0.0 < result.ear < 1.0, f"ear={result.ear} out of range"
        assert 0.0 < result.ema_ear < 1.0, f"ema_ear={result.ema_ear} out of range"

    def test_threshold_set_after_calibration(self, tmp_path: Path) -> None:
        mon, _, _ = _make_monitor(tmp_path)
        lm = _open_eye_lm()
        result = mon.process_landmarks(lm, 10.0)
        assert result.threshold > 0.0, (
            f"threshold={result.threshold} not set after calibration"
        )

    def test_drowsy_state_on_closed_eyes(self, tmp_path: Path) -> None:
        """Sustained closed eyes beyond ear_hold_s → DROWSY state."""
        from config import CFG  # noqa: PLC0415

        mon, _, _ = _make_monitor(tmp_path)
        lm = _closed_eye_lm()

        t = 10.0
        state = "ACTIVE"
        for i in range(60):
            t += 1 / 30
            result = mon.process_landmarks(lm, t)
            state = result.state

        assert state == "DROWSY", (
            f"Expected DROWSY after {60/30:.1f}s of closed eyes "
            f"(ear_hold_s={CFG.ear_hold_s}), got {state}"
        )

    def test_alert_fired_on_drowsy(self, tmp_path: Path) -> None:
        """alert_fired=True when DROWSY alert is triggered."""
        mon, _, _ = _make_monitor(tmp_path)
        lm = _closed_eye_lm()

        t = 10.0
        any_alert = False
        for i in range(80):
            t += 1 / 30
            result = mon.process_landmarks(lm, t)
            if result.alert_fired:
                any_alert = True
                break

        assert any_alert, "Expected at least one alert_fired=True during sustained DROWSY"

    def test_fatigue_decreases_on_drowsy(self, tmp_path: Path) -> None:
        """Fatigue score decreases when state is DROWSY."""
        mon, _, _ = _make_monitor(tmp_path)

        # Snapshot fatigue after calibration (should be near 100)
        initial = mon.process_landmarks(_open_eye_lm(), 10.0).fatigue_score

        lm = _closed_eye_lm()
        t = 10.1
        for i in range(60):
            t += 1 / 30
            result = mon.process_landmarks(lm, t)

        assert result.fatigue_score < initial, (
            f"Expected fatigue to decrease during DROWSY, "
            f"initial={initial:.1f}, final={result.fatigue_score:.1f}"
        )

    def test_blink_count_increments(self, tmp_path: Path) -> None:
        """A blink (open → closed → open) increments blink_count."""
        mon, _, _ = _make_monitor(tmp_path)
        open_lm = _open_eye_lm()
        closed_lm = _closed_eye_lm()

        t = 10.0
        mon.process_landmarks(open_lm, t)
        mon.process_landmarks(closed_lm, t + 0.05)
        result = mon.process_landmarks(open_lm, t + 0.10)

        assert result.blink_count >= 1, (
            f"Expected blink_count ≥ 1, got {result.blink_count}"
        )

    def test_no_alert_during_calibration(self, tmp_path: Path) -> None:
        from audio import NullAudio  # noqa: PLC0415
        from config import CFG  # noqa: PLC0415
        from logger import EventLogger  # noqa: PLC0415
        from monitor import DriverMonitor  # noqa: PLC0415

        audio = NullAudio()
        log = EventLogger(log_path=tmp_path / "log.csv")
        mon = DriverMonitor(audio_backend=audio, event_logger=log)
        lm = _closed_eye_lm()

        for i in range(CFG.calib_frames):
            mon.process_landmarks(lm, i / 30)

        assert len(audio.history) == 0, (
            f"Alerts fired during calibration: {audio.history}"
        )

    def test_time_injection_no_internal_clock(self, tmp_path: Path) -> None:
        """DriverMonitor must use the t argument, not time.time() internally.

        If t is constant (frozen clock), the drowsy hold timer should never
        advance, so DROWSY state should not be entered.
        """
        mon, _, _ = _make_monitor(tmp_path)
        lm = _closed_eye_lm()

        # Frozen clock: all frames at t=10.0 → ear_hold timer never advances
        for _ in range(100):
            result = mon.process_landmarks(lm, 10.0)

        assert result.state != "DROWSY", (
            "DriverMonitor is using an internal clock instead of the injected t. "
            f"Got state={result.state} with frozen clock."
        )

    def test_pitch_yaw_populated_after_calibration(self, tmp_path: Path) -> None:
        """pitch_deg and yaw_deg must be present in FrameResult after Stage C3."""
        mon, _, _ = _make_monitor(tmp_path)
        lm = _open_eye_lm()
        result = mon.process_landmarks(lm, 10.0)
        # Both fields must exist and be finite floats
        assert hasattr(result, "pitch_deg"), "FrameResult missing pitch_deg"
        assert hasattr(result, "yaw_deg"), "FrameResult missing yaw_deg"
        assert isinstance(result.pitch_deg, float), f"pitch_deg not float: {type(result.pitch_deg)}"
        assert isinstance(result.yaw_deg, float), f"yaw_deg not float: {type(result.yaw_deg)}"
        # Angles must be finite; exact value depends on the synthetic landmark geometry.
        # (The test helper _open_eye_lm() doesn't set all 6 solvePnP landmarks, so
        # the pose estimate is imprecise but must not be NaN or ±inf.)
        import math  # noqa: PLC0415
        assert math.isfinite(result.pitch_deg), f"pitch_deg={result.pitch_deg} is not finite"
        assert math.isfinite(result.yaw_deg),   f"yaw_deg={result.yaw_deg} is not finite"


