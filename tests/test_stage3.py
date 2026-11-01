"""Tests for Stage 3 extract+evaluate pipeline.

Tests use scripted landmark sequences (no video, no mediapipe).
Key invariant: alert times agree within one frame period at 10, 30, 60 FPS.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Repo root on sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evaluation"))


from config import CFG, Config  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers shared with test_monitor.py
# ---------------------------------------------------------------------------


def make_lm(n: int = 478) -> list[Any]:
    return [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(n)]


def set_lm(lm, i, x, y):
    lm[i] = SimpleNamespace(x=x, y=y, z=0.0)


def _open_eye_lm() -> list[Any]:
    lm = make_lm()
    set_lm(lm, 33,  0.300, 0.500)
    set_lm(lm, 133, 0.390, 0.500)
    set_lm(lm, 160, 0.315, 0.479)
    set_lm(lm, 144, 0.315, 0.521)
    set_lm(lm, 158, 0.375, 0.479)
    set_lm(lm, 153, 0.375, 0.521)
    set_lm(lm, 362, 0.610, 0.500)
    set_lm(lm, 263, 0.700, 0.500)
    set_lm(lm, 385, 0.615, 0.479)
    set_lm(lm, 380, 0.615, 0.521)
    set_lm(lm, 387, 0.675, 0.479)
    set_lm(lm, 373, 0.675, 0.521)
    for i in [469, 470, 471, 472]:
        set_lm(lm, i, 0.345, 0.500)
    for i in [474, 475, 476, 477]:
        set_lm(lm, i, 0.655, 0.500)
    set_lm(lm, 1, 0.5, 0.55)
    return lm


def _closed_eye_lm() -> list[Any]:
    lm = _open_eye_lm()
    for idx in [160, 144, 158, 153]:
        set_lm(lm, idx, lm[idx].x, 0.500)
    for idx in [385, 380, 387, 373]:
        set_lm(lm, idx, lm[idx].x, 0.500)
    return lm


def _make_monitor(cfg: Config | None = None):
    """Return a fresh DriverMonitor with NullAudio, using optional cfg override."""
    import config as _config_mod  # noqa: PLC0415
    from audio import NullAudio  # noqa: PLC0415
    from monitor import DriverMonitor  # noqa: PLC0415

    class _NullLog:
        def log_event(self, *a, **kw): pass

    orig = _config_mod.CFG
    if cfg is not None:
        _config_mod.CFG = cfg
    try:
        audio = NullAudio()
        mon = DriverMonitor(audio_backend=audio, event_logger=_NullLog())
        lm = _open_eye_lm()
        for i in range(CFG.calib_frames):
            mon.process_landmarks(lm, i / 30)
        # Force _pitch_neutral = 0 for deterministic pitch tests
        mon._pitch_neutral = 0.0
    finally:
        _config_mod.CFG = orig
    return mon, audio


def _run_fps(fps: float, n_calib: int, event_start_s: float, event_dur_s: float,
             lm_fn, cfg: Config | None = None):
    """Run monitor at given fps; inject event landmarks from event_start_s.

    Returns list of (time_s, state) for each alert_fired frame.
    """
    import config as _config_mod  # noqa: PLC0415
    from audio import NullAudio  # noqa: PLC0415
    from monitor import DriverMonitor  # noqa: PLC0415

    class _NullLog:
        def log_event(self, *a, **kw): pass

    orig = _config_mod.CFG
    if cfg is not None:
        _config_mod.CFG = cfg
    try:
        audio = NullAudio()
        mon = DriverMonitor(audio_backend=audio, event_logger=_NullLog())
        lm_open = _open_eye_lm()
        dt = 1.0 / fps
        # Calibration at fps
        for i in range(n_calib):
            mon.process_landmarks(lm_open, i * dt)
        mon._pitch_neutral = 0.0

        # Post-calibration
        total_s = event_start_s + event_dur_s + 3.0
        n_post = int(total_s * fps) + 1
        alerts = []
        for i in range(n_post):
            t = n_calib * dt + i * dt
            t_rel = t - n_calib * dt
            lm = lm_fn() if event_start_s <= t_rel < event_start_s + event_dur_s else lm_open
            result = mon.process_landmarks(lm, t)
            if result.alert_fired:
                alerts.append((t, result.state))
    finally:
        _config_mod.CFG = orig
    return alerts


# ---------------------------------------------------------------------------
# Tests: alert timing agrees within one frame period across 10/30/60 FPS
# ---------------------------------------------------------------------------


class TestAlertTimingFpsInvariance:
    """Alert time for a scripted DROWSY event must be within 1 frame period
    of the 30 FPS reference at 10 and 60 FPS.

    We measure DELAY = alert_time - event_start_time (relative to post-calib),
    not absolute timestamp, so calibration length differences don't matter.
    """

    # calib window length in seconds at each fps
    def _calib_end_s(self, fps: float) -> float:
        return CFG.calib_frames / fps

    def _first_drowsy_delay(self, fps: float) -> float | None:
        """Return delay = first_alert_time - (calib_end + event_start_s)."""
        n_calib = CFG.calib_frames
        event_start_s = 0.5
        alerts = _run_fps(fps, n_calib, event_start_s, event_dur_s=4.0, lm_fn=_closed_eye_lm)
        calib_end = n_calib / fps
        drowsy = [t - calib_end - event_start_s for t, s in alerts if s == "DROWSY"]
        return drowsy[0] if drowsy else None

    def test_drowsy_alert_fires_at_all_fps(self) -> None:
        for fps in (10.0, 30.0, 60.0):
            d = self._first_drowsy_delay(fps)
            assert d is not None, f"No DROWSY alert at {fps} FPS"

    def test_drowsy_alert_within_one_frame_period(self) -> None:
        """Detection delay at 10, 30, 60 FPS must agree within 1 frame of the slowest FPS."""
        delays = {}
        for fps in (10.0, 30.0, 60.0):
            d = self._first_drowsy_delay(fps)
            assert d is not None, f"No DROWSY alert at {fps} FPS"
            delays[fps] = d
        # 1 frame period at the slowest FPS tested (10 FPS → 0.1 s)
        tolerance = 1.0 / 10.0
        ref = delays[30.0]
        for fps, d in delays.items():
            assert abs(d - ref) <= tolerance + 1e-9, (
                f"Detection delay differs by >{tolerance:.3f}s between fps={fps} ({d:.4f}s) "
                f"and fps=30 ({ref:.4f}s)"
            )


# ---------------------------------------------------------------------------
# Ablation flag tests
# ---------------------------------------------------------------------------


class TestAblationFlags:
    """Ablation flag tests patch both config.CFG and monitor.CFG (local binding)."""

    def _patched_monitor(self, cfg, tmp_path):
        """Return (mon, audio) with cfg patched in both config and monitor modules."""
        import config as _config_mod  # noqa: PLC0415
        import monitor as _mon_mod  # noqa: PLC0415
        from audio import NullAudio  # noqa: PLC0415
        from logger import EventLogger  # noqa: PLC0415
        from monitor import DriverMonitor  # noqa: PLC0415

        _config_mod.CFG = cfg
        _mon_mod.CFG = cfg
        audio = NullAudio()
        log = EventLogger(log_path=tmp_path / "log.csv")
        return DriverMonitor(audio_backend=audio, event_logger=log), audio

    def test_ema_disabled_raw_ear_used(self, tmp_path: Path) -> None:
        """With ema_enabled=False, EAR EMA equals raw EAR each frame."""
        cfg_no_ema = dataclasses.replace(CFG, ema_enabled=False)
        import config as _config_mod  # noqa: PLC0415
        import monitor as _mon_mod  # noqa: PLC0415
        orig_cfg = _config_mod.CFG
        orig_mon = _mon_mod.CFG
        try:
            mon, _ = self._patched_monitor(cfg_no_ema, tmp_path)
            lm = _open_eye_lm()
            for i in range(cfg_no_ema.calib_frames):
                mon.process_landmarks(lm, i / 30)
            result = mon.process_landmarks(lm, cfg_no_ema.calib_frames / 30 + 0.1)
        finally:
            _config_mod.CFG = orig_cfg
            _mon_mod.CFG = orig_mon
        assert result.ema_ear == pytest.approx(result.ear, rel=1e-5), (
            f"Expected ema_ear == ear when EMA disabled; "
            f"ema_ear={result.ema_ear:.4f} ear={result.ear:.4f}"
        )

    def test_head_hysteresis_disabled(self, tmp_path: Path) -> None:
        """With head_hysteresis_enabled=False, exit threshold == enter threshold."""
        cfg_no_hyst = dataclasses.replace(CFG, head_hysteresis_enabled=False)
        import config as _config_mod  # noqa: PLC0415
        import detection as _det  # noqa: PLC0415
        import monitor as _mon_mod  # noqa: PLC0415
        orig_cfg = _config_mod.CFG
        orig_mon_cfg = _mon_mod.CFG
        orig_ghp = _det.get_head_pose
        try:
            _config_mod.CFG = cfg_no_hyst
            _mon_mod.CFG = cfg_no_hyst
            _det.get_head_pose = lambda lm, fw, fh: (20.0, 0.0)
            _mon_mod.get_head_pose = _det.get_head_pose
            mon, _ = self._patched_monitor(cfg_no_hyst, tmp_path)
            lm = _open_eye_lm()
            for i in range(cfg_no_hyst.calib_frames):
                mon.process_landmarks(lm, i / 30)
            mon._pitch_neutral = 0.0
            # Enter DOWN at 20 deg
            for i in range(30):
                mon.process_landmarks(lm, cfg_no_hyst.calib_frames / 30 + i / 30)
            assert mon._head_down, "Expected head_down=True at 20 deg"
            # Drop to 12 deg — with hysteresis disabled (exit=enter=15) must leave DOWN
            _det.get_head_pose = lambda lm, fw, fh: (12.0, 0.0)
            _mon_mod.get_head_pose = _det.get_head_pose
            result = mon.process_landmarks(lm, cfg_no_hyst.calib_frames / 30 + 1.1)
        finally:
            _config_mod.CFG = orig_cfg
            _mon_mod.CFG = orig_mon_cfg
            _det.get_head_pose = orig_ghp
            _mon_mod.get_head_pose = orig_ghp
        assert not result.head_down, (
            "Expected head_down=False at 12 deg with hysteresis disabled "
            "(exit threshold = enter threshold = 15 deg)"
        )

    def test_ear_fixed_threshold(self, tmp_path: Path) -> None:
        """With ear_fixed_threshold set, calibrated threshold is overridden."""
        fixed = 0.25
        cfg_fixed = dataclasses.replace(CFG, ear_fixed_threshold=fixed)
        import config as _config_mod  # noqa: PLC0415
        import monitor as _mon_mod  # noqa: PLC0415
        orig_cfg = _config_mod.CFG
        orig_mon = _mon_mod.CFG
        try:
            mon, _ = self._patched_monitor(cfg_fixed, tmp_path)
            lm = _open_eye_lm()
            for i in range(cfg_fixed.calib_frames):
                mon.process_landmarks(lm, i / 30)
            result = mon.process_landmarks(lm, cfg_fixed.calib_frames / 30 + 0.1)
        finally:
            _config_mod.CFG = orig_cfg
            _mon_mod.CFG = orig_mon
        assert result.threshold == pytest.approx(fixed), (
            f"Expected fixed threshold {fixed}, got {result.threshold:.4f}"
        )


# ---------------------------------------------------------------------------
# landmark_adapter — array round-trip
# ---------------------------------------------------------------------------


class TestLandmarkAdapter:
    def test_round_trip_preserves_values(self) -> None:
        """landmarks_to_array → array_to_landmarks preserves x, y, z."""
        from landmark_adapter import array_to_landmarks, landmarks_to_array  # noqa: PLC0415
        lm = _open_eye_lm()
        arr = landmarks_to_array(lm)
        lm2 = array_to_landmarks(arr)
        assert arr.shape == (478, 3)
        assert arr.dtype.name == "float32"
        for i in [33, 133, 362, 263]:
            assert lm2[i].x == pytest.approx(lm[i].x, rel=1e-5)
            assert lm2[i].y == pytest.approx(lm[i].y, rel=1e-5)

    def test_nan_row_becomes_neutral(self) -> None:
        """NaN rows in array become (0.5, 0.5, 0.0) after array_to_landmarks."""
        import numpy as np  # noqa: PLC0415

        from landmark_adapter import N_LANDMARKS, array_to_landmarks  # noqa: PLC0415
        arr = np.full((N_LANDMARKS, 3), np.nan, dtype="float32")
        lm = array_to_landmarks(arr)
        assert lm[0].x == pytest.approx(0.5)
        assert lm[0].y == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Config override parsing (used by evaluate.py)
# ---------------------------------------------------------------------------


class TestConfigOverrides:
    def test_float_override(self) -> None:
        from scripts.evaluate import _apply_overrides  # type: ignore[import]  # noqa: PLC0415
        cfg2 = _apply_overrides(CFG, ["ear_factor=0.75"])
        assert cfg2.ear_factor == pytest.approx(0.75)

    def test_bool_override(self) -> None:
        from scripts.evaluate import _apply_overrides  # type: ignore[import]  # noqa: PLC0415
        cfg2 = _apply_overrides(CFG, ["ema_enabled=false"])
        assert cfg2.ema_enabled is False

    def test_none_override(self) -> None:
        from scripts.evaluate import _apply_overrides  # type: ignore[import]  # noqa: PLC0415
        # ear_fixed_threshold defaults None; set then clear
        cfg2 = _apply_overrides(CFG, ["ear_fixed_threshold=0.25"])
        assert cfg2.ear_fixed_threshold == pytest.approx(0.25)
        cfg3 = _apply_overrides(cfg2, ["ear_fixed_threshold=None"])
        assert cfg3.ear_fixed_threshold is None

    def test_unknown_key_exits(self) -> None:
        from scripts.evaluate import _apply_overrides  # type: ignore[import]  # noqa: PLC0415
        with pytest.raises(SystemExit):
            _apply_overrides(CFG, ["no_such_field=1.0"])
