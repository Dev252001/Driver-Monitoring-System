"""DriverMonitor pipeline class for the AI-Based Smart Driver Monitoring System.

This module owns the entire per-frame detection pipeline.  Both the live loop
(main.py) and the offline evaluator (scripts/evaluate.py) use it via
``DriverMonitor.process_landmarks(lm, t, frame_w, frame_h) → FrameResult``.

No cv2 or mediapipe is imported at module level; the caller passes already-
decoded landmark objects so that the pipeline is fully unit-testable with
synthetic landmarks.

Time convention
---------------
``t`` is a monotonic timestamp in seconds, passed by the caller.
- Live:    caller passes ``time.monotonic()``.
- Offline: caller passes ``frame_index / fps``.
The pipeline never calls ``time.time()`` or ``time.monotonic()`` internally.

Bug notes
---------
- C7 (DROWSY fires once): StateManager blocks repeat alerts — fixed Stage C5.
- C7i (EMA FPS-dependence): fixed alpha, not time-constant — fixed Stage C7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import CFG
from detection import gaze_ratio, get_ear, get_head_pose
from state import StateManager


@dataclass
class FrameResult:
    """All per-frame outputs from DriverMonitor.process_landmarks().

    Fields
    ------
    ear:           Raw mean EAR (both eyes) for this frame (dimensionless).
    ema_ear:       Exponentially smoothed EAR (dimensionless).
    threshold:     Current EAR alert threshold (dimensionless).  0.0 until
                   calibration is complete.
    gaze:          Gaze direction: ``"LEFT"``, ``"RIGHT"``, or ``"CENTER"``.
    head_down:     True if the head-down heuristic fired (pitch > threshold).
    pitch_deg:     Head pitch in degrees from ``get_head_pose()``.
                   > 0 = head tilted DOWN.  0.0 until calibration is complete.
    yaw_deg:       Head yaw in degrees from ``get_head_pose()``.
                   > 0 = head turned RIGHT (driver's right).
    state:         Driver state: ``"CALIBRATING"``, ``"ACTIVE"``,
                   ``"DROWSY"``, ``"LEFT"``, ``"RIGHT"``, or ``"DOWN"``.
    fatigue_score: Fatigue score in [0, 100] (points).
    blink_count:   Blink count in the current ``blink_reset_window_s`` window.
    alert_fired:   True if an audio alert was fired for this frame.
    """

    ear: float
    ema_ear: float
    threshold: float
    gaze: str
    head_down: bool
    pitch_deg: float
    yaw_deg: float
    state: str
    fatigue_score: float
    blink_count: int
    alert_fired: bool


class DriverMonitor:
    """Full per-frame detection pipeline.

    Args:
        audio_backend: An object with a ``play(state: str)`` method.
                       Pass ``NullAudio()`` for headless / tests.
        event_logger:  An ``EventLogger`` instance for writing alert rows.
    """

    # Landmark index lists for EAR
    _RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    _LEFT_EYE  = [362, 385, 387, 263, 373, 380]

    def __init__(self, audio_backend: Any, event_logger: Any) -> None:
        # Inject both backends so that tests need no audio device or real log.
        self._state_manager = StateManager(
            audio_speak=audio_backend.play,
            log_event=self._log_alert,
        )
        self._logger = event_logger
        self._audio = audio_backend

        # EAR calibration
        self._ear_samples: list[float] = []
        self._ear_threshold: float = 0.0

        # Head-pose calibration (pitch neutral baseline from calib frames)
        self._pitch_neutral: float = 0.0
        self._pitch_samples: list[float] = []
        # Current head-pose
        self._pitch_deg: float = 0.0
        self._yaw_deg: float = 0.0

        # Per-frame smoothed values
        self._ema_ear: float = 0.25
        self._ema_gaze: float = 0.5

        # Drowsy hold gate
        self._drowsy_start: float | None = None

        # Blink counting
        self._blink_count: int = 0
        self._prev_eye_closed: bool = False
        self._last_blink_reset: float = 0.0

        # Gaze / turn-mode tracking
        self._gaze_start: float | None = None
        self._last_gaze: str = "CENTER"
        self._gaze_switch_count: int = 0
        self._last_switch_time: float = 0.0
        self._turn_mode: bool = False
        self._turn_start: float = 0.0

        # Head-down state (pitch-based, replaces normalised-y heuristic)
        self._head_down: bool = False

        # Fatigue score in [0, 100]
        self._fatigue: float = 100.0
        self._last_active_t: float = 0.0

        # Previous frame timestamp for dt computation (C7 EMA fix)
        self._prev_t: float | None = None

        # Track whether an alert was fired for the most recent frame
        self._alert_fired: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        """True once enough frames have been collected for calibration."""
        return len(self._ear_samples) >= CFG.calib_frames

    def process_landmarks(
        self,
        lm: list,
        t: float,
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> FrameResult:
        """Run the full per-frame detection pipeline on pre-decoded landmarks.

        Args:
            lm:      MediaPipe landmark list (normalised coords, length ≥ 478).
            t:       Monotonic timestamp in seconds from the caller.
            frame_w: Frame width in pixels (passed to get_head_pose; default 640).
            frame_h: Frame height in pixels (passed to get_head_pose; default 480).

        Returns:
            FrameResult with all per-frame outputs populated.
        """
        self._alert_fired = False

        # ── dt for FPS-independent EMA (C7) ─────────────────────────────
        # Clamp: first frame uses 1/30 s nominal; stalls capped at 0.5 s.
        if self._prev_t is None:
            dt = 1.0 / 30.0
        else:
            dt = min(t - self._prev_t, 0.5)
            dt = max(dt, 1e-6)   # guard against t not advancing (frozen clock tests)
        self._prev_t = t

        alpha_ear  = 1.0 - math.exp(-dt / CFG.tau_ear_s)
        alpha_gaze = 1.0 - math.exp(-dt / CFG.tau_gaze_s)

        # ── EAR ─────────────────────────────────────────────────────────
        ear = (
            get_ear(lm, self._RIGHT_EYE) + get_ear(lm, self._LEFT_EYE)
        ) / 2
        self._ema_ear = alpha_ear * ear + (1.0 - alpha_ear) * self._ema_ear

        # ── GAZE (EMA done here with time-based alpha) ───────────────────
        raw_gaze_ratio = gaze_ratio(lm)
        self._ema_gaze = alpha_gaze * raw_gaze_ratio + (1.0 - alpha_gaze) * self._ema_gaze
        if self._ema_gaze > CFG.gaze_left_threshold:
            gaze = "LEFT"
        elif self._ema_gaze < CFG.gaze_right_threshold:
            gaze = "RIGHT"
        else:
            gaze = "CENTER"

        # ── HEAD POSE (cv2.solvePnP) ─────────────────────────────────────
        self._pitch_deg, self._yaw_deg = get_head_pose(lm, frame_w, frame_h)

        # ── CALIBRATION ─────────────────────────────────────────────────
        if not self.calibrated:
            self._ear_samples.append(ear)
            self._pitch_samples.append(self._pitch_deg)

            if self.calibrated:
                # Compute thresholds on completion
                self._ear_threshold = (
                    float(np.mean(self._ear_samples)) * CFG.ear_factor
                )
                self._pitch_neutral = float(np.mean(self._pitch_samples))

            state = "CALIBRATING"

        else:
            # Determine head-down from pitch relative to neutral baseline
            pitch_delta = self._pitch_deg - self._pitch_neutral
            enter = CFG.head_enter_delta * (180.0 / 0.08)   # convert normalised to degrees
            exit_ = CFG.head_exit_delta  * (180.0 / 0.08)
            if self._head_down:
                self._head_down = pitch_delta > exit_
            else:
                self._head_down = pitch_delta > enter

            state = self._classify(t, ear, gaze)
            self._update_fatigue(state, t)
            self._state_manager.handle_state(state, t, calibrated=True)

        return FrameResult(
            ear=ear,
            ema_ear=self._ema_ear,
            threshold=self._ear_threshold,
            gaze=gaze,
            head_down=self._head_down,
            pitch_deg=self._pitch_deg,
            yaw_deg=self._yaw_deg,
            state=state,
            fatigue_score=self._fatigue,
            blink_count=self._blink_count,
            alert_fired=self._alert_fired,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_alert(self, state: str) -> None:
        """Called by StateManager when an alert fires; sets alert_fired flag."""
        self._alert_fired = True
        self._logger.log_event(
            state,
            ear=self._ema_ear,
            ear_threshold=self._ear_threshold,
            fatigue=self._fatigue,
        )

    def _classify(self, t: float, ear: float, gaze: str) -> str:
        """Classify the driver state for one post-calibration frame.

        Priority: DROWSY > DOWN > LEFT/RIGHT > ACTIVE.

        Args:
            t:    Injected monotonic time (seconds).
            ear:  Raw mean EAR for this frame.
            gaze: Gaze direction from get_gaze().

        Returns:
            State string.
        """
        # ── Blink / DROWSY ───────────────────────────────────────────────
        eye_closed = self._ema_ear < self._ear_threshold * CFG.ear_blink_multiplier

        if eye_closed:
            if not self._prev_eye_closed:
                self._blink_count += 1
                self._prev_eye_closed = True

            if self._drowsy_start is None:
                self._drowsy_start = t

            if t - self._drowsy_start > CFG.ear_hold_s:
                return "DROWSY"
            return "ACTIVE"

        else:
            self._drowsy_start = None
            self._prev_eye_closed = False

        # ── Blink counter reset ─────────────────────────────────────────
        if t - self._last_blink_reset > CFG.blink_reset_window_s:
            self._blink_count = 0
            self._last_blink_reset = t

        # ── HEAD DOWN ───────────────────────────────────────────────────
        if self._head_down:
            return "DOWN"

        # ── GAZE / turn-mode ────────────────────────────────────────────
        if gaze in ("LEFT", "RIGHT"):
            if self._gaze_start is None:
                self._gaze_start = t
                self._last_gaze = gaze
                self._gaze_switch_count = 0

            if gaze != self._last_gaze:
                self._gaze_switch_count += 1
                self._last_switch_time = t
                self._gaze_start = t

            rapid_switch = (
                self._gaze_switch_count >= CFG.gaze_switch_count
                and (t - self._last_switch_time < CFG.gaze_switch_window_s)
            )
            if rapid_switch:
                self._turn_mode = True
                self._turn_start = t

            if self._turn_mode:
                self._last_gaze = gaze
                return "ACTIVE"

            if t - self._gaze_start > CFG.gaze_hold_s:
                self._last_gaze = gaze
                return gaze

            self._last_gaze = gaze
            return "ACTIVE"

        else:
            self._gaze_start = None
            self._last_gaze = "CENTER"
            self._gaze_switch_count = 0

        # ── Turn-mode exit ──────────────────────────────────────────────
        if self._turn_mode and (t - self._turn_start > CFG.turn_mode_exit_s):
            self._turn_mode = False

        return "ACTIVE"

    def _update_fatigue(self, state: str, t: float) -> None:
        """Update fatigue score using per-second rates × dt.

        NOTE (C12): main.py still uses per-frame deltas.  This class uses the
        per-second rates from CFG, which reproduce the original 30 FPS
        behaviour and are frame-rate-accurate at any other rate.

        Args:
            state: Current driver state.
            t:     Injected monotonic time (seconds).
        """
        # dt ≈ 1/fps; clamp to avoid huge jumps on first frame or stalls.
        dt = min(t - self._last_active_t, 0.5) if self._last_active_t > 0 else 1 / 30

        if state == "DROWSY":
            self._fatigue -= CFG.fatigue_decay_drowsy_per_s * dt
        elif state in ("LEFT", "RIGHT", "DOWN"):
            self._fatigue -= CFG.fatigue_decay_alert_per_s * dt
        else:
            self._fatigue += CFG.fatigue_recover_per_s * dt

        if self._blink_count > CFG.blink_high_threshold:
            self._fatigue -= CFG.fatigue_high_blink_decay_per_s * dt

        self._fatigue = max(0.0, min(100.0, self._fatigue))
        self._last_active_t = t
