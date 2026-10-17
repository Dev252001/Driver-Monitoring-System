"""Configuration for the AI-Based Smart Driver Monitoring System.

All tuneable parameters live here.  Every field is annotated with its unit.
The dataclass is frozen so that nothing accidentally mutates a field at runtime.

Usage::

    from config import CFG
    threshold = CFG.ear_factor * calibrated_mean

Phase 3 will tune gaze and head-pose thresholds from recorded data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ── Calibration ──────────────────────────────────────────────────────────
    calib_frames: int = 60
    """Number of frames collected to calibrate per-driver EAR baseline and
    neutral head pose.  Driver should look straight ahead during this phase."""

    # ── EAR (Eye Aspect Ratio) ───────────────────────────────────────────────
    tau_ear_s: float = 0.0653
    """EAR EMA time constant in seconds (C7 fix: FPS-independent smoothing).
    Per-frame alpha is computed as: alpha = 1 - exp(-dt / tau_ear_s).
    Calibrated to match old fixed alpha=0.4 at 30 FPS:
      tau = -(1/30) / ln(1-0.4) = 0.0653 s  →  alpha(30fps)=0.400 ✓
    Unit: seconds."""

    ear_factor: float = 0.80
    """EAR threshold multiplier applied to the calibrated mean.
    drowsy_trigger = calibrated_mean_ear * ear_factor * ear_blink_multiplier."""

    ear_blink_multiplier: float = 1.1
    """Additional slack on the EAR threshold used for blink detection and the
    drowsy hold gate (main.py line 90 comment: EAR_THRESHOLD * 1.1)."""

    ear_hold_s: float = 1.5
    """Seconds of continuous sub-threshold EAR before DROWSY state is entered
    (driver's eyes must stay closed for this duration).  Unit: seconds."""

    # ── Gaze ─────────────────────────────────────────────────────────────────
    tau_gaze_s: float = 0.0935
    """Gaze EMA time constant in seconds (C7 fix: FPS-independent smoothing).
    Per-frame alpha is computed as: alpha = 1 - exp(-dt / tau_gaze_s).
    Calibrated to match old fixed alpha=0.3 at 30 FPS:
      tau = -(1/30) / ln(1-0.3) = 0.0935 s  →  alpha(30fps)=0.300 ✓
    Unit: seconds."""

    gaze_left_threshold: float = 0.60
    """Smoothed gaze ratio above this value → LEFT distraction.
    NOTE: iris-corner pairing is WRONG in the current code (finding C2);
    these values were tuned to compensate and will be re-derived in Phase 2 C1.
    Unit: dimensionless ratio in [0, 1]."""

    gaze_right_threshold: float = 0.40
    """Smoothed gaze ratio below this value → RIGHT distraction.  See note on
    gaze_left_threshold."""

    gaze_hold_s: float = 1.4
    """Seconds of sustained off-centre gaze before LEFT/RIGHT alert fires.
    Unit: seconds."""

    gaze_switch_count: int = 2
    """Rapid gaze switches ≥ this count within gaze_switch_window_s activates
    turn mode, suppressing distraction alerts during shoulder-checks."""

    gaze_switch_window_s: float = 1.5
    """Time window for counting rapid gaze switches.  Unit: seconds."""

    turn_mode_exit_s: float = 2.0
    """Seconds after the last rapid switch before turn mode is exited.
    Unit: seconds."""

    # ── Head pose (normalised-y heuristic, replaced by solvePnP in Phase 2 C3) ─
    head_enter_delta: float = 0.08
    """Head-down detection threshold: nose-to-eye y-diff must exceed
    baseline + head_enter_delta to enter DOWN state.  Unit: normalised coords
    (0–1 range).  Scale-dependent — see finding C5."""

    head_exit_delta: float = 0.05
    """Hysteresis exit threshold: must drop below baseline + head_exit_delta to
    leave DOWN state.  Unit: normalised coords (0–1 range)."""

    # ── State machine ────────────────────────────────────────────────────────
    stability_duration_s: float = 0.167
    """Minimum duration a non-ACTIVE state must persist before triggering an
    alert.  Replaces the old 5-frame count (≈5/30 s ≈ 0.167 s at 30 FPS).
    Unit: seconds."""

    cooldown_drowsy_s: float = 2.0
    """Minimum gap between successive DROWSY audio alerts.  Unit: seconds."""

    cooldown_other_s: float = 4.0
    """Minimum gap between successive LEFT / RIGHT / DOWN audio alerts.
    Unit: seconds."""

    audio_min_gap_s: float = 1.5
    """Minimum gap between any two audio events (independent of state).
    Unit: seconds."""

    # ── Fatigue score ────────────────────────────────────────────────────────
    fatigue_decay_drowsy_per_s: float = 60.0
    """Fatigue score decrease per second while state == DROWSY.
    Derived from original −2/frame × 30 FPS = −60/s.  Unit: points/second."""

    fatigue_decay_alert_per_s: float = 30.0
    """Fatigue score decrease per second while state ∈ {LEFT, RIGHT, DOWN}.
    Derived from original −1/frame × 30 FPS = −30/s.  Unit: points/second."""

    fatigue_recover_per_s: float = 15.0
    """Fatigue score increase per second while state == ACTIVE.
    Derived from original +0.5/frame × 30 FPS = +15/s.  Unit: points/second."""

    fatigue_high_blink_decay_per_s: float = 6.0
    """Extra fatigue decay per second when blink_count > blink_high_threshold.
    Derived from original −0.2/frame × 30 FPS.  Unit: points/second."""

    blink_high_threshold: int = 20
    """Blink count in a 30 s window above which fatigue decay accelerates."""

    blink_reset_window_s: float = 30.0
    """Window duration for resetting the blink counter.  Unit: seconds."""


# Module-level singleton — import and use CFG everywhere.
CFG = Config()
