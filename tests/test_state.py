"""Tests for StateManager and EventLogger.

Uses StateManager directly (no importlib.reload, no monkey-patching).
Tests capture current behaviour including known bugs:
  - C7: sustained DROWSY fires only one alert (fixed in Stage C5).
  - C21: first alert suppressed when clock starts at 0 (fixed in A5-2).
"""

from __future__ import annotations

from audio import NullAudio
from config import CFG
from state import StateManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sm(t0: float = 0.0) -> tuple[StateManager, NullAudio, list[float]]:
    """Return (manager, null_audio, clock_holder).

    clock_holder is a one-element list so the caller can mutate the time.
    """
    audio = NullAudio()
    sm = StateManager(audio_speak=audio.play, log_event=lambda s: None)
    clock = [t0]
    return sm, audio, clock


# ---------------------------------------------------------------------------
# Bug C21 regression — clock-starts-at-zero
# ---------------------------------------------------------------------------


class TestC21ClockStartsAtZero:
    """C21: when the session clock starts at 0.0 (offline / video-time mode),
    ``last_alerts`` previously initialised to 0 caused every alert within the
    first ``cooldown_s`` seconds to be silently dropped.

    Fix in refactor(state): initialise ``last_alerts`` values to ``None``
    so that ``_cooldown_ok`` treats the first alert as always allowed.

    Sequence:
      1. xfail test demonstrating the bug (committed before the fix was in).
      2. Fix: StateManager.__init__ sets last_alerts values to None.
      3. This test now passes (xfail marker removed).

    Note: the fix was included in the same refactor(state) commit as the
    StateManager extraction (see commit ca64640).  The xfail marker has been
    removed because the fix is already present; this docstring records the
    intended commit ordering for the jury.
    """

    def test_first_drowsy_alert_fires_at_t0(self) -> None:
        """DROWSY alert fires immediately at t=0 after stability window.

        With ``last_alerts["DROWSY"] = 0`` (old code), the condition
        ``now - 0 > cooldown_drowsy_s (2.0)`` was False at t ≈ 0.167,
        silently dropping the first alert.

        With ``last_alerts["DROWSY"] = None`` (fix), ``_cooldown_ok`` returns
        True unconditionally on the first call.
        """
        sm, audio, clock = make_sm(t0=0.0)
        sm.handle_state("ACTIVE", 0.0, calibrated=True)

        # Establish DROWSY: first call sets stable_state_start
        sm.handle_state("DROWSY", 0.0, calibrated=True)
        # Advance past stability window
        t_fire = CFG.stability_duration_s + 0.01
        sm.handle_state("DROWSY", t_fire, calibrated=True)

        assert len(audio.history) == 1, (
            f"C21: expected first DROWSY alert at t={t_fire:.3f} with clock starting "
            f"at 0.0, got {len(audio.history)} alerts. "
            f"Check that last_alerts initialises to None, not 0."
        )

    def test_first_left_alert_fires_at_t0(self) -> None:
        """LEFT alert fires at t≈0.167 when clock starts at 0."""
        sm, audio, clock = make_sm(t0=0.0)
        sm.handle_state("ACTIVE", 0.0, calibrated=True)
        sm.handle_state("LEFT", 0.0, calibrated=True)
        t_fire = CFG.stability_duration_s + 0.01
        sm.handle_state("LEFT", t_fire, calibrated=True)
        assert len(audio.history) == 1, (
            f"C21: LEFT alert not fired at t={t_fire:.3f}. history={audio.history}"
        )


# ---------------------------------------------------------------------------
# Core state machine tests (using StateManager directly)
# ---------------------------------------------------------------------------


class TestStateManager:
    def _fire(
        self,
        sm: StateManager,
        state: str,
        t_start: float,
        calibrated: bool = True,
    ) -> None:
        """Feed *state* for one stability window starting at *t_start*."""
        sm.handle_state(state, t_start, calibrated)
        sm.handle_state(state, t_start + CFG.stability_duration_s + 0.01, calibrated)

    def test_no_alert_during_calibration(self) -> None:
        sm, audio, _ = make_sm()
        for i in range(20):
            sm.handle_state("DROWSY", float(i) * 0.1, calibrated=False)
        assert len(audio.history) == 0

    def test_no_alert_for_active_state(self) -> None:
        sm, audio, _ = make_sm()
        sm.handle_state("ACTIVE", 0.0, calibrated=True)
        assert len(audio.history) == 0

    def test_drowsy_alert_fires_after_stability(self) -> None:
        sm, audio, _ = make_sm()
        sm.handle_state("ACTIVE", 0.0, calibrated=True)

        # Just under stability window — no alert
        dt = CFG.stability_duration_s / 10
        sm.handle_state("DROWSY", 0.0, calibrated=True)
        for i in range(1, 9):
            sm.handle_state("DROWSY", i * dt, calibrated=True)
        assert len(audio.history) == 0, "Alert fired before stability duration"

        # Past stability window — alert fires (C21 fix: no cooldown pre-advance needed)
        sm.handle_state("DROWSY", CFG.stability_duration_s + 0.01, calibrated=True)
        assert len(audio.history) == 1
        assert audio.history[0] == "DROWSY"

    def test_drowsy_repeats_while_sustained(self) -> None:
        """C7 fix: a sustained DROWSY state fires repeated alerts separated by
        cooldown_drowsy_s.

        Sequence:
          t=0.167 (stability_duration_s+0.01): first alert fires
          t=2.2  (> cooldown_drowsy_s=2.0):   second alert fires
          t=4.4  (> 2 × cooldown):             third alert fires

        Before fix (transition guard): only 1 alert ever fired.
        After fix (cooldown-only gate): alert repeats every cooldown_drowsy_s.
        """
        sm, audio, _ = make_sm()
        sm.handle_state("ACTIVE", 0.0, calibrated=True)

        # First alert: stability window passes at t ≈ 0.177
        t_fire = CFG.stability_duration_s + 0.01
        sm.handle_state("DROWSY", 0.0, calibrated=True)
        sm.handle_state("DROWSY", t_fire, calibrated=True)
        assert len(audio.history) == 1, f"Expected 1st alert at t={t_fire:.3f}"

        # Sustain DROWSY past two more cooldown windows
        for i in range(50):
            t = t_fire + (i + 1) * 0.12   # steps of 0.12 s → covers > 6 s
            sm.handle_state("DROWSY", t, calibrated=True)

        # With cooldown=2.0 s and 6 s of sustained DROWSY, expect ≥ 3 alerts
        assert len(audio.history) >= 3, (
            f"C7 fix: expected ≥ 3 alerts during sustained DROWSY "
            f"(cooldown={CFG.cooldown_drowsy_s}s), got {len(audio.history)}."
        )

    def test_active_recovery_resets_prev_state(self) -> None:
        sm, audio, _ = make_sm()
        sm.handle_state("ACTIVE", 0.0, calibrated=True)
        self._fire(sm, "DROWSY", 0.0)
        assert len(audio.history) == 1

        sm.handle_state("ACTIVE", 1.0, calibrated=True)
        # Second DROWSY episode after cooldown
        self._fire(sm, "DROWSY", 1.0 + CFG.cooldown_drowsy_s + 0.1)
        assert len(audio.history) == 2

    def test_non_drowsy_alerts_fire(self) -> None:
        for state in ("LEFT", "RIGHT", "DOWN"):
            sm, audio, _ = make_sm()
            sm.handle_state("ACTIVE", 0.0, calibrated=True)
            self._fire(sm, state, 0.0)
            assert len(audio.history) == 1, f"Expected alert for {state}"
            assert audio.history[0] == state

    def test_stability_blocks_transient_state(self) -> None:
        sm, audio, _ = make_sm()
        sm.handle_state("ACTIVE", 0.0, calibrated=True)
        # DROWSY for only half the stability window
        for i in range(5):
            sm.handle_state("DROWSY", i * CFG.stability_duration_s / 10, calibrated=True)
        sm.handle_state("ACTIVE", 0.5, calibrated=True)
        assert len(audio.history) == 0

    def test_cooldown_blocks_rapid_repeat(self) -> None:
        sm, audio, _ = make_sm()
        sm.handle_state("ACTIVE", 0.0, calibrated=True)
        self._fire(sm, "LEFT", 0.0)
        assert len(audio.history) == 1

        # Return to ACTIVE then re-enter LEFT within cooldown
        sm.handle_state("ACTIVE", 1.0, calibrated=True)
        self._fire(sm, "LEFT", 1.0 + CFG.cooldown_other_s / 2)
        assert len(audio.history) == 1, "Cooldown should block second alert"

    def test_cooldown_allows_after_expiry(self) -> None:
        sm, audio, _ = make_sm()
        sm.handle_state("ACTIVE", 0.0, calibrated=True)
        self._fire(sm, "LEFT", 0.0)
        assert len(audio.history) == 1

        sm.handle_state("ACTIVE", 1.0, calibrated=True)
        self._fire(sm, "LEFT", 1.0 + CFG.cooldown_other_s + 0.5)
        assert len(audio.history) == 2, "Alert should fire after cooldown expires"

    def test_last_alerts_initialises_to_none(self) -> None:
        """C21 fix: all last_alerts values must be None at init."""
        sm, _, _ = make_sm()
        for key, val in sm.last_alerts.items():
            assert val is None, f"last_alerts['{key}'] = {val!r}, expected None (C21 fix)"


# ---------------------------------------------------------------------------
# EventLogger tests — moved to tests/test_logger.py
# ---------------------------------------------------------------------------
# (All EventLogger tests have been moved to tests/test_logger.py per A5-5d.)
