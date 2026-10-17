"""Characterization tests for the StateManager / handle_state logic.

Uses a fake monotonic clock and NullAudio backend so no cv2/mediapipe/pygame
is imported.  Tests capture CURRENT behaviour (including known bugs C7, C8)
so that Stage C fixes show up as explicit test diffs.
"""

from __future__ import annotations

from audio import NullAudio
from config import CFG

# ---------------------------------------------------------------------------
# Helpers: fake clock and isolated state machine
# ---------------------------------------------------------------------------


class FakeClock:
    """Monotonically advancing fake clock for deterministic tests."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_state_manager() -> tuple:
    """Return (handle_state_fn, null_audio, fake_clock).

    Imports state module fresh each time to reset module-level globals.
    """
    # state.py uses module-level globals — we must reload per test.
    import importlib

    import state as state_mod  # noqa: PLC0415

    importlib.reload(state_mod)
    clock = FakeClock()
    null_audio = NullAudio()

    # Patch the module-level speak() and log_event() so no real audio/file I/O.
    state_mod.speak = null_audio.play  # type: ignore[attr-defined]

    state_mod.log_event = lambda event: None  # type: ignore[attr-defined]

    return state_mod.handle_state, null_audio, clock


# ---------------------------------------------------------------------------
# State machine characterization tests
# ---------------------------------------------------------------------------


class TestHandleStateCharacterization:
    def test_no_alert_during_calibration(self) -> None:
        """No alert should fire while calibrated=False."""
        handle_state, audio, clock = make_state_manager()
        for _ in range(20):
            handle_state("DROWSY", clock.t, calibrated=False)
            clock.advance(0.1)
        assert len(audio.history) == 0

    def test_no_alert_for_active_state(self) -> None:
        """ACTIVE state never fires an alert."""
        handle_state, audio, clock = make_state_manager()
        handle_state("ACTIVE", clock.t, calibrated=True)
        assert len(audio.history) == 0

    def test_drowsy_alert_fires_after_stability_duration(self) -> None:
        """DROWSY alert fires once state is stable for stability_duration_s AND
        the cooldown has elapsed.  last_alerts starts at 0, so we start the clock
        at CFG.cooldown_drowsy_s + 1 to ensure the cooldown is pre-satisfied."""
        handle_state, audio, clock = make_state_manager()
        # Start clock well past the initial cooldown so first alert is not blocked.
        clock.t = CFG.cooldown_drowsy_s + 1.0
        handle_state("ACTIVE", clock.t, calibrated=True)

        # Advance just short of stability_duration_s — no alert yet
        dt = CFG.stability_duration_s / 10
        for _ in range(9):
            handle_state("DROWSY", clock.t, calibrated=True)
            clock.advance(dt)
        assert len(audio.history) == 0, "Alert fired before stability duration"

        # One more step puts us past stability_duration_s
        handle_state("DROWSY", clock.t, calibrated=True)
        clock.advance(dt)
        handle_state("DROWSY", clock.t, calibrated=True)
        assert len(audio.history) == 1, f"Expected 1 alert, got {len(audio.history)}"
        assert audio.history[0] == "DROWSY"

    def test_bug_c7_drowsy_alert_fires_only_once_sustained(self) -> None:
        """CHARACTERIZATION (bug C7): a sustained DROWSY state fires only ONE
        alert because stable_state == prev_state causes early return on all
        subsequent frames.

        This test asserts the bug EXISTS.  It will need updating in Stage C5.
        """
        handle_state, audio, clock = make_state_manager()
        clock.t = CFG.cooldown_drowsy_s + 1.0
        handle_state("ACTIVE", clock.t, calibrated=True)

        # First DROWSY call: establishes stable_state=DROWSY and _stable_state_start
        handle_state("DROWSY", clock.t, calibrated=True)
        # Advance past stability window; next call fires the alert
        clock.advance(CFG.stability_duration_s + 0.01)
        handle_state("DROWSY", clock.t, calibrated=True)
        first_alert_count = len(audio.history)
        assert first_alert_count == 1, f"Expected 1 initial alert, got {first_alert_count}"

        # Now keep DROWSY for 3× the DROWSY cooldown — bug means no more alerts
        for _ in range(30):
            clock.advance(CFG.cooldown_drowsy_s / 10)
            handle_state("DROWSY", clock.t, calibrated=True)

        total = len(audio.history)
        # Bug assertion: still only 1 alert despite 3 cooldown windows elapsing
        assert total == 1, (
            f"Bug C7 may be fixed: got {total} alerts over sustained DROWSY. "
            "Expected 1 with the old code. Update this test for Stage C5."
        )

    def test_active_recovery_resets_prev_state(self) -> None:
        """After returning to ACTIVE, the next DROWSY triggers a new alert."""
        handle_state, audio, clock = make_state_manager()
        clock.t = CFG.cooldown_drowsy_s + 1.0
        handle_state("ACTIVE", clock.t, calibrated=True)

        # First DROWSY episode: establish then advance past stability
        handle_state("DROWSY", clock.t, calibrated=True)
        clock.advance(CFG.stability_duration_s + 0.01)
        handle_state("DROWSY", clock.t, calibrated=True)
        assert len(audio.history) == 1

        # Return to ACTIVE — resets prev_state
        handle_state("ACTIVE", clock.t, calibrated=True)

        # Second DROWSY episode after cooldown
        clock.advance(CFG.cooldown_drowsy_s + 0.01)
        handle_state("DROWSY", clock.t, calibrated=True)
        # Need another stability window to re-confirm DROWSY
        clock.advance(CFG.stability_duration_s + 0.01)
        handle_state("DROWSY", clock.t, calibrated=True)
        assert len(audio.history) == 2, (
            f"Expected 2 alerts after ACTIVE reset, got {len(audio.history)}"
        )

    def test_non_drowsy_alert_fires_after_stability(self) -> None:
        """LEFT/RIGHT/DOWN alert fires after stability_duration_s (and cooldown)."""
        for state in ("LEFT", "RIGHT", "DOWN"):
            handle_state, audio, clock = make_state_manager()
            clock.t = CFG.cooldown_other_s + 1.0
            handle_state("ACTIVE", clock.t, calibrated=True)

            # Establish the alert state, then advance past stability window
            handle_state(state, clock.t, calibrated=True)
            clock.advance(CFG.stability_duration_s + 0.01)
            handle_state(state, clock.t, calibrated=True)
            assert len(audio.history) == 1, (
                f"Expected 1 alert for {state}, got {len(audio.history)}"
            )
            assert audio.history[0] == state

    def test_stability_blocks_transient_state(self) -> None:
        """A state that flickers for less than stability_duration_s fires no alert."""
        handle_state, audio, clock = make_state_manager()
        handle_state("ACTIVE", clock.t, calibrated=True)

        # DROWSY for only half the stability window
        for _ in range(5):
            handle_state("DROWSY", clock.t, calibrated=True)
            clock.advance(CFG.stability_duration_s / 10)
        # Back to ACTIVE — stability counter resets
        handle_state("ACTIVE", clock.t, calibrated=True)

        assert len(audio.history) == 0, (
            f"Transient state should not fire alert; got {len(audio.history)} alerts"
        )

    def test_cooldown_blocks_rapid_repeat(self) -> None:
        """After an alert fires, the same state within cooldown fires nothing extra."""
        handle_state, audio, clock = make_state_manager()
        clock.t = CFG.cooldown_other_s + 1.0
        handle_state("ACTIVE", clock.t, calibrated=True)

        # Establish LEFT, advance past stability, fire alert
        handle_state("LEFT", clock.t, calibrated=True)
        clock.advance(CFG.stability_duration_s + 0.01)
        handle_state("LEFT", clock.t, calibrated=True)
        assert len(audio.history) == 1

        # Briefly go ACTIVE and come back — within cooldown
        handle_state("ACTIVE", clock.t, calibrated=True)
        clock.advance(CFG.cooldown_other_s / 2)  # half cooldown
        clock.advance(CFG.stability_duration_s + 0.01)
        handle_state("LEFT", clock.t, calibrated=True)

        # Still only 1 alert because cooldown hasn't passed
        assert len(audio.history) == 1, (
            f"Expected cooldown to block second alert, got {len(audio.history)}"
        )


# ---------------------------------------------------------------------------
# EventLogger characterization tests
# ---------------------------------------------------------------------------


class TestEventLogger:
    def test_header_written_on_creation(self, tmp_path) -> None:
        from logger import EventLogger  # noqa: PLC0415

        EventLogger(session_id="s01", log_path=tmp_path / "test.csv")
        content = (tmp_path / "test.csv").read_text()
        assert content.startswith("iso_utc_time"), "Header missing"

    def test_row_contains_all_columns(self, tmp_path) -> None:
        from logger import EventLogger  # noqa: PLC0415

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
        from logger import EventLogger  # noqa: PLC0415

        p = tmp_path / "test.csv"
        EventLogger(session_id="s1", log_path=p).log_event("LEFT")
        EventLogger(session_id="s2", log_path=p).log_event("RIGHT")
        rows = p.read_text().splitlines()
        assert rows[0].startswith("iso_utc_time"), "First row must be header"
        assert not rows[1].startswith("iso_utc_time"), "Second row must be data"
        assert len(rows) == 3, f"Expected header + 2 data rows, got {len(rows)}"
