"""Alert state machine for the driver monitoring system.

The ``StateManager`` class encapsulates all mutable state (alert timers,
debounce counters, previous state) as instance attributes so that:

  - Each ``DriverMonitor`` instance owns its own ``StateManager``.
  - Tests create a fresh instance per test case — no ``importlib.reload``.
  - Offline evaluation with video-time clocks works without side effects.

The module-level ``handle_state()`` function is kept for backward
compatibility with the existing ``main.py``; it delegates to a lazily
created module-level instance.  New code should instantiate ``StateManager``
directly.

Time convention
---------------
All ``now`` parameters are monotonic seconds from the caller.  In live mode
the caller passes ``time.monotonic()``.  In offline evaluation the caller
passes ``frame_index / fps``.

Bug C7 (alert fires only once for sustained state) is characterised here
and will be fixed in Stage C5.
Bug C21 (first alert suppressed when clock starts at 0) is fixed here:
``last_alerts`` initialises to ``None`` so the first alert always fires.
"""

from __future__ import annotations

from collections.abc import Callable

from config import CFG


class StateManager:
    """Encapsulates all alert state machine state as instance attributes.

    Args:
        audio_speak:  Callable(state: str) -> None that plays the alert sound.
        log_event:    Callable(state: str) -> None that writes to the event log.
    """

    def __init__(
        self,
        audio_speak: Callable[[str], None],
        log_event: Callable[[str], None],
    ) -> None:
        self._speak = audio_speak
        self._log = log_event

        # Debounce / stability tracking
        self.stable_state: str = "ACTIVE"
        self.state_count: int = 0
        self._stable_state_start: float = 0.0

        # Transition tracking (C7: fixes after Stage C5)
        self.prev_state: str = "ACTIVE"

        # Per-state last-alert monotonic timestamps.
        # None means "never fired" — avoids the C21 bug where 0 suppresses
        # the first alert when the session clock starts at 0.0 (offline mode).
        self.last_alerts: dict[str, float | None] = {
            "DROWSY": None,
            "LEFT": None,
            "RIGHT": None,
            "DOWN": None,
        }

    def _cooldown_ok(self, state: str, now: float, cooldown_s: float) -> bool:
        """Return True if enough time has elapsed since the last alert for *state*."""
        last = self.last_alerts.get(state)
        return last is None or (now - last) > cooldown_s

    def handle_state(self, state: str, now: float, calibrated: bool) -> None:
        """Process one frame's detected driver state and fire alerts as needed.

        Args:
            state:      Detected state string: ``"ACTIVE"``, ``"CALIBRATING"``,
                        ``"DROWSY"``, ``"LEFT"``, ``"RIGHT"``, or ``"DOWN"``.
            now:        Monotonic timestamp in seconds (from caller).
            calibrated: Whether the per-driver calibration is complete.
        """
        if not calibrated or state in ("ACTIVE", "CALIBRATING"):
            # Reset so the next alert state is never silently skipped
            if state == "ACTIVE":
                self.prev_state = "ACTIVE"
            self.stable_state = state
            self.state_count = 0
            return

        # ── Debounce / stability ──────────────────────────────────────────
        if state == self.stable_state:
            self.state_count += 1
        else:
            self.stable_state = state
            self.state_count = 1
            self._stable_state_start = now

        # Wait for stability_duration_s before acting
        if now - self._stable_state_start < CFG.stability_duration_s:
            return

        # ── Transition guard (C7: still fires only on transition; fixed in C5) ──
        if self.stable_state == self.prev_state:
            return

        # ── DROWSY (highest priority, shorter cooldown) ───────────────────
        if self.stable_state == "DROWSY":
            if self._cooldown_ok("DROWSY", now, CFG.cooldown_drowsy_s):
                self._speak("DROWSY")
                self._log("DROWSY")
                self.last_alerts["DROWSY"] = now
            self.prev_state = "DROWSY"
            return

        # ── LEFT / RIGHT / DOWN ───────────────────────────────────────────
        if self._cooldown_ok(self.stable_state, now, CFG.cooldown_other_s):
            self._speak(self.stable_state)
            self._log(self.stable_state)
            self.last_alerts[self.stable_state] = now

        self.prev_state = self.stable_state


# ---------------------------------------------------------------------------
# Module-level compat shim (used by main.py until Stage B replaces it)
# ---------------------------------------------------------------------------

_manager: StateManager | None = None


def _get_manager() -> StateManager:
    global _manager
    if _manager is None:
        # Import here to avoid circular imports and import-time side effects.
        from audio import speak as _speak  # noqa: PLC0415
        from logger import log_event as _log  # noqa: PLC0415

        _manager = StateManager(audio_speak=_speak, log_event=_log)
    return _manager


def handle_state(state: str, now: float, calibrated: bool) -> None:
    """Module-level handle_state() kept for backward compatibility with main.py.

    New code should create a ``StateManager`` instance directly.
    """
    _get_manager().handle_state(state, now, calibrated)
