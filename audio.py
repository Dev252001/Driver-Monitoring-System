"""Priority audio alert player for the driver monitoring system.

Two backends are available:
- ``PygameAudio`` (default): initialises ``pygame.mixer`` lazily on first use
  so that importing this module never touches audio hardware.  Loads sound
  files in priority order; tries ``.ogg`` first, then ``.wav``, then ``.mp3``
  (with a warning, as MP3 is not guaranteed by pygame on all platforms).
- ``NullAudio``: silent drop-in used by unit tests and headless evaluation.

Usage::

    from audio import AudioPlayer
    player = AudioPlayer()          # real audio (live mode)
    player = AudioPlayer("null")    # silent (tests / headless)
    player.speak("DROWSY", t=time.monotonic())
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Callable
from typing import Protocol

# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class AudioBackend(Protocol):
    """Minimal interface that a sound backend must implement."""

    def play(self, state: str) -> None: ...
    def is_playing(self) -> bool: ...
    def stop(self) -> None: ...
    def current_state(self) -> str | None: ...


# ---------------------------------------------------------------------------
# NullAudio — silent backend for tests and headless evaluation
# ---------------------------------------------------------------------------

class NullAudio:
    """Silent audio backend.  Records the last spoken state for assertions."""

    def __init__(self) -> None:
        self._current: str | None = None
        self.history: list[str] = []

    def play(self, state: str) -> None:
        self._current = state
        self.history.append(state)

    def is_playing(self) -> bool:
        return False  # null backend never blocks

    def stop(self) -> None:
        self._current = None

    def current_state(self) -> str | None:
        return self._current


# ---------------------------------------------------------------------------
# PygameAudio — real audio backend with lazy initialisation
# ---------------------------------------------------------------------------

_SOUND_EXTS = [".ogg", ".wav", ".mp3"]

_PRIORITY: dict[str, int] = {
    "DROWSY": 3,
    "DOWN": 2,
    "LEFT": 1,
    "RIGHT": 1,
}


class PygameAudio:
    """pygame.mixer backend.  Initialised lazily on first ``play()`` call."""

    def __init__(self, sounds_dir: str | None = None) -> None:
        self._dir = sounds_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "sounds"
        )
        self._sounds: dict[str, object] = {}   # state → pygame.mixer.Sound
        self._channel: object = None           # pygame.mixer.Channel | None
        self._current: str | None = None
        self._ready = False

    # -- lazy init -----------------------------------------------------------

    def _ensure_init(self) -> None:
        if self._ready:
            return
        import pygame  # noqa: PLC0415 (intentional lazy import)

        pygame.mixer.init()
        for state in list(_PRIORITY):
            snd = self._load_sound(state)
            if snd is not None:
                self._sounds[state] = snd
        self._ready = True

    def _load_sound(self, state: str) -> object | None:
        import pygame  # noqa: PLC0415

        for ext in _SOUND_EXTS:
            path = os.path.join(self._dir, state.lower() + ext)
            if os.path.exists(path):
                if ext == ".mp3":
                    warnings.warn(
                        f"Loading {path} as MP3; pygame guarantees OGG/WAV only. "
                        "Convert to .ogg for reliable cross-platform support.",
                        stacklevel=2,
                    )
                return pygame.mixer.Sound(path)
        warnings.warn(f"No sound file found for state '{state}' in {self._dir}")
        return None

    # -- AudioBackend interface ----------------------------------------------

    def play(self, state: str) -> None:
        self._ensure_init()

        if state not in self._sounds:
            return

        # reset channel tracker if sound finished
        if self._channel is not None and not self._channel.get_busy():  # type: ignore[attr-defined]
            self._current = None

        # skip same state
        if state == self._current:
            return

        # priority check
        cur_priority = _PRIORITY.get(self._current or "", 0)
        new_priority = _PRIORITY.get(state, 0)

        if self._current is not None and new_priority < cur_priority:
            return

        # interrupt lower-priority sound if playing
        if self._channel is not None and self._channel.get_busy():  # type: ignore[attr-defined]
            if new_priority > cur_priority:
                self._channel.stop()  # type: ignore[attr-defined]
            else:
                return

        self._channel = self._sounds[state].play()  # type: ignore[union-attr]
        self._current = state

    def is_playing(self) -> bool:
        if self._channel is None:
            return False
        return bool(self._channel.get_busy())  # type: ignore[attr-defined]

    def stop(self) -> None:
        if self._channel is not None:
            self._channel.stop()  # type: ignore[attr-defined]
        self._current = None

    def current_state(self) -> str | None:
        if not self.is_playing():
            self._current = None
        return self._current


# ---------------------------------------------------------------------------
# AudioPlayer — thin wrapper that adds the MIN_AUDIO_GAP and DROWSY bypass
# ---------------------------------------------------------------------------

class AudioPlayer:
    """High-level alert player used by the state machine.

    Args:
        backend: ``"pygame"`` (default), ``"null"``, or an ``AudioBackend`` instance.
        clock:   Callable returning monotonic time in seconds.  Injectable for tests.
    """

    def __init__(
        self,
        backend: str | AudioBackend = "pygame",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if backend == "pygame":
            self._backend: AudioBackend = PygameAudio()
        elif backend == "null":
            self._backend = NullAudio()
        else:
            self._backend = backend  # type: ignore[assignment]
        self._clock = clock
        self._last_play_time: float = 0.0

    def speak(self, state: str, t: float | None = None) -> None:
        """Play the alert sound for *state* if priority and gap rules allow.

        Args:
            state: One of ``"DROWSY"``, ``"DOWN"``, ``"LEFT"``, ``"RIGHT"``.
            t:     Current monotonic time (seconds).  Uses ``self._clock()`` if None.
        """
        from config import CFG  # noqa: PLC0415

        now = t if t is not None else self._clock()

        # gap control (DROWSY bypasses the minimum gap)
        if state != "DROWSY" and (now - self._last_play_time < CFG.audio_min_gap_s):
            return

        self._backend.play(state)
        self._last_play_time = now


# ---------------------------------------------------------------------------
# Module-level convenience (backward-compatible with old speak() callers)
# ---------------------------------------------------------------------------

_default_player: AudioPlayer | None = None


def speak(state: str) -> None:
    """Module-level speak() kept for backward compatibility.

    New code should create an ``AudioPlayer`` instance directly.
    """
    global _default_player
    if _default_player is None:
        _default_player = AudioPlayer()
    _default_player.speak(state)
