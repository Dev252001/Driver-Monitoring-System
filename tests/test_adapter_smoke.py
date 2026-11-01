"""Smoke test for landmark_adapter.process_frame.

Requires mediapipe at runtime; skipped in CI where mediapipe is not installed.

    pytest.importorskip("mediapipe")  ← skips entire module if missing

The test processes a black BGR frame and asserts:
  - No face is detected (lm is None).
  - Frame dimensions are returned correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

mediapipe = pytest.importorskip("mediapipe")


def test_black_frame_no_face() -> None:
    """process_frame on a black 640×480 frame must return (None, 640, 480)."""
    from landmark_adapter import process_frame  # noqa: PLC0415

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lm, frame_w, frame_h = process_frame(frame)

    assert lm is None, (
        "Expected no face landmarks on a black frame, "
        f"but got a landmark list of length {len(lm) if lm is not None else 0}"
    )
    assert frame_w == 640, f"Expected frame_w=640, got {frame_w}"
    assert frame_h == 480, f"Expected frame_h=480, got {frame_h}"
