"""Smoke check: verify library versions and that landmark_adapter works on a black frame.

Uses landmark_adapter.process_frame (the shared adapter used by main.py and
extract.py) instead of importing mp.solutions directly, so this test exercises
the same code path as production.

Run: python scripts/smoke_check.py
Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    import cv2
    import mediapipe as mp
    import numpy as np
    import pygame

    from landmark_adapter import process_frame  # noqa: PLC0415

    print(f"Python   : {sys.version}")
    print(f"numpy    : {np.__version__}")
    print(f"cv2      : {cv2.__version__}")
    print(f"mediapipe: {mp.__version__}")
    print(f"pygame   : {pygame.__version__}")
    print()

    # Process a black 480×640 BGR frame through the shared adapter.
    # Expects no face detected and no exception raised.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print("Calling landmark_adapter.process_frame on black 640×480 frame...", end=" ")
    lm, frame_w, frame_h = process_frame(frame)
    assert lm is None, "Expected no face on black frame, got landmarks"
    assert frame_w == 640 and frame_h == 480, f"Unexpected frame size: {frame_w}×{frame_h}"
    print(f"OK (no face, {frame_w}×{frame_h})")

    print()
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
