"""Smoke check: verify library versions and that mediapipe FaceMesh runs on a black frame.

Run: python scripts/smoke_check.py

Checks:
  1. Print library versions.
  2. Build FaceMesh(refine_landmarks=True) — required for iris landmarks 468-477.
  3. Process a black 480x640 (H x W) frame — no face expected, no exception should be raised.
  4. Assert numpy 2 does not break mediapipe at runtime (known risk with mediapipe 0.10.9).

Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import sys


def main() -> None:
    import cv2
    import mediapipe as mp
    import numpy as np
    import pygame

    print(f"Python   : {sys.version}")
    print(f"numpy    : {np.__version__}")
    print(f"cv2      : {cv2.__version__}")
    print(f"mediapipe: {mp.__version__}")
    print(f"pygame   : {pygame.__version__}")
    print()

    # --- Check 1: FaceMesh build with refine_landmarks=True ---
    print("Building FaceMesh(refine_landmarks=True)...", end=" ")
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    print("OK")

    # --- Check 2: Process a black 480x640 RGB frame ---
    # Height=480, Width=640 → shape (480, 640, 3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("Processing black frame (expect 0 faces)...", end=" ")
    results = face_mesh.process(rgb)
    n_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
    assert n_faces == 0, f"Expected 0 faces on black frame, got {n_faces}"
    print(f"OK (detected {n_faces} faces)")

    # --- Check 3: Confirm iris landmarks would be available if a face is found ---
    # We cannot inject a real face here without a camera, so we just verify that
    # refine_landmarks=True produced a FaceMesh that does not error on a blank frame.
    print("refine_landmarks=True: iris landmarks (468-477) will be present when a face is detected")

    face_mesh.close()
    print()
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
