"""Shared landmark extraction adapter for live and offline modes.

Both ``main.py`` (live webcam) and ``scripts/extract.py`` (offline video)
use this module so that the frame → landmarks pipeline is not duplicated.

The mediapipe FaceMesh instance is created lazily on first call to
``process_frame`` so that importing this module never touches mediapipe at
import time (keeps headless CI clean).

Convention (matches detection.py and monitor.py):
  - Frame is pre-processed with ``cv2.flip(frame, 1)`` (mirror) so that
    image-left = driver-right.
  - Landmarks are MediaPipe normalised coords (x, y ∈ [0,1]).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

# 478 landmarks × 3 (x, y, z) — dtype used in npz export
LM_DTYPE = np.float32
N_LANDMARKS = 478


def _make_face_mesh():  # type: ignore[return]
    """Create and return a MediaPipe FaceMesh instance (lazy import)."""
    import mediapipe as mp  # noqa: PLC0415
    return mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)


_face_mesh = None   # module-level singleton, initialised on first use


def process_frame(frame) -> tuple[list[Any] | None, int, int]:
    """Run MediaPipe FaceMesh on one BGR frame and return landmarks.

    Applies brightness boost and horizontal flip before running FaceMesh
    (same pre-processing as main.py).

    Args:
        frame: BGR numpy array from cv2.VideoCapture.

    Returns:
        Tuple of (landmark_list | None, frame_w, frame_h).
        landmark_list is the first face's landmark list (length 478) or None
        if no face was detected.
    """
    import cv2  # noqa: PLC0415

    global _face_mesh  # noqa: PLW0603
    if _face_mesh is None:
        _face_mesh = _make_face_mesh()

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _face_mesh.process(rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark, frame_w, frame_h
    return None, frame_w, frame_h


def landmarks_to_array(lm: list[Any]) -> np.ndarray:
    """Convert a MediaPipe landmark list to a (N_LANDMARKS, 3) float32 array."""
    arr = np.empty((N_LANDMARKS, 3), dtype=LM_DTYPE)
    for i, p in enumerate(lm):
        arr[i, 0] = p.x
        arr[i, 1] = p.y
        arr[i, 2] = p.z
    return arr


def array_to_landmarks(arr: np.ndarray) -> list[Any]:
    """Convert a (N_LANDMARKS, 3) float32 array back to a SimpleNamespace list.

    NaN rows become (x=0.5, y=0.5, z=0.0) so that downstream code never
    receives NaN coordinates (the monitor clips to neutral-ish values).
    """
    out: list[Any] = []
    for row in arr:
        if np.any(np.isnan(row)):
            out.append(SimpleNamespace(x=0.5, y=0.5, z=0.0))
        else:
            out.append(SimpleNamespace(x=float(row[0]), y=float(row[1]), z=float(row[2])))
    return out
