"""Detection functions for the AI-Based Smart Driver Monitoring System.

Eye Aspect Ratio (Soukupová & Čech 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    where p1/p4 are the horizontal corners and p2/p3/p5/p6 are the
    vertical landmarks of the eyelid contour.

All landmark coordinates are MediaPipe normalised values: x in [0, 1],
y in [0, 1], independent of image resolution.

NOTE (finding C1): EAR computed on normalised coordinates is
aspect-ratio-dependent (640×480 ≠ 1280×720 for the same physical eye).
Mitigation: per-driver calibration normalises the threshold to the
driver's own EAR distribution; see Config.ear_factor.

NOTE (finding C2 — fixed Stage C1): gaze_ratio() now pairs each iris
with its own eye's corners.  See CHANGELOG.md for the arithmetic.
"""

from __future__ import annotations

import numpy as np


def get_ear(lm: list, eye: list[int]) -> float:
    """Compute Eye Aspect Ratio for one eye.

    Args:
        lm:  Full landmark list (MediaPipe normalised coords).
        eye: Six landmark indices [p1, p2, p3, p4, p5, p6] where
             p1/p4 are horizontal corners and p2/p3/p5/p6 are
             vertical lid landmarks.

    Returns:
        EAR value (dimensionless).  0 for fully closed, ~0.3 for open.
    """
    p2 = np.array([lm[eye[1]].x, lm[eye[1]].y])
    p6 = np.array([lm[eye[5]].x, lm[eye[5]].y])
    p3 = np.array([lm[eye[2]].x, lm[eye[2]].y])
    p5 = np.array([lm[eye[4]].x, lm[eye[4]].y])
    p1 = np.array([lm[eye[0]].x, lm[eye[0]].y])
    p4 = np.array([lm[eye[3]].x, lm[eye[3]].y])

    den = 2 * np.linalg.norm(p1 - p4) + 1e-6
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / den)


def iris_center(lm: list, pts: list[int]) -> np.ndarray:
    """Return the mean (x, y) of the given iris landmark indices."""
    return np.array([
        np.mean([lm[i].x for i in pts]),
        np.mean([lm[i].y for i in pts]),
    ])


def gaze_ratio(lm: list) -> float:
    """Compute the raw horizontal gaze ratio from iris and eye-corner landmarks.

    Each iris is normalised against its OWN eye's corners (fix for bug C2):

        ratio_r = (right_iris_x - lm[33].x)  / (lm[133].x - lm[33].x)
        ratio_l = (left_iris_x  - lm[362].x) / (lm[263].x - lm[362].x)
        raw = (ratio_r + ratio_l) / 2

    MediaPipe landmark convention (after cv2.flip, mirrored frame):
        lm[33]  = right-eye outer corner (driver's left eye, image right)
        lm[133] = right-eye inner corner
        lm[362] = left-eye inner corner  (driver's right eye, image left)
        lm[263] = left-eye outer corner

    0.5 = centred gaze.
    > 0.5 → irises shifted image-right → driver looking LEFT.
    < 0.5 → irises shifted image-left  → driver looking RIGHT.

    Returns:
        Raw gaze ratio (dimensionless, in [0, 1] for typical gaze angles).
    """
    left_iris  = iris_center(lm, [474, 475, 476, 477])
    right_iris = iris_center(lm, [469, 470, 471, 472])

    # Each iris normalised against its own eye's corners (C2 fix).
    ratio_r = (right_iris[0] - lm[33].x)  / (lm[133].x - lm[33].x  + 1e-6)
    ratio_l = (left_iris[0]  - lm[362].x) / (lm[263].x - lm[362].x + 1e-6)
    return float((ratio_r + ratio_l) / 2)


def get_gaze(lm: list, ema_ratio: float) -> tuple[str, float]:
    """Apply EMA smoothing to the raw gaze ratio and classify gaze direction.

    Args:
        lm:        Full landmark list (MediaPipe normalised coords).
        ema_ratio: Previous EMA value (initialise to 0.5 for centred gaze).

    Returns:
        Tuple of (direction, new_ema_ratio) where direction is one of
        ``"LEFT"``, ``"RIGHT"``, ``"CENTER"``.
    """
    from config import CFG  # noqa: PLC0415 (avoid import-time side effects)

    ratio = gaze_ratio(lm)
    ema_ratio = CFG.gaze_ema_alpha * ratio + (1 - CFG.gaze_ema_alpha) * ema_ratio

    if ema_ratio > CFG.gaze_left_threshold:
        return "LEFT", ema_ratio
    elif ema_ratio < CFG.gaze_right_threshold:
        return "RIGHT", ema_ratio
    else:
        return "CENTER", ema_ratio

def get_head(lm, baseline, prev_down=False):
    nose = lm[1].y
    eye = (lm[33].y + lm[263].y) / 2
    diff = nose - eye

    if baseline is None:
        return False, diff

    # 🔥 hysteresis (stable detection)
    enter = baseline + 0.08
    exit_  = baseline + 0.05

    if prev_down:
        return (diff > exit_), diff
    else:
        return (diff > enter), diff
