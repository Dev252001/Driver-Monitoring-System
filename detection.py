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

NOTE (finding C2 — BUG, not yet fixed): gaze_ratio() pairs each iris
with the OPPOSITE eye's corners.  The fix is Stage C1 in Phase 2.
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

    Formula (current code — BUG C2: corners are swapped between eyes):
        left_iris_x  normalised by RIGHT eye corners  (lm[33], lm[133])
        right_iris_x normalised by LEFT  eye corners  (lm[362], lm[263])
        raw = mean of the two per-eye ratios

    0.5 = centred gaze (only when both eyes are symmetric — see C2).
    > 0.5 → driver looking left (image right).
    < 0.5 → driver looking right (image left).

    The swap does not affect symmetric frontal faces but introduces a
    systematic error proportional to the difference in eye widths.
    Fix is Stage C1 (Phase 2).

    Returns:
        Raw gaze ratio (dimensionless, roughly [0, 1]).
    """
    left_iris = iris_center(lm, [474, 475, 476, 477])
    right_iris = iris_center(lm, [469, 470, 471, 472])

    # BUG C2: left iris uses RIGHT eye corners (33, 133);
    #         right iris uses LEFT eye corners (362, 263).
    ratio_l = (left_iris[0] - lm[33].x) / (lm[133].x - lm[33].x + 1e-6)
    ratio_r = (right_iris[0] - lm[362].x) / (lm[263].x - lm[362].x + 1e-6)
    return float((ratio_l + ratio_r) / 2)


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
