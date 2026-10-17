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
    """Compute Eye Aspect Ratio for one eye (Soukupová & Čech 2016).

    Formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        lm:  Full landmark list (MediaPipe normalised coords: x,y ∈ [0,1]).
        eye: Six landmark indices [p1, p2, p3, p4, p5, p6] where
             p1/p4 are horizontal corners and p2/p3/p5/p6 are
             vertical lid landmarks.

    Returns:
        EAR value (dimensionless).  0 for fully closed, ~0.2–0.4 for open.

    Notes (finding C1 — aspect-ratio dependence):
        The Euclidean distances are computed in normalised coordinate space
        (x ∈ [0,1] relative to image width, y ∈ [0,1] relative to image
        height).  On a non-square sensor (e.g. 640×480 vs 1280×720) the
        horizontal and vertical scales differ, so the same physical eye
        geometry produces different EAR values at different aspect ratios.

        **Mitigation:** per-driver calibration records the driver's own
        open-eye EAR distribution (``Config.calib_frames`` frames) and sets
        the threshold as ``mean(EAR_calib) * Config.ear_factor``.  The
        threshold adapts to each camera/aspect-ratio automatically; only
        the relative drop matters.  A fixed cross-camera threshold of ~0.25
        would not be reliable (see ``Config.ear_factor``).
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


def get_gaze(lm: list, ema_ratio: float, dt: float = 1.0 / 30) -> tuple[str, float]:
    """Apply EMA smoothing to the raw gaze ratio and classify gaze direction.

    This function is kept for backward compatibility; new callers should
    call ``gaze_ratio()`` directly and manage the EMA with a time-based
    alpha (see ``DriverMonitor.process_landmarks``).

    Args:
        lm:        Full landmark list (MediaPipe normalised coords).
        ema_ratio: Previous EMA value (initialise to 0.5 for centred gaze).
        dt:        Frame duration in seconds for time-based EMA alpha.
                   Default 1/30 s (30 FPS).

    Returns:
        Tuple of (direction, new_ema_ratio) where direction is one of
        ``"LEFT"``, ``"RIGHT"``, ``"CENTER"``.
    """
    import math  # noqa: PLC0415

    from config import CFG  # noqa: PLC0415 (avoid import-time side effects)

    ratio = gaze_ratio(lm)
    alpha = 1.0 - math.exp(-dt / CFG.tau_gaze_s)
    ema_ratio = alpha * ratio + (1.0 - alpha) * ema_ratio

    if ema_ratio > CFG.gaze_left_threshold:
        return "LEFT", ema_ratio
    elif ema_ratio < CFG.gaze_right_threshold:
        return "RIGHT", ema_ratio
    else:
        return "CENTER", ema_ratio

# ---------------------------------------------------------------------------
# Standard 6-point 3D face model for solvePnP (in mm, canonical pose).
# Landmark indices correspond to MediaPipe topology (after cv2.flip).
# After cv2.flip:  image-left = driver-right, image-right = driver-left.
#
# Convention: +pitch = head down, +yaw = head right (driver perspective).
# ---------------------------------------------------------------------------
_HEAD_POSE_MODEL_3D = [
    (  0.0,    0.0,    0.0),   # 1:   nose tip
    (  0.0,  -63.6,  -12.5),   # 152: chin
    ( 43.3,   32.7,  -26.0),   # 33:  right-eye outer (driver-left, image-right)
    (-43.3,   32.7,  -26.0),   # 263: left-eye outer  (driver-right, image-left)
    ( 28.9,  -28.9,  -24.1),   # 61:  right mouth corner
    (-28.9,  -28.9,  -24.1),   # 291: left  mouth corner
]
_HEAD_POSE_LM_IDX = [1, 152, 33, 263, 61, 291]


def get_head_pose(lm: list, frame_w: int = 640, frame_h: int = 480) -> tuple[float, float]:
    """Estimate head pitch and yaw via cv2.solvePnP.

    Converts MediaPipe normalised landmark coordinates to pixel coordinates,
    then solves the 2D-3D correspondence with an approximate pinhole camera
    model (focal length = max(frame_w, frame_h), principal point = centre).

    Args:
        lm:      Full landmark list (MediaPipe normalised coords).
        frame_w: Frame width in pixels (default 640).
        frame_h: Frame height in pixels (default 480).

    Returns:
        Tuple (pitch_deg, yaw_deg) in degrees.
        pitch_deg > 0 → head tilted DOWN.
        yaw_deg   > 0 → head turned RIGHT (driver's right, after cv2.flip).

    Notes:
        cv2 is imported lazily so that importing detection.py never touches
        the OpenCV runtime (important for headless CI).
        The approximate camera matrix introduces a small systematic offset
        (typically < 5°); it is sufficient for alert thresholding.
        Phase 3 evaluation should characterise per-camera calibration error.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    # 2D image points: normalised → pixel coordinates
    pts_2d = np.array(
        [[lm[i].x * frame_w, lm[i].y * frame_h] for i in _HEAD_POSE_LM_IDX],
        dtype="float64",
    )
    pts_3d = np.array(_HEAD_POSE_MODEL_3D, dtype="float64")

    # Approximate pinhole camera matrix
    focal = float(max(frame_w, frame_h))
    cam_matrix = np.array(
        [[focal, 0, frame_w / 2],
         [0, focal, frame_h / 2],
         [0, 0, 1]],
        dtype="float64",
    )

    ok, rvec, _ = cv2.solvePnP(
        pts_3d, pts_2d, cam_matrix, None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0

    # Rodrigues → rotation matrix → Euler angles
    rot_mat, _ = cv2.Rodrigues(rvec)

    sy = float(np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2))
    if sy > 1e-6:
        pitch_rad = float(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
        yaw_rad   = float(np.arctan2(-rot_mat[2, 0], sy))
    else:
        # Gimbal lock
        pitch_rad = float(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))
        yaw_rad   = float(np.arctan2(-rot_mat[2, 0], sy))

    pitch_deg = float(np.degrees(pitch_rad))
    yaw_deg   = float(np.degrees(yaw_rad))
    return pitch_deg, yaw_deg


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
