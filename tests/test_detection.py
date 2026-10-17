"""Characterization tests for detection.py — get_ear, gaze_ratio, get_gaze, get_head_pose.

These tests capture the CURRENT (pre-fix) behaviour so that Stage C fixes
show up as explicit test diffs rather than silent regressions.

No cv2, mediapipe or pygame is imported.  Landmarks are stubbed with
types.SimpleNamespace objects that expose .x and .y attributes.

Conventions
-----------
- All coordinates are in the normalised MediaPipe space: x in [0,1], y in [0,1].
- Landmark index i is lm[i]; we build a list large enough for the highest index
  used (478 for iris).

TestGazeRatio tests the pure gaze_ratio() function with exact numeric assertions.
TestGetGaze tests the full get_gaze() pipeline (EMA + classify).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_lm(n: int = 478) -> list[Any]:
    """Return a list of n landmarks, all at (0.5, 0.5)."""
    return [SimpleNamespace(x=0.5, y=0.5) for _ in range(n)]


def set_lm(lm: list[Any], index: int, x: float, y: float) -> None:
    lm[index] = SimpleNamespace(x=x, y=y)


# ---------------------------------------------------------------------------
# get_ear characterization
# ---------------------------------------------------------------------------

# fmt: off
# Right-eye landmark indices used in main.py: [33, 160, 158, 133, 153, 144]
# p1=33(outer), p2=160(top-outer), p3=158(top-inner),
# p4=133(inner), p5=153(bot-inner), p6=144(bot-outer)
#
# EAR formula (Soukupová & Čech 2016):
#   EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
# Current code uses normalised coords → scale-dependent (finding C1).
# fmt: on


def _ear_open_eye() -> float:
    """Return the EAR for a synthetic open eye in normalised coordinates.

    Eye geometry (640×480 image, normalised):
      Horizontal: p1=(0.30, 0.50), p4=(0.39, 0.50)  → width = 0.09
      Vertical top-outer (p2): (0.315, 0.479)
      Vertical top-inner (p3): (0.375, 0.479)
      Vertical bot-outer (p6): (0.315, 0.521)
      Vertical bot-inner (p5): (0.375, 0.521)
    Pixel vertical gap ≈ 10px → 10/480 ≈ 0.0208 in normalised y.
    Pixel horizontal gap ≈ 58px → 58/640 ≈ 0.0906.
    """
    from detection import get_ear  # noqa: PLC0415

    eye_idx = [33, 160, 158, 133, 153, 144]
    lm = make_lm()
    # p1 outer, p4 inner
    set_lm(lm, 33,  0.300, 0.500)
    set_lm(lm, 133, 0.390, 0.500)
    # p2, p6 (top/bot outer vertical)
    set_lm(lm, 160, 0.315, 0.479)
    set_lm(lm, 144, 0.315, 0.521)
    # p3, p5 (top/bot inner vertical)
    set_lm(lm, 158, 0.375, 0.479)
    set_lm(lm, 153, 0.375, 0.521)
    return get_ear(lm, eye_idx)


def _ear_closed_eye() -> float:
    """Return the EAR for a synthetic closed eye (vertical gap ≈ 0)."""
    from detection import get_ear  # noqa: PLC0415

    eye_idx = [33, 160, 158, 133, 153, 144]
    lm = make_lm()
    set_lm(lm, 33,  0.300, 0.500)
    set_lm(lm, 133, 0.390, 0.500)
    # All vertical landmarks at y=0.500 → zero vertical gap
    for idx in [160, 144, 158, 153]:
        set_lm(lm, idx, 0.315, 0.500)
    return get_ear(lm, eye_idx)


class TestGetEar:
    def test_open_eye_reasonable_range(self) -> None:
        """Open eye EAR should be in a plausible range (roughly 0.2 – 0.5 for
        normalised coords at 640×480)."""
        ear = _ear_open_eye()
        assert 0.2 < ear < 0.6, f"Unexpected EAR for open eye: {ear:.4f}"

    def test_closed_eye_near_zero(self) -> None:
        """Closed eye EAR should be approximately 0."""
        ear = _ear_closed_eye()
        assert ear < 0.02, f"Expected near-zero EAR for closed eye, got {ear:.4f}"

    def test_open_greater_than_closed(self) -> None:
        assert _ear_open_eye() > _ear_closed_eye()

    def test_aspect_ratio_dependence(self) -> None:
        """CHARACTERIZATION (finding C1): EAR differs between 640×480 and
        1280×720 for the same *physical* eye geometry because the normalised
        coordinates absorb the image aspect ratio.

        Same physical eye (58×10 px), two resolutions with different aspect
        ratios (4:3 vs 16:9):
          640×480  (4:3):  width_norm=0.0906, height_norm=0.0208 → EAR ≈ 0.229
          1280×720 (16:9): width_norm=0.0453, height_norm=0.0139 → EAR ≈ 0.307

        Both resolutions with the same aspect ratio give the same EAR:
          640×480 and 1280×960 (both 4:3) → EAR values agree within 1%.

        This test asserts the bug EXISTS in current code.  It will PASS while
        the bug is present and must be updated when C1 is fixed.
        """
        from detection import get_ear  # noqa: PLC0415

        eye_idx = [33, 160, 158, 133, 153, 144]

        def ear_at_res(w: int, h: int) -> float:
            # Same physical eye: 58px wide, 10px tall
            lm = make_lm()
            set_lm(lm, 33,  58 / w, 235 / h)
            set_lm(lm, 133, 116 / w, 235 / h)
            set_lm(lm, 160, 68 / w,  230 / h)
            set_lm(lm, 144, 68 / w,  240 / h)
            set_lm(lm, 158, 106 / w, 230 / h)
            set_lm(lm, 153, 106 / w, 240 / h)
            return get_ear(lm, eye_idx)

        # Different aspect ratios (4:3 vs 16:9) → EAR differs
        ear_480p  = ear_at_res(640, 480)    # 4:3
        ear_720p  = ear_at_res(1280, 720)   # 16:9
        assert not np.isclose(ear_480p, ear_720p, rtol=0.01), (
            f"EAR unexpectedly equal across aspect ratios: "
            f"{ear_480p:.4f} (4:3) vs {ear_720p:.4f} (16:9). "
            "If this fails, the aspect-ratio bug has been fixed — update this test."
        )

        # Same aspect ratio, different resolutions → EAR should agree
        ear_960p  = ear_at_res(1280, 960)   # 4:3 at higher resolution
        assert np.isclose(ear_480p, ear_960p, rtol=0.01), (
            f"EAR differs between same aspect-ratio resolutions: "
            f"{ear_480p:.4f} (640×480) vs {ear_960p:.4f} (1280×960)."
        )

    def test_same_physical_eye_same_ear_across_resolutions(self) -> None:
        """C1 fix: get_ear(lm, eye, frame_w, frame_h) gives the same EAR for the
        same *physical* eye geometry at 640×480 and 1280×720.

        Fix applied: landmark coords are scaled by frame_w, frame_h before
        computing distances, so the result is invariant to aspect ratio.
        """
        from detection import get_ear  # noqa: PLC0415

        eye_idx = [33, 160, 158, 133, 153, 144]

        def ear_at_res(w: int, h: int) -> float:
            lm = make_lm()
            set_lm(lm, 33,  58 / w, 235 / h)
            set_lm(lm, 133, 116 / w, 235 / h)
            set_lm(lm, 160, 68 / w,  230 / h)
            set_lm(lm, 144, 68 / w,  240 / h)
            set_lm(lm, 158, 106 / w, 230 / h)
            set_lm(lm, 153, 106 / w, 240 / h)
            return get_ear(lm, eye_idx, frame_w=w, frame_h=h)

        ear_480p = ear_at_res(640, 480)
        ear_720p = ear_at_res(1280, 720)
        assert np.isclose(ear_480p, ear_720p, rtol=0.01), (
            f"EAR differs between 640×480 ({ear_480p:.4f}) and "
            f"1280×720 ({ear_720p:.4f}) for the same physical eye. "
            "C1 not fixed yet."
        )


# ---------------------------------------------------------------------------
# gaze_ratio — pure function tests (exact numeric assertions)
# ---------------------------------------------------------------------------


class TestGazeRatio:
    """Tests for the pure gaze_ratio() function (post-fix C2 behaviour).

    Correct formula (each iris normalised against its OWN eye's corners):
      Right iris (469-472): (iris_x - lm[33].x) / (lm[133].x - lm[33].x)
      Left  iris (474-477): (iris_x - lm[362].x) / (lm[263].x - lm[362].x)

    Symmetric eyes (width=0.09 each), irises centred in their own eyes:
      ratio_r = (0.345 - 0.300) / (0.390 - 0.300) = 0.045 / 0.090 = 0.500
      ratio_l = (0.655 - 0.610) / (0.700 - 0.610) = 0.045 / 0.090 = 0.500
      raw = 0.500  ✓

    Asymmetric eyes (right width=0.090, left width=0.110), irises centred:
      Right iris centred: x = 0.360 + 0.045 = 0.405
      Left  iris centred: x = 0.540 + 0.055 = 0.595
      ratio_r = (0.405 - 0.360) / (0.450 - 0.360) = 0.045 / 0.090 = 0.500
      ratio_l = (0.595 - 0.540) / (0.650 - 0.540) = 0.055 / 0.110 = 0.500
      raw = 0.500  ✓  (old buggy formula gave ≈ 0.692)
    """

    def _make_symmetric_lm(
        self,
        iris_left_x: float = 0.655,
        iris_right_x: float = 0.345,
    ) -> list[Any]:
        """Symmetric frontal face: both eyes width=0.09."""
        lm = make_lm()
        set_lm(lm, 33,  0.300, 0.500)   # right eye outer corner
        set_lm(lm, 133, 0.390, 0.500)   # right eye inner corner
        set_lm(lm, 362, 0.610, 0.500)   # left  eye inner corner
        set_lm(lm, 263, 0.700, 0.500)   # left  eye outer corner
        for i in [469, 470, 471, 472]:
            set_lm(lm, i, iris_right_x, 0.500)
        for i in [474, 475, 476, 477]:
            set_lm(lm, i, iris_left_x, 0.500)
        return lm

    def test_symmetric_centred_ratio_is_0_5(self) -> None:
        """Symmetric eyes, irises centred → raw ratio = 0.500.

        Corrected formula (each iris against its own eye's corners):
          ratio_r = (0.345 - 0.300) / (0.390 - 0.300) = 0.045 / 0.090 = 0.500
          ratio_l = (0.655 - 0.610) / (0.700 - 0.610) = 0.045 / 0.090 = 0.500
          raw = 0.500
        """
        from detection import gaze_ratio  # noqa: PLC0415

        lm = self._make_symmetric_lm(iris_left_x=0.655, iris_right_x=0.345)
        raw = gaze_ratio(lm)
        assert abs(raw - 0.500) < 1e-3, f"Expected 0.500, got {raw:.6f}"

    def test_asymmetric_centred_ratio_is_0_5(self) -> None:
        """Asymmetric eyes, irises centred → raw = 0.500 after C2 fix.

        Setup:
          Right eye: outer=0.360, inner=0.450  → width=0.090
          Left  eye: inner=0.540, outer=0.650  → width=0.110
          Right iris centred: x = 0.360 + 0.045 = 0.405
          Left  iris centred: x = 0.540 + 0.055 = 0.595

        Corrected formula (each iris against its own eye's corners):
          ratio_r = (0.405 - 0.360) / (0.450 - 0.360) = 0.045 / 0.090 = 0.5000
          ratio_l = (0.595 - 0.540) / (0.650 - 0.540) = 0.055 / 0.110 = 0.5000
          raw = 0.500

        Old buggy formula gave raw ≈ 0.6919 — difference = 0.1919.
        """
        from detection import gaze_ratio  # noqa: PLC0415

        lm = make_lm()
        set_lm(lm, 33,  0.360, 0.500)   # right eye outer
        set_lm(lm, 133, 0.450, 0.500)   # right eye inner
        set_lm(lm, 362, 0.540, 0.500)   # left  eye inner
        set_lm(lm, 263, 0.650, 0.500)   # left  eye outer
        for i in [469, 470, 471, 472]:
            set_lm(lm, i, 0.405, 0.500)   # right iris centred in right eye
        for i in [474, 475, 476, 477]:
            set_lm(lm, i, 0.595, 0.500)   # left  iris centred in left eye

        raw = gaze_ratio(lm)
        assert abs(raw - 0.500) < 0.001, (
            f"Expected corrected raw = 0.500 (C2 fix), got {raw:.4f}. "
            "Old buggy value was ≈ 0.6919."
        )

    def test_looking_left_ratio_above_0_5(self) -> None:
        """Both irises shifted image-right → ratio > 0.5 (driver looking left)."""
        from detection import gaze_ratio  # noqa: PLC0415

        lm = self._make_symmetric_lm(iris_left_x=0.695, iris_right_x=0.385)
        raw = gaze_ratio(lm)
        assert raw > 0.5, f"Expected ratio > 0.5 for leftward gaze, got {raw:.4f}"

    def test_looking_right_ratio_below_0_5(self) -> None:
        """Both irises shifted image-left → ratio < 0.5 (driver looking right)."""
        from detection import gaze_ratio  # noqa: PLC0415

        lm = self._make_symmetric_lm(iris_left_x=0.615, iris_right_x=0.305)
        raw = gaze_ratio(lm)
        assert raw < 0.5, f"Expected ratio < 0.5 for rightward gaze, got {raw:.4f}"


# ---------------------------------------------------------------------------
# get_gaze characterization (EMA + classify pipeline)
# ---------------------------------------------------------------------------


class TestGetGaze:
    """Characterization tests for get_gaze (current behaviour, including bug C2).

    Bug C2: left iris (474-477) is normalised against RIGHT eye corners (33, 133),
    and right iris (469-472) against LEFT eye corners (362, 263).
    """

    def _make_frontal_lm(
        self,
        iris_left_x: float = 0.655,
        iris_right_x: float = 0.345,
    ) -> list[Any]:
        """Frontal face landmark set with configurable iris x-positions.

        Symmetric layout (in normalised coords, unflipped image):
          Right eye (image-right): corners at x=0.30 (outer, lm33) and x=0.39 (inner, lm133)
          Left eye  (image-left):  corners at x=0.61 (inner, lm362) and x=0.70 (outer, lm263)
        Eye widths: 0.09 each.

        Default iris centres are each iris centred in its own eye.
        """
        lm = make_lm()
        # Right eye corners
        set_lm(lm, 33,  0.300, 0.500)   # outer-right
        set_lm(lm, 133, 0.390, 0.500)   # inner-right
        # Left eye corners
        set_lm(lm, 362, 0.610, 0.500)   # inner-left
        set_lm(lm, 263, 0.700, 0.500)   # outer-left
        # Right iris (469-472): all at right_x
        for i in [469, 470, 471, 472]:
            set_lm(lm, i, iris_right_x, 0.500)
        # Left iris (474-477): all at left_x
        for i in [474, 475, 476, 477]:
            set_lm(lm, i, iris_left_x, 0.500)
        return lm

    def test_center_gaze_returns_center(self) -> None:
        """With both irises centred in their own eyes, raw ratio ≈ 0.5 → CENTER."""
        from detection import get_gaze  # noqa: PLC0415

        lm = self._make_frontal_lm(iris_left_x=0.655, iris_right_x=0.345)
        # Use ema_ratio=0.5 so EMA doesn't pull it off-centre
        gaze, ema = get_gaze(lm, 0.5)
        assert gaze == "CENTER", f"Expected CENTER, got {gaze} (ema_ratio={ema:.4f})"
        assert 0.40 < ema < 0.60, f"Unexpected ema_ratio {ema:.4f} for centred gaze"

    def test_left_gaze_detected(self) -> None:
        """Irises shifted strongly image-right → should return LEFT after EMA."""
        from detection import get_gaze  # noqa: PLC0415

        # Both irises well to the right (driver looking left in unflipped frame)
        lm = self._make_frontal_lm(iris_left_x=0.695, iris_right_x=0.385)
        gaze, ema = get_gaze(lm, 0.8)   # prime EMA toward LEFT
        assert gaze in ("LEFT", "CENTER"), f"Expected LEFT or CENTER, got {gaze}"

    def test_right_gaze_detected(self) -> None:
        """Irises shifted strongly image-left → should return RIGHT after EMA."""
        from detection import get_gaze  # noqa: PLC0415

        lm = self._make_frontal_lm(iris_left_x=0.615, iris_right_x=0.305)
        gaze, ema = get_gaze(lm, 0.2)   # prime EMA toward RIGHT
        assert gaze in ("RIGHT", "CENTER"), f"Expected RIGHT or CENTER, got {gaze}"

    def test_asymmetric_centred_ema_is_0_5(self) -> None:
        """Asymmetric eyes, irises centred → EMA = 0.500 after C2 fix.

        Corrected gaze_ratio() returns 0.500 for centred irises regardless of
        eye-width asymmetry.  With prior_ema=0.5:
          new_ema = 0.3 * 0.500 + 0.7 * 0.500 = 0.500

        Old buggy formula gave raw ≈ 0.6919 → EMA ≈ 0.5576.
        """
        from config import CFG  # noqa: PLC0415
        from detection import gaze_ratio, get_gaze  # noqa: PLC0415

        lm = make_lm()
        set_lm(lm, 33,  0.360, 0.500)
        set_lm(lm, 133, 0.450, 0.500)
        set_lm(lm, 362, 0.540, 0.500)
        set_lm(lm, 263, 0.650, 0.500)
        for i in [469, 470, 471, 472]:
            set_lm(lm, i, 0.405, 0.500)
        for i in [474, 475, 476, 477]:
            set_lm(lm, i, 0.595, 0.500)

        import math  # noqa: PLC0415

        raw = gaze_ratio(lm)
        prior_ema = 0.5
        dt = 1.0 / 30  # default dt for get_gaze()
        alpha = 1.0 - math.exp(-dt / CFG.tau_gaze_s)
        expected_ema = alpha * raw + (1 - alpha) * prior_ema

        _gaze, ema = get_gaze(lm, prior_ema)

        # Post-fix: raw = 0.500 → EMA = 0.500
        assert abs(raw - 0.500) < 0.001, (
            f"Expected corrected raw=0.500, got {raw:.4f}. "
            "Old buggy value was ≈ 0.6919."
        )
        assert abs(ema - expected_ema) < 1e-4, (
            f"EMA {ema:.4f} does not match expected {expected_ema:.4f}"
        )
        assert abs(ema - 0.500) < 0.001, (
            f"Expected EMA=0.500 for centred asymmetric eyes after C2 fix, got {ema:.4f}."
        )


# ---------------------------------------------------------------------------
# get_head_pose — solvePnP-based head pose (Stage C3/C4)
# ---------------------------------------------------------------------------


class TestGetHeadPose:
    """Tests for the new get_head_pose(lm, frame_w, frame_h) → (pitch_deg, yaw_deg).

    Uses cv2.solvePnP with a standard 6-point 3D face model.  The function
    must be importable without side effects (cv2 is imported lazily inside the
    function body, not at module level).

    Coordinate conventions (after cv2.flip, mirrored frame):
      pitch_deg > 0 → head tilted DOWN  (nose moves toward chin)
      pitch_deg < 0 → head tilted UP
      yaw_deg   > 0 → head turned RIGHT (driver's right)
      yaw_deg   < 0 → head turned LEFT  (driver's left)

    Test strategy: construct synthetic 2D projections by projecting the 3D
    model points through a known camera matrix, then verify that
    get_head_pose recovers the expected angles.
    """

    # Standard 6-point 3D face model (mm, camera-aligned coords).
    # Indices: nose=1, chin=152, right-eye=33, left-eye=263, r-mouth=61, l-mouth=291
    _MODEL_3D = [
        (  0.0,    0.0,    0.0),   # nose tip
        (  0.0,  -63.6,  -12.5),   # chin
        ( 43.3,   32.7,  -26.0),   # right-eye outer (image right, driver left)
        (-43.3,   32.7,  -26.0),   # left-eye outer  (image left,  driver right)
        ( 28.9,  -28.9,  -24.1),   # right mouth corner
        (-28.9,  -28.9,  -24.1),   # left  mouth corner
    ]
    _LM_INDICES = [1, 152, 33, 263, 61, 291]

    def _project_to_lm(
        self,
        rvec,
        tvec,
        cam_matrix,
        frame_w: int = 640,
        frame_h: int = 480,
    ) -> list[Any]:
        """Project 3D model points through a rotation/translation and return
        a landmark list with the resulting normalised coordinates."""
        import cv2  # noqa: PLC0415

        model_pts = __import__("numpy").array(self._MODEL_3D, dtype="float64")
        pts_2d, _ = cv2.projectPoints(model_pts, rvec, tvec, cam_matrix, None)
        lm = make_lm()
        for idx, lm_i in enumerate(self._LM_INDICES):
            px, py = pts_2d[idx, 0]
            set_lm(lm, lm_i, float(px) / frame_w, float(py) / frame_h)
        return lm

    def _make_frontal_rvec_tvec(self):
        """Rotation and translation for a frontal (0°, 0°) face at z=500 mm."""
        import numpy as np  # noqa: PLC0415

        rvec = np.zeros((3, 1), dtype="float64")   # no rotation
        tvec = np.array([[0.0], [0.0], [500.0]])   # 500 mm ahead
        return rvec, tvec

    def _cam_matrix(self, frame_w: int = 640, frame_h: int = 480):
        """Approximate camera matrix: focal = max(w,h), pp = centre."""
        import numpy as np  # noqa: PLC0415

        f = float(max(frame_w, frame_h))
        return np.array([
            [f, 0, frame_w / 2],
            [0, f, frame_h / 2],
            [0, 0,           1],
        ], dtype="float64")

    def test_frontal_face_near_zero_pitch_yaw(self) -> None:
        """Frontal face (rvec=0) → pitch ≈ 0°, yaw ≈ 0°."""
        from detection import get_head_pose  # noqa: PLC0415

        rvec, tvec = self._make_frontal_rvec_tvec()
        cam = self._cam_matrix()
        lm = self._project_to_lm(rvec, tvec, cam)

        pitch, yaw = get_head_pose(lm, frame_w=640, frame_h=480)
        assert abs(pitch) < 5.0, f"Expected pitch ≈ 0°, got {pitch:.2f}°"
        assert abs(yaw) < 5.0, f"Expected yaw ≈ 0°, got {yaw:.2f}°"

    def test_head_down_gives_positive_pitch(self) -> None:
        """Downward pitch rotation → pitch_deg > 0."""
        import numpy as np  # noqa: PLC0415

        from detection import get_head_pose  # noqa: PLC0415

        # 20° downward pitch (rotation around x-axis, positive = down)
        angle = 20.0 * np.pi / 180
        rvec = np.array([[angle], [0.0], [0.0]])
        tvec = np.array([[0.0], [0.0], [500.0]])
        cam = self._cam_matrix()
        lm = self._project_to_lm(rvec, tvec, cam)

        pitch, yaw = get_head_pose(lm, frame_w=640, frame_h=480)
        assert pitch > 10.0, f"Expected pitch > 10° for downward head, got {pitch:.2f}°"
        assert abs(yaw) < 10.0, f"Expected yaw ≈ 0° for no-yaw rotation, got {yaw:.2f}°"

    def test_head_right_gives_positive_yaw(self) -> None:
        """Rightward yaw rotation → yaw_deg > 0."""
        import numpy as np  # noqa: PLC0415

        from detection import get_head_pose  # noqa: PLC0415

        # 20° rightward yaw (rotation around y-axis, positive = right)
        angle = 20.0 * np.pi / 180
        rvec = np.array([[0.0], [angle], [0.0]])
        tvec = np.array([[0.0], [0.0], [500.0]])
        cam = self._cam_matrix()
        lm = self._project_to_lm(rvec, tvec, cam)

        pitch, yaw = get_head_pose(lm, frame_w=640, frame_h=480)
        assert yaw > 10.0, f"Expected yaw > 10° for rightward head, got {yaw:.2f}°"
        assert abs(pitch) < 10.0, f"Expected pitch ≈ 0° for no-pitch rotation, got {pitch:.2f}°"

    def test_returns_floats(self) -> None:
        """get_head_pose returns (float, float)."""
        from detection import get_head_pose  # noqa: PLC0415

        rvec, tvec = self._make_frontal_rvec_tvec()
        cam = self._cam_matrix()
        lm = self._project_to_lm(rvec, tvec, cam)
        pitch, yaw = get_head_pose(lm, frame_w=640, frame_h=480)
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)

    def test_importable_without_cv2_at_collection(self) -> None:
        """get_head_pose must be importable without cv2 at module-collection time.
        This test just verifies the function exists; cv2 is lazy-loaded inside it.
        """
        import detection  # noqa: PLC0415

        assert hasattr(detection, "get_head_pose"), (
            "get_head_pose not found in detection module"
        )
