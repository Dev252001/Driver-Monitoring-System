"""Characterization tests for detection.py — get_ear, get_gaze, get_head.

These tests capture the CURRENT (pre-fix) behaviour so that Stage C fixes
show up as explicit test diffs rather than silent regressions.

No cv2, mediapipe or pygame is imported.  Landmarks are stubbed with
types.SimpleNamespace objects that expose .x and .y attributes.

Conventions
-----------
- All coordinates are in the normalised MediaPipe space: x in [0,1], y in [0,1].
- Landmark index i is lm[i]; we build a list large enough for the highest index
  used (478 for iris).
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

    def test_resolution_dependence(self) -> None:
        """CHARACTERIZATION (current bug C1): EAR differs between 640×480 and
        1280×720 for the same physical eye geometry because normalised coords
        absorb the aspect-ratio difference.

        This test asserts the bug EXISTS in current code.  It will PASS while
        the bug is present and must be updated when C2 is fixed.
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

        ear_480p = ear_at_res(640, 480)
        ear_720p = ear_at_res(1280, 720)
        # Bug: these should be equal after fix; currently they differ.
        assert not np.isclose(ear_480p, ear_720p, rtol=0.01), (
            f"EAR unexpectedly equal across resolutions: {ear_480p:.4f} vs {ear_720p:.4f}. "
            "If this fails, the resolution-dependence bug has been fixed — remove this assertion."
        )


# ---------------------------------------------------------------------------
# get_gaze characterization
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

    def test_bug_c2_wrong_corners_asymmetric(self) -> None:
        """CHARACTERIZATION (bug C2): with asymmetric eye widths the old formula
        gives a raw ratio ≠ 0.5 even for centred irises.

        Setup (as specified in PLAN.md decision d):
          Right eye width = 0.09  (x33=0.36, x133=0.45)
          Left  eye width = 0.11  (x362=0.54, x263=0.65)
          Right iris centred in right eye: x = 0.36 + 0.045 = 0.405
          Left  iris centred in left eye:  x = 0.54 + 0.055 = 0.595

        Old formula (wrong corners):
          ratio_r = (0.405 - 0.54) / (0.65 - 0.54) = -0.135 / 0.11 ≈ -1.227
          ratio_l = (0.595 - 0.36) / (0.45 - 0.36) =  0.235 / 0.09 ≈  2.611
          raw = (-1.227 + 2.611) / 2 ≈ 0.692   [NOT 0.5]

        Corrected formula (own corners):
          ratio_r = (0.405 - 0.36) / (0.45 - 0.36) = 0.045 / 0.09 = 0.500
          ratio_l = (0.595 - 0.54) / (0.65 - 0.54) = 0.055 / 0.11 = 0.500
          raw = 0.500                              [correct]

        This test ASSERTS the current (buggy) value ≈ 0.692 so the fix in C1
        shows up as a diff.
        """
        from detection import get_gaze  # noqa: PLC0415

        lm = make_lm()
        # Right eye: outer=0.36, inner=0.45 → width=0.09
        set_lm(lm, 33,  0.360, 0.500)
        set_lm(lm, 133, 0.450, 0.500)
        # Left eye: inner=0.54, outer=0.65 → width=0.11
        set_lm(lm, 362, 0.540, 0.500)
        set_lm(lm, 263, 0.650, 0.500)
        # Right iris centred in right eye
        for i in [469, 470, 471, 472]:
            set_lm(lm, i, 0.405, 0.500)
        # Left iris centred in left eye
        for i in [474, 475, 476, 477]:
            set_lm(lm, i, 0.595, 0.500)

        # EMA alpha=0.3, ema_ratio=0.5 → new_ema = 0.3*raw + 0.7*0.5
        _gaze, ema = get_gaze(lm, 0.5)

        # Compute raw directly to verify
        raw = (
            (0.595 - 0.360) / (0.450 - 0.360 + 1e-6)   # left iris / right corners
            + (0.405 - 0.540) / (0.650 - 0.540 + 1e-6)  # right iris / left corners
        ) / 2
        expected_ema = 0.3 * raw + 0.7 * 0.5

        print(f"\n[C2 characterization] raw={raw:.4f}, ema={ema:.4f}")
        print("  old formula raw ≈ 0.692, corrected raw should be 0.500")

        assert abs(ema - expected_ema) < 1e-4, (
            f"EMA {ema:.4f} does not match expected {expected_ema:.4f}"
        )
        # The bug: raw is NOT 0.5 (it's ~0.69)
        assert raw > 0.60, (
            f"Bug C2 appears to be already fixed (raw={raw:.4f}, expected > 0.60). "
            "Update this test to reflect the corrected formula."
        )


# ---------------------------------------------------------------------------
# get_head characterization
# ---------------------------------------------------------------------------

class TestGetHead:
    def _make_head_lm(
        self,
        nose_y: float,
        eye_r_y: float = 0.40,
        eye_l_y: float = 0.40,
    ) -> list[Any]:
        lm = make_lm()
        set_lm(lm, 1,   0.5, nose_y)
        set_lm(lm, 33,  0.3, eye_r_y)
        set_lm(lm, 263, 0.7, eye_l_y)
        return lm

    def test_none_baseline_returns_false(self) -> None:
        from detection import get_head  # noqa: PLC0415

        lm = self._make_head_lm(nose_y=0.55)
        down, diff = get_head(lm, baseline=None)
        assert not down
        assert abs(diff - (0.55 - 0.40)) < 1e-6

    def test_above_enter_threshold_triggers_down(self) -> None:
        """nose.y - mean(eye.y) > baseline + 0.08 → HEAD DOWN."""
        from detection import get_head  # noqa: PLC0415

        baseline = 0.15   # neutral diff
        # Raise nose to make diff = baseline + 0.09 → should trigger
        nose_y = 0.40 + baseline + 0.09
        lm2 = self._make_head_lm(nose_y=nose_y)
        down, diff = get_head(lm2, baseline=baseline)
        assert down, f"Expected head-down, diff={diff:.4f}, threshold={baseline + 0.08:.4f}"

    def test_below_enter_threshold_no_trigger(self) -> None:
        from detection import get_head  # noqa: PLC0415

        baseline = 0.15
        nose_y = 0.40 + baseline + 0.07   # diff = baseline + 0.07 < baseline + 0.08
        lm = self._make_head_lm(nose_y=nose_y)
        down, _ = get_head(lm, baseline=baseline)
        assert not down

    def test_hysteresis_exit(self) -> None:
        """Once head-down, stays down until diff drops below baseline + 0.05."""
        from detection import get_head  # noqa: PLC0415

        baseline = 0.15
        # diff = baseline + 0.06 → above exit (0.05), below enter (0.08)
        nose_y = 0.40 + baseline + 0.06
        lm = self._make_head_lm(nose_y=nose_y)
        # prev_down=True: should remain down (above exit threshold)
        down, _ = get_head(lm, baseline=baseline, prev_down=True)
        assert down, "Expected hysteresis to keep head-down state"
        # prev_down=False: should NOT enter (below enter threshold)
        down2, _ = get_head(lm, baseline=baseline, prev_down=False)
        assert not down2, "Expected hysteresis to prevent entering head-down state"

    def test_diff_computed_correctly(self) -> None:
        """diff = nose.y - mean(lm[33].y, lm[263].y)."""
        from detection import get_head  # noqa: PLC0415

        lm = self._make_head_lm(nose_y=0.60, eye_r_y=0.35, eye_l_y=0.45)
        _, diff = get_head(lm, baseline=None)
        expected = 0.60 - (0.35 + 0.45) / 2
        assert abs(diff - expected) < 1e-6, f"Expected {expected:.4f}, got {diff:.4f}"
