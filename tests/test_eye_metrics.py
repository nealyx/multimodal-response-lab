"""Unit tests for EAR calculation and eye measurement extraction.

All tests are pure-Python — no webcam, no MediaPipe runtime required.
Landmark lists are stubbed with simple objects that expose .x, .y, .z.
"""

import math
import pytest
from src.signals.eye_metrics import (
    EAR_CLOSED_REF,
    EAR_OPEN_REF,
    compute_ear,
    ear_to_openness,
    extract_eye_measurements,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _FakeLandmark:
    """Minimal stand-in for mediapipe NormalizedLandmark."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_fake_landmarks(n: int = 478) -> list[_FakeLandmark]:
    """Return n neutral-position fake landmarks (all at 0.5, 0.5)."""
    return [_FakeLandmark(0.5, 0.5) for _ in range(n)]


def _set_open_eye(landmarks: list, indices: tuple, frame_w=640, frame_h=480) -> None:
    """Place EAR landmarks in a geometrically consistent open-eye configuration.

    Horizontal width = 40 px, vertical gap = 14 px.
    Expected EAR ≈ (7 + 7) / (2 × 40) = 0.175  …scaled from pixel coords,
    which is within the (0.15, 0.35) valid-open range used in tests.
    """
    p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx = indices
    cx, cy = 0.5, 0.5          # eye centre (normalised)
    hw = 40 / frame_w          # half-width  (40 px → normalised)
    vg =  7 / frame_h          # half vertical gap

    landmarks[p1_idx] = _FakeLandmark(cx - hw, cy)       # left corner
    landmarks[p4_idx] = _FakeLandmark(cx + hw, cy)       # right corner
    landmarks[p2_idx] = _FakeLandmark(cx - hw/2, cy - vg)  # upper-left
    landmarks[p3_idx] = _FakeLandmark(cx + hw/2, cy - vg)  # upper-right
    landmarks[p6_idx] = _FakeLandmark(cx - hw/2, cy + vg)  # lower-left
    landmarks[p5_idx] = _FakeLandmark(cx + hw/2, cy + vg)  # lower-right


def _set_closed_eye(landmarks: list, indices: tuple, frame_w=640, frame_h=480) -> None:
    """Collapse lids so EAR ≈ 0 (p2/p6 and p3/p5 coincide)."""
    p1_idx, p2_idx, p3_idx, p4_idx, p5_idx, p6_idx = indices
    cx, cy = 0.5, 0.5
    hw = 40 / frame_w

    landmarks[p1_idx] = _FakeLandmark(cx - hw, cy)
    landmarks[p4_idx] = _FakeLandmark(cx + hw, cy)
    for idx in (p2_idx, p3_idx, p5_idx, p6_idx):
        landmarks[idx] = _FakeLandmark(cx, cy)  # all at centre → 0 vertical gap


# ── compute_ear ───────────────────────────────────────────────────────────────

class TestComputeEar:
    def test_symmetric_open_eye(self):
        """Symmetric open eye: EAR should be > 0.15."""
        # Horizontal: width 80 px.  Vertical gap: 14 px each side.
        # EAR = (14 + 14) / (2 * 80) = 0.175
        p1 = (0.0, 0.0);  p4 = (80.0, 0.0)
        p2 = (20.0, -7.0); p6 = (20.0,  7.0)
        p3 = (60.0, -7.0); p5 = (60.0,  7.0)
        ear = compute_ear(p1, p2, p3, p4, p5, p6)
        assert abs(ear - 0.175) < 1e-6

    def test_closed_eye_zero(self):
        """Lids touching: EAR should be exactly 0."""
        p1 = (0.0, 0.0); p4 = (80.0, 0.0)
        p2 = p3 = p5 = p6 = (40.0, 0.0)   # all at midline
        ear = compute_ear(p1, p2, p3, p4, p5, p6)
        assert ear == 0.0

    def test_degenerate_horizontal_returns_zero(self):
        """Width of zero (p1 == p4) should return 0 not divide-by-zero."""
        p1 = p4 = (0.0, 0.0)
        p2 = (0.0, -5.0); p6 = (0.0, 5.0)
        p3 = (0.0, -5.0); p5 = (0.0, 5.0)
        assert compute_ear(p1, p2, p3, p4, p5, p6) == 0.0

    def test_ear_scale_invariance(self):
        """EAR should be equal for identical shapes at different scales."""
        def make_pts(scale):
            p1 = (0.0, 0.0);    p4 = (scale * 80, 0.0)
            p2 = (scale * 20, -scale * 7);  p6 = (scale * 20, scale * 7)
            p3 = (scale * 60, -scale * 7);  p5 = (scale * 60, scale * 7)
            return p1, p2, p3, p4, p5, p6
        ear1 = compute_ear(*make_pts(1.0))
        ear2 = compute_ear(*make_pts(2.5))
        assert abs(ear1 - ear2) < 1e-9

    def test_returns_float(self):
        p = (0.0, 0.0)
        result = compute_ear(p, (10.0, -5.0), (10.0, -5.0),
                             (20.0, 0.0), (10.0, 5.0), (10.0, 5.0))
        assert isinstance(result, float)

    def test_open_ear_greater_than_closed(self):
        """Open eye EAR must exceed closed eye EAR."""
        p1 = (0.0, 0.0); p4 = (80.0, 0.0)
        open_ear  = compute_ear(p1, (20.0,-7.0),(60.0,-7.0), p4, (60.0,7.0),(20.0,7.0))
        closed_ear = compute_ear(p1, (40.0, 0.0),(40.0, 0.0), p4, (40.0,0.0),(40.0,0.0))
        assert open_ear > closed_ear


# ── ear_to_openness ───────────────────────────────────────────────────────────

class TestEarToOpenness:
    def test_fully_open(self):
        assert ear_to_openness(EAR_OPEN_REF) == pytest.approx(1.0)

    def test_fully_closed(self):
        assert ear_to_openness(EAR_CLOSED_REF) == pytest.approx(0.0)

    def test_midpoint(self):
        mid = (EAR_CLOSED_REF + EAR_OPEN_REF) / 2
        assert ear_to_openness(mid) == pytest.approx(0.5)

    def test_clamps_above_one(self):
        assert ear_to_openness(1.0) == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        assert ear_to_openness(0.0) == pytest.approx(0.0)

    def test_custom_refs(self):
        assert ear_to_openness(0.5, closed_ref=0.0, open_ref=1.0) == pytest.approx(0.5)

    def test_degenerate_refs_returns_zero(self):
        # open_ref <= closed_ref → division undefined, should return 0.
        assert ear_to_openness(0.3, closed_ref=0.3, open_ref=0.3) == 0.0


# ── extract_eye_measurements ──────────────────────────────────────────────────

class TestExtractEyeMeasurements:
    def test_returns_none_for_short_list(self):
        """Too few landmarks → None (face partially detected)."""
        lm = _make_fake_landmarks(n=200)
        result = extract_eye_measurements(lm, 640, 480, timestamp=0.0, frame_index=0)
        assert result is None

    def test_returns_measurement_for_full_list(self):
        """Full landmark list → non-None EyeMeasurement."""
        from src.signals.eye_metrics import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX
        lm = _make_fake_landmarks(n=478)
        _set_open_eye(lm, LEFT_EYE_EAR_IDX)
        _set_open_eye(lm, RIGHT_EYE_EAR_IDX)
        result = extract_eye_measurements(lm, 640, 480, timestamp=1.0, frame_index=5)
        assert result is not None
        assert result.frame_index == 5
        assert result.timestamp  == pytest.approx(1.0)

    def test_open_eye_ear_positive(self):
        from src.signals.eye_metrics import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX
        lm = _make_fake_landmarks(n=478)
        _set_open_eye(lm, LEFT_EYE_EAR_IDX)
        _set_open_eye(lm, RIGHT_EYE_EAR_IDX)
        result = extract_eye_measurements(lm, 640, 480, timestamp=0.0, frame_index=0)
        assert result.left_ear  > 0.0
        assert result.right_ear > 0.0
        assert result.mean_ear  > 0.0

    def test_closed_eye_ear_near_zero(self):
        from src.signals.eye_metrics import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX
        lm = _make_fake_landmarks(n=478)
        _set_closed_eye(lm, LEFT_EYE_EAR_IDX)
        _set_closed_eye(lm, RIGHT_EYE_EAR_IDX)
        result = extract_eye_measurements(lm, 640, 480, timestamp=0.0, frame_index=0)
        assert result.left_ear  < 0.05
        assert result.right_ear < 0.05

    def test_openness_in_unit_range(self):
        from src.signals.eye_metrics import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX
        lm = _make_fake_landmarks(n=478)
        _set_open_eye(lm, LEFT_EYE_EAR_IDX)
        _set_open_eye(lm, RIGHT_EYE_EAR_IDX)
        result = extract_eye_measurements(lm, 640, 480, timestamp=0.0, frame_index=0)
        for v in (result.left_openness, result.right_openness, result.mean_openness):
            assert 0.0 <= v <= 1.0

    def test_mean_ear_is_average(self):
        from src.signals.eye_metrics import LEFT_EYE_EAR_IDX, RIGHT_EYE_EAR_IDX
        lm = _make_fake_landmarks(n=478)
        _set_open_eye(lm, LEFT_EYE_EAR_IDX)
        _set_closed_eye(lm, RIGHT_EYE_EAR_IDX)
        result = extract_eye_measurements(lm, 640, 480, timestamp=0.0, frame_index=0)
        assert result.mean_ear == pytest.approx(
            (result.left_ear + result.right_ear) / 2.0
        )
