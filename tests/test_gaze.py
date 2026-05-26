"""Unit tests for iris-based gaze estimation.

All tests inject fake landmarks with controlled iris positions and
fake GazeMeasurements with controlled ratios — no webcam required.
"""

import math
import pytest
from src.signals.gaze import (
    GazeDetector,
    GazeMeasurement,
    GazeZone,
    IRIS_MIN_LANDMARKS,
    extract_gaze_measurements,
    _ratio,
    _L_IRIS, _R_IRIS,
    _L_H_LEFT, _L_H_RIGHT, _L_V_TOP, _L_V_BOT,
    _R_H_LEFT, _R_H_RIGHT, _R_V_TOP, _R_V_BOT,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class _FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x; self.y = y; self.z = z


def _make_landmarks(n: int = 478) -> list:
    return [_FakeLandmark(0.5, 0.5) for _ in range(n)]


W, H = 640, 480


def _place_eye_box(lm: list, *, eye: str, norm_h: float, norm_v: float) -> None:
    """Position eye-box and iris landmarks so that h/v ratios equal norm_h/norm_v.

    eye: 'left' (subject's left, indices 473/263/362/386/374)
         'right' (subject's right, indices 468/133/33/159/145)
    The eye box spans [0.3, 0.7] horizontally and [0.4, 0.6] vertically
    (in normalised coords), giving a clear non-degenerate geometry.
    """
    if eye == "left":
        iris_idx, hl_idx, hr_idx, vt_idx, vb_idx = (
            _L_IRIS, _L_H_LEFT, _L_H_RIGHT, _L_V_TOP, _L_V_BOT
        )
        h_left, h_right = 0.20, 0.45   # outer/inner corners for left eye
        v_top,  v_bot   = 0.40, 0.55
    else:
        iris_idx, hl_idx, hr_idx, vt_idx, vb_idx = (
            _R_IRIS, _R_H_LEFT, _R_H_RIGHT, _R_V_TOP, _R_V_BOT
        )
        h_left, h_right = 0.55, 0.80   # inner/outer corners for right eye
        v_top,  v_bot   = 0.40, 0.55

    lm[hl_idx] = _FakeLandmark(h_left,  0.47)
    lm[hr_idx] = _FakeLandmark(h_right, 0.47)
    lm[vt_idx] = _FakeLandmark(0.325,   v_top)
    lm[vb_idx] = _FakeLandmark(0.325,   v_bot)

    iris_x = h_left + norm_h * (h_right - h_left)
    iris_y = v_top  + norm_v * (v_bot   - v_top)
    lm[iris_idx] = _FakeLandmark(iris_x, iris_y)


def _measurement(mean_h: float, mean_v: float, ts: float = 0.0, idx: int = 0,
                 reliable: bool = True) -> GazeMeasurement:
    return GazeMeasurement(
        left_h=mean_h, left_v=mean_v, right_h=mean_h, right_v=mean_v,
        mean_h=mean_h, mean_v=mean_v,
        left_iris_px=(0.0, 0.0), right_iris_px=(0.0, 0.0),
        is_reliable=reliable, timestamp=ts, frame_index=idx,
    )


_BASE_CFG = {
    "gaze": {
        "center_h_lo": 0.35, "center_h_hi": 0.65,
        "center_v_lo": 0.25, "center_v_hi": 0.70,
        "head_yaw_max_on_screen":   25.0,
        "head_pitch_max_on_screen": 20.0,
        "stability_window_s": 5.0,
    }
}


# ── _ratio helper ─────────────────────────────────────────────────────────────

class TestRatioHelper:
    def test_center_of_range(self):
        assert _ratio(5.0, 0.0, 10.0) == pytest.approx(0.5)

    def test_at_lo(self):
        assert _ratio(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_at_hi(self):
        assert _ratio(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_clamp_below(self):
        assert _ratio(-5.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_clamp_above(self):
        assert _ratio(15.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_degenerate_returns_half(self):
        assert _ratio(5.0, 5.0, 5.0) == pytest.approx(0.5)


# ── extract_gaze_measurements ─────────────────────────────────────────────────

class TestExtractGazeMeasurements:
    def test_returns_none_for_short_list(self):
        lm = _make_landmarks(n=400)
        assert extract_gaze_measurements(lm, W, H, mean_ear=0.3) is None

    def test_returns_none_at_min_minus_one(self):
        lm = _make_landmarks(n=IRIS_MIN_LANDMARKS - 1)
        assert extract_gaze_measurements(lm, W, H, mean_ear=0.3) is None

    def test_returns_measurement_for_full_landmarks(self):
        lm = _make_landmarks(n=478)
        _place_eye_box(lm, eye="left",  norm_h=0.5, norm_v=0.5)
        _place_eye_box(lm, eye="right", norm_h=0.5, norm_v=0.5)
        result = extract_gaze_measurements(
            lm, W, H, mean_ear=0.3, timestamp=1.0, frame_index=5
        )
        assert result is not None
        assert result.frame_index == 5
        assert result.timestamp   == pytest.approx(1.0)

    def test_centered_iris_gives_half_ratio(self):
        lm = _make_landmarks(n=478)
        _place_eye_box(lm, eye="left",  norm_h=0.5, norm_v=0.5)
        _place_eye_box(lm, eye="right", norm_h=0.5, norm_v=0.5)
        r = extract_gaze_measurements(lm, W, H, mean_ear=0.3)
        assert r.left_h  == pytest.approx(0.5, abs=0.02)
        assert r.right_h == pytest.approx(0.5, abs=0.02)
        assert r.mean_h  == pytest.approx(0.5, abs=0.02)

    def test_iris_at_left_corner_gives_zero(self):
        lm = _make_landmarks(n=478)
        _place_eye_box(lm, eye="left",  norm_h=0.0, norm_v=0.5)
        _place_eye_box(lm, eye="right", norm_h=0.0, norm_v=0.5)
        r = extract_gaze_measurements(lm, W, H, mean_ear=0.3)
        assert r.left_h  == pytest.approx(0.0, abs=0.02)
        assert r.right_h == pytest.approx(0.0, abs=0.02)

    def test_iris_at_right_corner_gives_one(self):
        lm = _make_landmarks(n=478)
        _place_eye_box(lm, eye="left",  norm_h=1.0, norm_v=0.5)
        _place_eye_box(lm, eye="right", norm_h=1.0, norm_v=0.5)
        r = extract_gaze_measurements(lm, W, H, mean_ear=0.3)
        assert r.left_h  == pytest.approx(1.0, abs=0.02)
        assert r.right_h == pytest.approx(1.0, abs=0.02)

    def test_iris_at_top_gives_zero_v(self):
        lm = _make_landmarks(n=478)
        _place_eye_box(lm, eye="left",  norm_h=0.5, norm_v=0.0)
        _place_eye_box(lm, eye="right", norm_h=0.5, norm_v=0.0)
        r = extract_gaze_measurements(lm, W, H, mean_ear=0.3)
        assert r.left_v  == pytest.approx(0.0, abs=0.02)
        assert r.right_v == pytest.approx(0.0, abs=0.02)

    def test_iris_at_bottom_gives_one_v(self):
        lm = _make_landmarks(n=478)
        _place_eye_box(lm, eye="left",  norm_h=0.5, norm_v=1.0)
        _place_eye_box(lm, eye="right", norm_h=0.5, norm_v=1.0)
        r = extract_gaze_measurements(lm, W, H, mean_ear=0.3)
        assert r.left_v  == pytest.approx(1.0, abs=0.02)
        assert r.right_v == pytest.approx(1.0, abs=0.02)

    def test_mean_is_average_of_left_right(self):
        lm = _make_landmarks(n=478)
        _place_eye_box(lm, eye="left",  norm_h=0.3, norm_v=0.5)
        _place_eye_box(lm, eye="right", norm_h=0.7, norm_v=0.5)
        r = extract_gaze_measurements(lm, W, H, mean_ear=0.3)
        assert r.mean_h == pytest.approx((r.left_h + r.right_h) / 2.0, abs=1e-6)

    def test_unreliable_when_ear_too_low(self):
        lm = _make_landmarks(n=478)
        r = extract_gaze_measurements(lm, W, H, mean_ear=0.05)
        assert r is not None
        assert r.is_reliable is False

    def test_reliable_when_ear_adequate(self):
        lm = _make_landmarks(n=478)
        r = extract_gaze_measurements(lm, W, H, mean_ear=0.30)
        assert r.is_reliable is True

    def test_iris_pixel_coords_match_landmarks(self):
        lm = _make_landmarks(n=478)
        _place_eye_box(lm, eye="left",  norm_h=0.5, norm_v=0.5)
        _place_eye_box(lm, eye="right", norm_h=0.5, norm_v=0.5)
        r = extract_gaze_measurements(lm, W, H, mean_ear=0.3)
        assert r.left_iris_px[0]  == pytest.approx(lm[_L_IRIS].x * W, abs=1e-3)
        assert r.right_iris_px[1] == pytest.approx(lm[_R_IRIS].y * H, abs=1e-3)


# ── GazeDetector zone classification ─────────────────────────────────────────

class TestGazeZoneClassification:
    def test_center(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.50))
        assert a.zone == GazeZone.CENTER

    def test_left(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.20, 0.50))
        assert a.zone == GazeZone.LEFT

    def test_right(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.80, 0.50))
        assert a.zone == GazeZone.RIGHT

    def test_up(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.10))
        assert a.zone == GazeZone.UP

    def test_down(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.90))
        assert a.zone == GazeZone.DOWN

    def test_up_left(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.20, 0.10))
        assert a.zone == GazeZone.UP_LEFT

    def test_down_right(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.80, 0.90))
        assert a.zone == GazeZone.DOWN_RIGHT

    def test_unreliable_when_not_reliable(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.50, reliable=False))
        assert a.zone == GazeZone.UNRELIABLE
        assert a.is_reliable is False

    def test_boundary_exactly_at_h_lo_is_center(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.35, 0.50))
        assert a.zone == GazeZone.CENTER

    def test_boundary_just_below_h_lo_is_left(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.34, 0.50))
        assert a.zone == GazeZone.LEFT


# ── GazeDetector on_screen logic ─────────────────────────────────────────────

class TestOnScreen:
    def test_center_gaze_is_on_screen(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.50))
        assert a.is_on_screen is True

    def test_off_center_gaze_is_not_on_screen(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.20, 0.50))
        assert a.is_on_screen is False

    def test_head_looking_away_overrides_center_gaze(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.50), head_zone="looking_away")
        assert a.is_on_screen is False

    def test_head_focused_does_not_block_on_screen(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.50), head_zone="focused")
        assert a.is_on_screen is True

    def test_large_yaw_overrides_center_gaze(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.50), head_yaw=40.0)
        assert a.is_on_screen is False

    def test_small_yaw_keeps_on_screen(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.50), head_yaw=10.0)
        assert a.is_on_screen is True

    def test_large_pitch_overrides_center_gaze(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.50), head_pitch=30.0)
        assert a.is_on_screen is False

    def test_unreliable_gaze_not_on_screen(self):
        det = GazeDetector(_BASE_CFG)
        a = det.update(_measurement(0.50, 0.50, reliable=False))
        assert a.is_on_screen is False


# ── Rolling stability and on_screen fraction ──────────────────────────────────

class TestRolling:
    def test_stability_zero_for_constant_gaze(self):
        det = GazeDetector(_BASE_CFG)
        for i in range(10):
            det.update(_measurement(0.50, 0.50, ts=float(i) * 0.1, idx=i))
        a = det.no_gaze_update()
        assert a.stability_h == pytest.approx(0.0, abs=1e-9)
        assert a.stability_v == pytest.approx(0.0, abs=1e-9)

    def test_stability_positive_for_varying_gaze(self):
        det = GazeDetector(_BASE_CFG)
        for i in range(10):
            det.update(_measurement(float(i) * 0.05, 0.50, ts=float(i) * 0.1, idx=i))
        a = det.no_gaze_update()
        assert a.stability_h > 0.0

    def test_on_screen_fraction_all_on(self):
        det = GazeDetector(_BASE_CFG)
        for i in range(10):
            det.update(_measurement(0.50, 0.50, ts=float(i) * 0.1, idx=i))
        assert det.no_gaze_update().on_screen_fraction == pytest.approx(1.0)

    def test_on_screen_fraction_all_off(self):
        det = GazeDetector(_BASE_CFG)
        for i in range(10):
            det.update(_measurement(0.10, 0.50, ts=float(i) * 0.1, idx=i))
        assert det.no_gaze_update().on_screen_fraction == pytest.approx(0.0)

    def test_window_pruning(self):
        det = GazeDetector(_BASE_CFG)
        for i in range(10):
            det.update(_measurement(0.10, 0.50, ts=float(i) * 0.1, idx=i))
        # Advance time beyond 5 s window
        for i in range(10):
            det.update(_measurement(0.50, 0.50, ts=10.0 + float(i) * 0.1, idx=100+i))
        # Old off-screen frames should be pruned
        assert det.no_gaze_update().on_screen_fraction == pytest.approx(1.0)

    def test_no_gaze_returns_none_initially(self):
        det = GazeDetector(_BASE_CFG)
        assert det.no_gaze_update() is None

    def test_no_gaze_returns_last_after_update(self):
        det = GazeDetector(_BASE_CFG)
        det.update(_measurement(0.50, 0.50))
        assert det.no_gaze_update() is not None

    def test_reset_clears_history(self):
        det = GazeDetector(_BASE_CFG)
        for i in range(5):
            det.update(_measurement(0.50, 0.50, ts=float(i) * 0.1))
        det.reset()
        assert len(det._history) == 0
