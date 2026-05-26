"""Unit tests for head pose estimation utilities.

Tests cover:
  - build_camera_matrix   (shape, focal/principal values)
  - rotation_matrix_to_euler (identity, pure rotations, sign conventions)
  - estimate_head_pose     (too-few-landmarks → None, valid landmarks → result,
                            frontal-ish pose close to zero angles)
  - AttentionDetector      (zone classification, stability window, attention fraction)

No webcam or MediaPipe runtime required.
"""

import math
import pytest
import numpy as np

from src.signals.head_pose import (
    build_camera_matrix,
    estimate_head_pose,
    rotation_matrix_to_euler,
    _LANDMARK_INDICES,
    _MIN_LANDMARKS,
)
from src.signals.attention import AttentionDetector, AttentionZone


# ── build_camera_matrix ───────────────────────────────────────────────────────

class TestBuildCameraMatrix:
    def test_shape(self):
        m = build_camera_matrix(640, 480)
        assert m.shape == (3, 3)

    def test_dtype(self):
        m = build_camera_matrix(640, 480)
        assert m.dtype == np.float64

    def test_focal_default(self):
        m = build_camera_matrix(640, 480, focal_scale=1.0)
        assert m[0, 0] == pytest.approx(640.0)   # fx
        assert m[1, 1] == pytest.approx(640.0)   # fy

    def test_focal_scale(self):
        m = build_camera_matrix(640, 480, focal_scale=0.8)
        assert m[0, 0] == pytest.approx(640 * 0.8)

    def test_principal_point(self):
        m = build_camera_matrix(640, 480)
        assert m[0, 2] == pytest.approx(320.0)   # cx
        assert m[1, 2] == pytest.approx(240.0)   # cy

    def test_last_row(self):
        m = build_camera_matrix(640, 480)
        np.testing.assert_array_equal(m[2], [0.0, 0.0, 1.0])

    def test_different_resolution(self):
        m = build_camera_matrix(1280, 720)
        assert m[0, 0] == pytest.approx(1280.0)
        assert m[0, 2] == pytest.approx(640.0)
        assert m[1, 2] == pytest.approx(360.0)


# ── rotation_matrix_to_euler ──────────────────────────────────────────────────

class TestRotationMatrixToEuler:
    def test_identity_all_zeros(self):
        pitch, yaw, roll = rotation_matrix_to_euler(np.eye(3))
        assert pitch == pytest.approx(0.0, abs=1e-9)
        assert yaw   == pytest.approx(0.0, abs=1e-9)
        assert roll  == pytest.approx(0.0, abs=1e-9)

    def test_ry_30_gives_yaw_30(self):
        """Pure rotation around Y by 30° → yaw = 30°, pitch = roll = 0."""
        theta = math.radians(30.0)
        c, s = math.cos(theta), math.sin(theta)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pitch, yaw, roll = rotation_matrix_to_euler(ry)
        assert yaw   == pytest.approx(30.0, abs=0.1)
        assert pitch == pytest.approx(0.0,  abs=0.1)
        assert roll  == pytest.approx(0.0,  abs=0.1)

    def test_rx_30_gives_pitch_30(self):
        """Pure rotation around X by 30° → pitch = 30°."""
        theta = math.radians(30.0)
        c, s = math.cos(theta), math.sin(theta)
        rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        pitch, yaw, roll = rotation_matrix_to_euler(rx)
        assert pitch == pytest.approx(30.0, abs=0.1)
        assert yaw   == pytest.approx(0.0,  abs=0.1)

    def test_rz_30_gives_roll_30(self):
        """Pure rotation around Z by 30° → roll = 30°."""
        theta = math.radians(30.0)
        c, s = math.cos(theta), math.sin(theta)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pitch, yaw, roll = rotation_matrix_to_euler(rz)
        assert roll  == pytest.approx(30.0, abs=0.1)
        assert pitch == pytest.approx(0.0,  abs=0.1)

    def test_negative_yaw(self):
        """Rotation by -30° around Y → yaw = -30°."""
        theta = math.radians(-30.0)
        c, s = math.cos(theta), math.sin(theta)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pitch, yaw, roll = rotation_matrix_to_euler(ry)
        assert yaw == pytest.approx(-30.0, abs=0.1)

    def test_returns_floats(self):
        pitch, yaw, roll = rotation_matrix_to_euler(np.eye(3))
        assert isinstance(pitch, float)
        assert isinstance(yaw,   float)
        assert isinstance(roll,  float)


# ── estimate_head_pose ────────────────────────────────────────────────────────

class _FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x; self.y = y; self.z = z


def _make_fake_landmarks(n: int = 478) -> list:
    return [_FakeLandmark(0.5, 0.5) for _ in range(n)]


def _place_frontal_landmarks(landmarks: list, w=640, h=480) -> None:
    """Place the 6 solvePnP landmarks at exact projected positions for R=I.

    Computed by projecting _CANONICAL_FACE_3D with R=identity, tvec=[0,0,300],
    fx=fy=640, cx=320, cy=240.  solvePnP should recover rvec≈0, tvec≈[0,0,300]
    and therefore yaw≈pitch≈roll≈0.
    """
    # Precomputed: u = 640*X/(Z+300)+320, v = 640*Y/(Z+300)+240, normalized
    # _LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
    positions = [
        (0.5000, 0.5000),   # 1   nose tip       (0,0,0)→(320,240)
        (0.5000, 0.7953),   # 152 chin            (0,63.6,-12.5)→(320,382)
        (0.3415, 0.3415),   # 33  left eye outer  (-43.3,-32.7,-26.0)→(219,164)
        (0.6585, 0.3415),   # 263 right eye outer (43.3,-32.7,-26.0)→(421,164)
        (0.3949, 0.6396),   # 61  left mouth      (-28.9,28.9,-24.1)→(253,307)
        (0.6051, 0.6396),   # 291 right mouth     (28.9,28.9,-24.1)→(387,307)
    ]
    for idx, (nx, ny) in zip(_LANDMARK_INDICES, positions):
        landmarks[idx] = _FakeLandmark(nx, ny)


class TestEstimateHeadPose:
    def test_returns_none_for_short_list(self):
        lm = _make_fake_landmarks(n=50)
        cam = build_camera_matrix(640, 480)
        assert estimate_head_pose(lm, 640, 480, cam) is None

    def test_returns_none_at_min_minus_one(self):
        lm = _make_fake_landmarks(n=_MIN_LANDMARKS - 1)
        cam = build_camera_matrix(640, 480)
        assert estimate_head_pose(lm, 640, 480, cam) is None

    def test_returns_measurement_for_full_list(self):
        lm = _make_fake_landmarks(n=478)
        _place_frontal_landmarks(lm)
        cam = build_camera_matrix(640, 480)
        result = estimate_head_pose(lm, 640, 480, cam, timestamp=1.0, frame_index=5)
        assert result is not None
        assert result.frame_index == 5
        assert result.timestamp   == pytest.approx(1.0)

    def test_frontal_pose_angles_reasonable(self):
        """Frontal face positions → |yaw|, |pitch|, |roll| all < 30°."""
        lm = _make_fake_landmarks(n=478)
        _place_frontal_landmarks(lm)
        cam = build_camera_matrix(640, 480)
        result = estimate_head_pose(lm, 640, 480, cam)
        assert result is not None
        assert abs(result.yaw)   < 30.0
        assert abs(result.pitch) < 30.0
        assert abs(result.roll)  < 30.0

    def test_reprojection_error_is_float(self):
        lm = _make_fake_landmarks(n=478)
        _place_frontal_landmarks(lm)
        cam = build_camera_matrix(640, 480)
        result = estimate_head_pose(lm, 640, 480, cam)
        assert isinstance(result.reprojection_error, float)
        assert result.reprojection_error >= 0.0

    def test_rvec_tvec_shapes(self):
        lm = _make_fake_landmarks(n=478)
        _place_frontal_landmarks(lm)
        cam = build_camera_matrix(640, 480)
        result = estimate_head_pose(lm, 640, 480, cam)
        assert result.rvec.shape == (3, 1)
        assert result.tvec.shape == (3, 1)


# ── AttentionDetector ─────────────────────────────────────────────────────────

_ATT_CFG = {
    "head_pose": {
        "attention": {
            "yaw_focus_deg":     20.0,
            "pitch_focus_deg":   15.0,
            "yaw_glance_deg":    35.0,
            "pitch_glance_deg":  25.0,
            "stability_window_s": 5.0,
        }
    }
}


def _pose(yaw: float, pitch: float, ts: float, idx: int = 0):
    """Build a minimal PoseMeasurement for attention testing."""
    from src.signals.head_pose import PoseMeasurement
    return PoseMeasurement(
        yaw=yaw, pitch=pitch, roll=0.0,
        rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)),
        reprojection_error=1.0,
        timestamp=ts, frame_index=idx,
    )


class TestAttentionDetector:
    def test_focused_zone(self):
        det = AttentionDetector(_ATT_CFG)
        a = det.update(_pose(0.0, 0.0, 0.0))
        assert a.zone == AttentionZone.FOCUSED

    def test_glance_zone_yaw(self):
        det = AttentionDetector(_ATT_CFG)
        a = det.update(_pose(28.0, 0.0, 0.0))   # beyond focus but within glance
        assert a.zone == AttentionZone.GLANCE

    def test_looking_away_zone(self):
        det = AttentionDetector(_ATT_CFG)
        a = det.update(_pose(50.0, 0.0, 0.0))
        assert a.zone == AttentionZone.LOOKING_AWAY

    def test_pitch_boundary_focus(self):
        det = AttentionDetector(_ATT_CFG)
        a = det.update(_pose(0.0, 14.9, 0.0))   # just inside focus
        assert a.zone == AttentionZone.FOCUSED

    def test_pitch_boundary_glance(self):
        det = AttentionDetector(_ATT_CFG)
        a = det.update(_pose(0.0, 20.0, 0.0))   # outside focus, inside glance
        assert a.zone == AttentionZone.GLANCE

    def test_no_pose_update_returns_none_initially(self):
        det = AttentionDetector(_ATT_CFG)
        assert det.no_pose_update() is None

    def test_no_pose_update_returns_last(self):
        det = AttentionDetector(_ATT_CFG)
        det.update(_pose(0.0, 0.0, 0.0))
        result = det.no_pose_update()
        assert result is not None
        assert result.zone == AttentionZone.FOCUSED

    def test_attention_fraction_all_focused(self):
        det = AttentionDetector(_ATT_CFG)
        for i in range(10):
            det.update(_pose(0.0, 0.0, float(i) * 0.1, i))
        assert det.no_pose_update().attention_fraction == pytest.approx(1.0)

    def test_attention_fraction_half(self):
        det = AttentionDetector(_ATT_CFG)
        for i in range(5):
            det.update(_pose(0.0,  0.0, float(i) * 0.1, i))
        for i in range(5, 10):
            det.update(_pose(50.0, 0.0, float(i) * 0.1, i))
        frac = det.no_pose_update().attention_fraction
        assert frac == pytest.approx(0.5, abs=0.01)

    def test_stability_std_zero_for_constant_pose(self):
        det = AttentionDetector(_ATT_CFG)
        for i in range(10):
            det.update(_pose(10.0, 5.0, float(i) * 0.1, i))
        a = det.no_pose_update()
        assert a.yaw_std   == pytest.approx(0.0, abs=1e-9)
        assert a.pitch_std == pytest.approx(0.0, abs=1e-9)

    def test_stability_std_positive_for_varying_pose(self):
        det = AttentionDetector(_ATT_CFG)
        for i in range(10):
            det.update(_pose(float(i * 2), 0.0, float(i) * 0.1, i))
        a = det.no_pose_update()
        assert a.yaw_std > 0.0

    def test_window_pruning(self):
        """Frames older than stability_window_s should be pruned."""
        det = AttentionDetector(_ATT_CFG)
        # Feed 10 LOOKING_AWAY frames at t=0..0.9
        for i in range(10):
            det.update(_pose(60.0, 0.0, float(i) * 0.1, i))
        # Feed 10 FOCUSED frames well beyond the 5 s window
        for i in range(10):
            det.update(_pose(0.0, 0.0, 10.0 + float(i) * 0.1, 100 + i))
        # Only the recent FOCUSED frames should be in the window
        assert det.no_pose_update().attention_fraction == pytest.approx(1.0)

    def test_reset_clears_history(self):
        det = AttentionDetector(_ATT_CFG)
        for i in range(5):
            det.update(_pose(0.0, 0.0, float(i) * 0.1, i))
        det.reset()
        assert len(det._history) == 0
