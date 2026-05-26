"""Head pose estimation from MediaPipe face landmarks via solvePnP.

Model and approach
------------------
A 6-point canonical 3-D face model (nose tip, chin, left/right eye outer
corners, left/right mouth corners) is matched to the corresponding 2-D image
landmarks by cv2.solvePnP with the ITERATIVE flag.  The result is a rotation
vector (rvec) and translation vector (tvec) in camera space.

Camera intrinsics are approximated from the frame dimensions:
  focal_length ≈ focal_scale × frame_width
  principal_point = (cx, cy) ≈ (frame_width/2, frame_height/2)

This is a reasonable first-order approximation for a frontal webcam but will
introduce a few degrees of systematic error if the real focal length differs
significantly.  The config key head_pose.focal_scale lets you tune this per
camera without rewriting code.

Euler angles
------------
cv2.Rodrigues converts rvec → R (3×3 rotation matrix).
cv2.RQDecomp3x3 decomposes R → (rotX, rotY, rotZ) in degrees where:
  rotY  ≈  yaw   (positive = face turned right from camera's view)
  rotX  ≈  pitch (positive = face tilted down / chin toward chest)
  rotZ  ≈  roll  (positive = head tilted right / right ear toward shoulder)

Sign conventions verified: identity → (0,0,0), Ry(+30°) → yaw=+30°.

Coordinate frame caveat
-----------------------
The 3-D model and solvePnP output live in "camera space" (Z forward, X right,
Y down in OpenCV convention).  yaw/pitch/roll are extracted from that space;
they approximate but do not perfectly equal head-in-world angles because the
camera is not necessarily centred on the face.  For a typical webcam
interaction this is close enough for attention scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ── Canonical 3-D face model ──────────────────────────────────────────────────
# Six robust, well-separated landmarks in a generic face-coordinate system
# (units ≈ mm, centred near the nose tip).  Sourced from standard solvePnP
# literature; values work well across a range of distances.
#
# MediaPipe landmark indices:
#   1   nose tip
#   152 chin
#   33  left eye outer corner  (from subject's left)
#   263 right eye outer corner
#   61  left mouth corner
#   291 right mouth corner
#
_CANONICAL_FACE_3D = np.array([
    [  0.0,   0.0,   0.0],   # 1  nose tip        (origin)
    [  0.0,  63.6, -12.5],   # 152 chin            Y>0 = below nose (Y-down)
    [-43.3, -32.7, -26.0],   # 33  left eye outer  Y<0 = above nose
    [ 43.3, -32.7, -26.0],   # 263 right eye outer
    [-28.9,  28.9, -24.1],   # 61  left mouth corner
    [ 28.9,  28.9, -24.1],   # 291 right mouth corner
], dtype=np.float64)

_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]  # matches _CANONICAL_FACE_3D rows

_MIN_LANDMARKS = max(_LANDMARK_INDICES) + 1  # 264


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PoseMeasurement:
    """Raw output of one solvePnP call."""
    yaw:         float   # degrees, + = face turned right in camera view
    pitch:       float   # degrees, + = chin toward chest (face tilted down)
    roll:        float   # degrees, + = right ear toward shoulder
    rvec:        np.ndarray   # (3,1) rotation vector (Rodrigues)
    tvec:        np.ndarray   # (3,1) translation vector, camera space (mm)
    reprojection_error: float  # mean pixel reprojection error (quality indicator)
    timestamp:   float
    frame_index: int


# ── Camera intrinsics ─────────────────────────────────────────────────────────

def build_camera_matrix(
    width: int,
    height: int,
    focal_scale: float = 1.0,
) -> np.ndarray:
    """Return a 3×3 intrinsic matrix approximated from frame dimensions.

    focal_scale tunes the assumed focal length as a multiple of frame width.
    Values in [0.8, 1.2] cover most common webcam configurations.
    """
    focal = focal_scale * width
    cx    = width  / 2.0
    cy    = height / 2.0
    return np.array([
        [focal,   0.0, cx],
        [  0.0, focal, cy],
        [  0.0,   0.0,  1.0],
    ], dtype=np.float64)


# ── Euler decomposition (public so tests can call it independently) ────────────

def rotation_matrix_to_euler(rmat: np.ndarray) -> tuple[float, float, float]:
    """Decompose a 3×3 rotation matrix into (pitch, yaw, roll) in degrees.

    Uses cv2.RQDecomp3x3 which returns (rotX, rotY, rotZ).  We re-label:
      pitch = rotX,  yaw = rotY,  roll = rotZ
    so the return order is (pitch, yaw, roll) — consistent with aerospace
    convention where pitch is rotation around X, yaw around Y, roll around Z.
    """
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch = float(angles[0])
    yaw   = float(angles[1])
    roll  = float(angles[2])
    return pitch, yaw, roll


# ── Main estimator ────────────────────────────────────────────────────────────

def estimate_head_pose(
    landmarks:     list,
    width:         int,
    height:        int,
    camera_matrix: np.ndarray,
    *,
    timestamp:   float = 0.0,
    frame_index: int   = 0,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[PoseMeasurement]:
    """Estimate head pose from a MediaPipe landmark list.

    Returns None if the landmark list is too short or solvePnP fails.

    Parameters
    ----------
    landmarks     : list[NormalizedLandmark] — raw output from FaceLandmarker
    width, height : frame dimensions in pixels
    camera_matrix : 3×3 intrinsic matrix (from build_camera_matrix)
    dist_coeffs   : lens distortion coefficients; None → assume zero
    """
    if len(landmarks) < _MIN_LANDMARKS:
        return None

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Project normalised landmarks to pixel coords for the 6 model points
    pts_2d = np.array([
        [landmarks[i].x * width, landmarks[i].y * height]
        for i in _LANDMARK_INDICES
    ], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        _CANONICAL_FACE_3D,
        pts_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    # Rodrigues → rotation matrix → Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = rotation_matrix_to_euler(rmat)

    # Reprojection error as a quality metric
    projected, _ = cv2.projectPoints(
        _CANONICAL_FACE_3D, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(-1, 2)
    reprojection_error = float(np.mean(np.linalg.norm(pts_2d - projected, axis=1)))

    return PoseMeasurement(
        yaw=yaw, pitch=pitch, roll=roll,
        rvec=rvec, tvec=tvec,
        reprojection_error=reprojection_error,
        timestamp=timestamp,
        frame_index=frame_index,
    )
