"""Facial landmark index groups and coordinate extraction utilities.

MediaPipe 0.10+ Task API
------------------------
The legacy mp.solutions namespace is gone.  Connections are now
FaceLandmarksConnections.Connection objects (dataclasses with .start / .end
integer attributes) grouped in plain Python lists, not frozensets of tuples.

Landmark lists are also plain Python lists of NormalizedLandmark objects
(x, y, z ∈ float) rather than protobuf NormalizedLandmarkList messages.
The coordinate space is identical: x, y ∈ [0.0, 1.0] normalised to image
dimensions; z is relative depth normalised to face size (negative = closer).

Indexing approach
-----------------
CONNECTIONS re-exports the canonical connection lists from FaceLandmarksConnections
under readable names.  Using the library's own constants prevents topology
drift when MediaPipe updates its face model.

INDEX maps semantic names to stable single-point indices for geometry work
(eye corners, nose tip, etc.).  These sit on bone-adjacent landmarks that are
less sensitive to expression changes than mid-face soft-tissue points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as _FLC

logger = logging.getLogger(__name__)


# ── Connection sets ───────────────────────────────────────────────────────────
# Each value is list[FaceLandmarksConnections.Connection] with .start/.end ints.
# draw_landmarks() in the Task API consumes these directly.

CONNECTIONS: dict[str, list] = {
    "tesselation":   _FLC.FACE_LANDMARKS_TESSELATION,
    "contour":       _FLC.FACE_LANDMARKS_FACE_OVAL,
    "left_eye":      _FLC.FACE_LANDMARKS_LEFT_EYE,
    "right_eye":     _FLC.FACE_LANDMARKS_RIGHT_EYE,
    "left_eyebrow":  _FLC.FACE_LANDMARKS_LEFT_EYEBROW,
    "right_eyebrow": _FLC.FACE_LANDMARKS_RIGHT_EYEBROW,
    "lips":          _FLC.FACE_LANDMARKS_LIPS,
    "left_iris":     _FLC.FACE_LANDMARKS_LEFT_IRIS,
    "right_iris":    _FLC.FACE_LANDMARKS_RIGHT_IRIS,
}


# ── Semantic single-point indices ─────────────────────────────────────────────

INDEX: dict[str, int] = {
    "nose_tip":          1,
    "chin":              152,
    "forehead":          10,
    "left_eye_inner":    133,
    "left_eye_outer":    33,
    "right_eye_inner":   362,
    "right_eye_outer":   263,
    "left_eye_top":      159,
    "left_eye_bottom":   145,
    "right_eye_top":     386,
    "right_eye_bottom":  374,
    "mouth_left":        61,
    "mouth_right":       291,
    "upper_lip":         13,
    "lower_lip":         14,
}


# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LandmarkPoint:
    """Single pixel-space landmark with depth."""
    x:     float
    y:     float
    z:     float    # normalised depth; negative = closer to camera
    index: int


# ── Extraction utilities ──────────────────────────────────────────────────────

def to_pixel(lm, width: int, height: int) -> tuple[int, int]:
    """Convert one NormalizedLandmark to integer pixel coordinates."""
    return int(lm.x * width), int(lm.y * height)


def extract_all(landmarks: list, width: int, height: int) -> list[LandmarkPoint]:
    """Convert every landmark in a list[NormalizedLandmark] to pixel-space points.

    landmarks is the list returned by FaceLandmarkerResult.face_landmarks[face_idx].
    """
    return [
        LandmarkPoint(x=lm.x * width, y=lm.y * height, z=lm.z, index=i)
        for i, lm in enumerate(landmarks)
    ]


def extract_region(
    landmarks:       list,
    width:           int,
    height:          int,
    connection_list: list,
) -> np.ndarray:
    """Return (N, 2) float32 pixel array for all unique indices in a connection list.

    connection_list is a list[Connection] with .start and .end integer attributes.
    Collecting unique indices avoids visiting the same landmark twice when a
    point participates in multiple edges.
    """
    unique_indices = {c.start for c in connection_list} | {c.end for c in connection_list}
    return np.array(
        [[landmarks[i].x * width, landmarks[i].y * height] for i in unique_indices],
        dtype=np.float32,
    )


def get_named_landmark(
    landmarks: list,
    name:      str,
    width:     int,
    height:    int,
) -> Optional[LandmarkPoint]:
    """Retrieve a named semantic landmark in pixel space, or None if unknown."""
    idx = INDEX.get(name)
    if idx is None:
        logger.warning("Unknown landmark name '%s'. Valid: %s", name, list(INDEX))
        return None
    lm = landmarks[idx]
    return LandmarkPoint(x=lm.x * width, y=lm.y * height, z=lm.z, index=idx)
