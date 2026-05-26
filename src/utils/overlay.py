"""Overlay rendering utilities for drawing face mesh landmarks onto BGR frames.

MediaPipe 0.10+ Task API
------------------------
Drawing is now done via mediapipe.tasks.python.vision.drawing_utils,
not the legacy mediapipe.solutions.drawing_utils.  The draw_landmarks()
function in the new API takes:
  - image:          np.ndarray (BGR or RGB — it writes in place)
  - landmark_list:  list[NormalizedLandmark]  (plain Python list)
  - connections:    list[Connection]  (.start / .end integer attributes)
  - *_drawing_spec: DrawingSpec or Mapping[tuple[int,int], DrawingSpec]

Ownership convention
--------------------
Every public function returns a *new* ndarray and never modifies the input
frame in place.  MediaPipe's draw_landmarks does modify its image argument,
so we pass it a copy (frame.copy()) — the copy is then returned as the result.

Performance notes
-----------------
DrawingSpec constants are module-level.  Constructing one per frame allocates
small objects that accumulate GC pressure during long runs.
"""

from __future__ import annotations

import cv2
import numpy as np
from mediapipe.tasks.python.vision import drawing_styles as _styles
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec, draw_landmarks
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as _FLC

from src.utils.landmarks import CONNECTIONS, extract_region


# ── Module-level DrawingSpec constants ───────────────────────────────────────
# is_drawing_landmarks=False suppresses the per-point dot — cleaner at high
# landmark density.  Pass None as landmark_drawing_spec to achieve the same.

_SPEC_CONTOUR = DrawingSpec(color=(80,  230,  90), thickness=2, circle_radius=1)
_SPEC_EYE     = DrawingSpec(color=(50,  200, 255), thickness=1, circle_radius=1)
_SPEC_LIPS    = DrawingSpec(color=(0,   160, 255), thickness=1, circle_radius=1)


# ── Public rendering functions ────────────────────────────────────────────────

def draw_face_mesh(
    frame:            np.ndarray,
    landmarks:        list,          # list[NormalizedLandmark] for one face
    *,
    draw_tesselation: bool = True,
    draw_contour:     bool = True,
    draw_eyes:        bool = True,
    draw_lips:        bool = True,
    draw_irises:      bool = False,
) -> np.ndarray:
    """Draw face mesh connection groups onto a copy of *frame*.

    draw_irises requires the model to have been created with refine_landmarks=True
    (adds 10 iris points at indices 468–477).
    """
    canvas = frame.copy()

    if draw_tesselation:
        draw_landmarks(
            canvas, landmarks,
            connections=_FLC.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=_styles.get_default_face_mesh_tesselation_style(),
            is_drawing_landmarks=False,
        )

    if draw_contour:
        draw_landmarks(
            canvas, landmarks,
            connections=_FLC.FACE_LANDMARKS_FACE_OVAL,
            landmark_drawing_spec=None,
            connection_drawing_spec=_SPEC_CONTOUR,
            is_drawing_landmarks=False,
        )

    if draw_eyes:
        for conn in (
            _FLC.FACE_LANDMARKS_LEFT_EYE,
            _FLC.FACE_LANDMARKS_RIGHT_EYE,
            _FLC.FACE_LANDMARKS_LEFT_EYEBROW,
            _FLC.FACE_LANDMARKS_RIGHT_EYEBROW,
        ):
            draw_landmarks(
                canvas, landmarks,
                connections=conn,
                landmark_drawing_spec=None,
                connection_drawing_spec=_SPEC_EYE,
                is_drawing_landmarks=False,
            )

    if draw_lips:
        draw_landmarks(
            canvas, landmarks,
            connections=_FLC.FACE_LANDMARKS_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_SPEC_LIPS,
            is_drawing_landmarks=False,
        )

    if draw_irises:
        iris_style = _styles.get_default_face_mesh_iris_connections_style()
        for conn in (_FLC.FACE_LANDMARKS_LEFT_IRIS, _FLC.FACE_LANDMARKS_RIGHT_IRIS):
            draw_landmarks(
                canvas, landmarks,
                connections=conn,
                landmark_drawing_spec=None,
                connection_drawing_spec=iris_style,
                is_drawing_landmarks=False,
            )

    return canvas


def draw_bounding_box(
    frame:     np.ndarray,
    landmarks: list,
    *,
    color:     tuple[int, int, int] = (0, 255, 100),
    thickness: int = 1,
    padding:   int = 8,
) -> np.ndarray:
    """Draw a tight bounding box around the face contour on a copy of *frame*."""
    canvas = frame.copy()
    h, w   = frame.shape[:2]
    pts    = extract_region(landmarks, w, h, CONNECTIONS["contour"])
    if pts.size == 0:
        return canvas
    x1 = max(0,     int(pts[:, 0].min()) - padding)
    y1 = max(0,     int(pts[:, 1].min()) - padding)
    x2 = min(w - 1, int(pts[:, 0].max()) + padding)
    y2 = min(h - 1, int(pts[:, 1].max()) + padding)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return canvas


def draw_fps_counter(
    frame:  np.ndarray,
    fps:    float,
    *,
    origin: tuple[int, int]      = (10, 32),
    color:  tuple[int, int, int] = (0, 255, 100),
) -> np.ndarray:
    canvas = frame.copy()
    cv2.putText(
        canvas, f"FPS: {fps:.1f}", origin,
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA,
    )
    return canvas


def draw_status_text(
    frame: np.ndarray,
    text:  str,
    *,
    color: tuple[int, int, int] = (0, 80, 255),
) -> np.ndarray:
    """Draw centred status text (e.g. 'No face detected') on a copy of *frame*."""
    canvas      = frame.copy()
    h, w        = canvas.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(
        canvas, text,
        ((w - tw) // 2, (h + th) // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
    )
    return canvas
