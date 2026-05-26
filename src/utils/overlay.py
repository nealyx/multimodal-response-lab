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


def draw_eye_metrics(
    frame:    np.ndarray,
    analysis: "BlinkAnalysis",        # src.signals.blink_detector.BlinkAnalysis
    ear_l:    float,
    ear_r:    float,
    *,
    origin:   tuple[int, int] = (10, 60),
) -> np.ndarray:
    """Draw an eye-metrics HUD panel onto a copy of *frame*.

    Renders: state badge, EAR values, openness bars, blink count + rate,
    and a fatigue indicator.  Positioned at *origin* (top-left corner of panel).

    The type annotation for analysis is a string to avoid a circular import —
    this function is called from run_blink_test.py which imports both modules.
    """
    from src.signals.blink_detector import BlinkState  # local to avoid circular

    canvas = frame.copy()
    x, y   = origin
    lh     = 22          # line height in pixels
    font   = cv2.FONT_HERSHEY_SIMPLEX

    # ── State badge ──────────────────────────────────────────────────────────
    state_colors = {
        BlinkState.OPEN:    (0, 230, 80),
        BlinkState.CLOSING: (0, 200, 255),
        BlinkState.CLOSED:  (0, 60,  255),
        BlinkState.OPENING: (0, 200, 255),
    }
    state_col = state_colors.get(analysis.state, (200, 200, 200))
    _text(canvas, f"Eye: {analysis.state.value.upper()}", x, y,       font, 0.55, state_col)

    # ── EAR values ───────────────────────────────────────────────────────────
    _text(canvas, f"EAR  L:{ear_l:.3f}  R:{ear_r:.3f}  M:{analysis.mean_ear:.3f}",
          x, y + lh,   font, 0.48, (200, 200, 200))

    # ── Openness bars (left / right) ─────────────────────────────────────────
    bar_w    = 80
    bar_h    = 8
    bar_y    = y + lh * 2 + 4
    bar_x_l  = x
    bar_x_r  = x + bar_w + 12

    _openness_bar(canvas, bar_x_l, bar_y, bar_w, bar_h,
                  analysis.mean_openness, label="")
    _text(canvas, f"open {analysis.mean_openness:.2f}",
          bar_x_l, bar_y + bar_h + 12, font, 0.42, (180, 180, 180))

    # ── Blink stats ───────────────────────────────────────────────────────────
    rate_s = f"{analysis.blink_rate_per_min:.1f}/min"
    _text(canvas, f"Blinks: {analysis.blink_count}  rate: {rate_s}",
          x, y + lh * 4, font, 0.48, (200, 200, 200))

    # ── Fatigue badge ─────────────────────────────────────────────────────────
    fatigue_colors = {"LOW": (0, 200, 80), "MODERATE": (0, 160, 255), "HIGH": (0, 50, 255)}
    fat_col = fatigue_colors.get(analysis.fatigue_label, (200, 200, 200))
    flags   = []
    if analysis.is_low_blink_rate:    flags.append("LOW-BLINK")
    if analysis.is_high_blink_rate:   flags.append("HIGH-BLINK")
    if analysis.is_prolonged_closure: flags.append("PROLONGED")
    flag_str = "  ".join(flags) if flags else "normal"
    _text(canvas, f"Fatigue: {analysis.fatigue_label}  [{flag_str}]",
          x, y + lh * 5, font, 0.48, fat_col)

    return canvas


def _text(
    img:   "np.ndarray",
    text:  str,
    x:     int,
    y:     int,
    font:  int,
    scale: float,
    color: tuple,
    thickness: int = 1,
) -> None:
    """Draw text with a thin black shadow for readability on any background."""
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0),       thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color,             thickness,     cv2.LINE_AA)


def _openness_bar(
    img:      "np.ndarray",
    x:        int,
    y:        int,
    width:    int,
    height:   int,
    fraction: float,
    label:    str,
) -> None:
    """Draw a filled progress bar representing openness in [0, 1]."""
    fraction = max(0.0, min(1.0, fraction))
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    # Fill
    fill_w = int(width * fraction)
    if fill_w > 0:
        # Colour transitions green → yellow → red with openness
        g = int(230 * fraction)
        r = int(230 * (1.0 - fraction))
        cv2.rectangle(img, (x, y), (x + fill_w, y + height), (0, g, r), -1)
    # Border
    cv2.rectangle(img, (x, y), (x + width, y + height), (120, 120, 120), 1)


def draw_head_axes(
    frame:         "np.ndarray",
    rvec:          "np.ndarray",
    tvec:          "np.ndarray",
    camera_matrix: "np.ndarray",
    *,
    axis_length:   float = 50.0,
    dist_coeffs:   "Optional[np.ndarray]" = None,
) -> "np.ndarray":
    """Draw 3-D coordinate axes projected onto the face (X=red, Y=green, Z=blue).

    The axes originate at the nose tip (the solvePnP model origin).
    axis_length is in the same units as the 3-D face model (≈ mm).
    """
    import numpy as np  # already imported at module level; local alias for clarity

    canvas = frame.copy()
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    x_end_3d  = np.array([[axis_length, 0.0,          0.0]], dtype=np.float64)
    y_end_3d  = np.array([[0.0,         axis_length,   0.0]], dtype=np.float64)
    z_end_3d  = np.array([[0.0,         0.0,  -axis_length]], dtype=np.float64)

    pts, _  = cv2.projectPoints(
        np.vstack([origin_3d, x_end_3d, y_end_3d, z_end_3d]),
        rvec, tvec, camera_matrix, dist_coeffs,
    )
    pts = pts.reshape(-1, 2).astype(int)
    o, px, py, pz = tuple(pts[0]), tuple(pts[1]), tuple(pts[2]), tuple(pts[3])

    cv2.arrowedLine(canvas, o, px, (0,   0, 220), 2, cv2.LINE_AA, tipLength=0.2)
    cv2.arrowedLine(canvas, o, py, (0, 200,   0), 2, cv2.LINE_AA, tipLength=0.2)
    cv2.arrowedLine(canvas, o, pz, (220,  0,   0), 2, cv2.LINE_AA, tipLength=0.2)
    return canvas


def draw_attention_metrics(
    frame:    "np.ndarray",
    analysis: "AttentionAnalysis",   # src.signals.attention.AttentionAnalysis
    *,
    origin:   tuple = (10, 200),
) -> "np.ndarray":
    """Draw head-pose and attention HUD panel onto a copy of *frame*."""
    from src.signals.attention import AttentionZone  # local to avoid circular

    canvas = frame.copy()
    x, y   = origin
    lh     = 22
    font   = cv2.FONT_HERSHEY_SIMPLEX

    zone_colors = {
        AttentionZone.FOCUSED:      (0, 230, 80),
        AttentionZone.GLANCE:       (0, 200, 255),
        AttentionZone.LOOKING_AWAY: (0, 50,  255),
    }
    zone_col = zone_colors.get(analysis.zone, (200, 200, 200))
    _text(canvas, f"Zone: {analysis.zone.value.upper()}", x, y, font, 0.55, zone_col)

    _text(canvas,
          f"Yaw:{analysis.yaw:+.1f}°  Pitch:{analysis.pitch:+.1f}°  Roll:{analysis.roll:+.1f}°",
          x, y + lh, font, 0.45, (200, 200, 200))

    _text(canvas,
          f"Stability  Y-std:{analysis.yaw_std:.1f}°  P-std:{analysis.pitch_std:.1f}°",
          x, y + lh * 2, font, 0.45, (200, 200, 200))

    attn_pct = analysis.attention_fraction * 100.0
    attn_col = (0, 230, 80) if attn_pct >= 70 else (0, 200, 255) if attn_pct >= 40 else (0, 50, 255)
    _text(canvas, f"Attention: {attn_pct:.0f}%", x, y + lh * 3, font, 0.48, attn_col)

    _text(canvas,
          f"Reproj err: {analysis.reprojection_error:.1f} px",
          x, y + lh * 4, font, 0.42, (160, 160, 160))

    return canvas


def draw_iris_markers(
    frame:       "np.ndarray",
    measurement: "GazeMeasurement",   # src.signals.gaze.GazeMeasurement
    *,
    radius:      int = 4,
) -> "np.ndarray":
    """Draw filled circles at each iris centre and a cross-hair at gaze centroid.

    The circle colour indicates reliability:
      cyan  — reliable (EAR adequate)
      red   — unreliable (eyes too closed)
    """
    canvas = frame.copy()
    color  = (255, 220, 0) if measurement.is_reliable else (0, 80, 220)

    lx, ly = int(measurement.left_iris_px[0]),  int(measurement.left_iris_px[1])
    rx, ry = int(measurement.right_iris_px[0]), int(measurement.right_iris_px[1])

    cv2.circle(canvas, (lx, ly), radius,     color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (lx, ly), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(canvas, (rx, ry), radius,     color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (rx, ry), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def draw_gaze_metrics(
    frame:    "np.ndarray",
    analysis: "GazeAnalysis",   # src.signals.gaze.GazeAnalysis
    *,
    origin:   tuple = (10, 380),
) -> "np.ndarray":
    """Draw gaze zone HUD panel onto a copy of *frame*."""
    from src.signals.gaze import GazeZone  # local to avoid circular

    canvas = frame.copy()
    x, y   = origin
    lh     = 22
    font   = cv2.FONT_HERSHEY_SIMPLEX

    zone_colors = {
        GazeZone.CENTER:     (0, 230, 80),
        GazeZone.UP:         (0, 200, 255),
        GazeZone.DOWN:       (0, 200, 255),
        GazeZone.LEFT:       (0, 160, 255),
        GazeZone.RIGHT:      (0, 160, 255),
        GazeZone.UP_LEFT:    (0, 100, 255),
        GazeZone.UP_RIGHT:   (0, 100, 255),
        GazeZone.DOWN_LEFT:  (0, 100, 255),
        GazeZone.DOWN_RIGHT: (0, 100, 255),
        GazeZone.UNRELIABLE: (80, 80, 80),
    }
    zone_col = zone_colors.get(analysis.zone, (200, 200, 200))
    _text(canvas, f"Gaze: {analysis.zone.value.upper().replace('_', '-')}",
          x, y, font, 0.55, zone_col)

    _text(canvas,
          f"H:{analysis.mean_h:.2f}  V:{analysis.mean_v:.2f}  "
          f"{'reliable' if analysis.is_reliable else 'unreliable'}",
          x, y + lh, font, 0.45, (200, 200, 200))

    _text(canvas,
          f"Stability  H-std:{analysis.stability_h:.3f}  V-std:{analysis.stability_v:.3f}",
          x, y + lh * 2, font, 0.45, (200, 200, 200))

    on_col = (0, 230, 80) if analysis.is_on_screen else (0, 50, 255)
    on_str = "ON SCREEN" if analysis.is_on_screen else "OFF SCREEN"
    _text(canvas, f"Screen: {on_str}", x, y + lh * 3, font, 0.50, on_col)

    frac_pct = analysis.on_screen_fraction * 100.0
    frac_col = (0, 230, 80) if frac_pct >= 70 else (0, 200, 255) if frac_pct >= 40 else (0, 50, 255)
    _text(canvas, f"On-screen: {frac_pct:.0f}%  (last 5 s)",
          x, y + lh * 4, font, 0.45, frac_col)

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
