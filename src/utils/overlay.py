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


def draw_engagement_dashboard(
    frame:    "np.ndarray",
    score:    "EngagementScore",   # src.signals.engagement.EngagementScore
    blink:    "Optional[BlinkAnalysis]" = None,
    *,
    origin:   tuple = (10, 10),
    bar_width: int = 160,
) -> "np.ndarray":
    """Draw a composite engagement dashboard panel onto a copy of *frame*.

    Panel layout (top-down at *origin*):
      - State badge with colour coding
      - Composite score + confidence progress bars
      - Component sub-bars: gaze / head / eye
      - Rolling focused % and engaged %
      - Blink rate and fatigue label (when blink is provided)
    """
    from src.signals.engagement import EngagementState  # local to avoid circular

    canvas = frame.copy()
    x, y   = origin
    lh     = 20
    font   = cv2.FONT_HERSHEY_SIMPLEX

    # ── State badge ──────────────────────────────────────────────────────────
    state_colors = {
        EngagementState.FOCUSED:    (0, 220, 80),
        EngagementState.DRIFTING:   (0, 200, 255),
        EngagementState.DISTRACTED: (0, 80,  255),
        EngagementState.FATIGUED:   (0, 50,  200),
        EngagementState.UNRELIABLE: (80, 80,  80),
    }
    state_col = state_colors.get(score.state, (200, 200, 200))
    label     = score.state.value.upper()
    _text(canvas, f"Engagement: {label}", x, y + lh, font, 0.58, state_col, thickness=2)

    # ── Composite score bar ───────────────────────────────────────────────────
    row = y + lh * 2 + 4
    _text(canvas, f"Score  {score.smoothed_score:.2f}", x, row, font, 0.45, (200, 200, 200))
    _score_bar(canvas, x + 100, row - 12, bar_width, 10, score.smoothed_score,
               lo_color=(0, 60, 230), hi_color=(0, 200, 60))

    # ── Confidence bar ────────────────────────────────────────────────────────
    row += lh
    conf_col = (0, 200, 60) if score.confidence >= 0.6 else (0, 160, 230) if score.confidence >= 0.3 else (80, 80, 80)
    _text(canvas, f"Conf   {score.confidence:.2f}", x, row, font, 0.45, conf_col)
    _score_bar(canvas, x + 100, row - 12, bar_width, 10, score.confidence,
               lo_color=(60, 60, 80), hi_color=(0, 200, 160))

    # ── Component bars ────────────────────────────────────────────────────────
    row += lh + 4
    _text(canvas, "-- components --", x, row, font, 0.38, (120, 120, 120))

    components = [
        ("Gaze ", score.gaze_score),
        ("Head ", score.head_score),
        ("Eye  ", score.eye_score),
    ]
    for label_c, val in components:
        row += lh - 2
        _text(canvas, f"{label_c} {val:.2f}", x, row, font, 0.42, (180, 180, 180))
        _score_bar(canvas, x + 100, row - 11, bar_width, 8, val,
                   lo_color=(0, 60, 230), hi_color=(0, 200, 80))

    # ── Rolling fractions ─────────────────────────────────────────────────────
    row += lh + 2
    f_pct = score.focused_fraction  * 100.0
    e_pct = score.engaged_fraction  * 100.0
    f_col = (0, 220, 80) if f_pct >= 70 else (0, 200, 255) if f_pct >= 40 else (0, 60, 255)
    _text(canvas, f"Focused {f_pct:.0f}%  Engaged {e_pct:.0f}%",
          x, row, font, 0.42, f_col)

    # ── Blink / fatigue (optional) ────────────────────────────────────────────
    if blink is not None:
        row += lh
        fatigue_colors = {"LOW": (0, 200, 80), "MODERATE": (0, 160, 255), "HIGH": (0, 50, 255)}
        fat_col  = fatigue_colors.get(blink.fatigue_label, (200, 200, 200))
        _text(canvas, f"Blink {blink.blink_rate_per_min:.1f}/min  [{blink.fatigue_label}]",
              x, row, font, 0.42, fat_col)

    # ── Active flags ──────────────────────────────────────────────────────────
    flags = []
    if score.is_fatigued:      flags.append("FATIGUE")
    if score.is_looking_away:  flags.append("AWAY")
    if not score.is_gaze_reliable:  flags.append("IRIS?")
    if flags:
        row += lh
        _text(canvas, "Flags: " + "  ".join(flags), x, row, font, 0.40, (0, 120, 255))

    return canvas


def _score_bar(
    img:       "np.ndarray",
    x:         int,
    y:         int,
    width:     int,
    height:    int,
    fraction:  float,
    *,
    lo_color:  tuple = (0, 60, 230),
    hi_color:  tuple = (0, 200, 60),
) -> None:
    """Draw a horizontal progress bar that blends between lo_color and hi_color."""
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(img, (x, y), (x + width, y + height), (40, 40, 40), -1)
    fill_w = int(width * fraction)
    if fill_w > 0:
        r = int(lo_color[2] + (hi_color[2] - lo_color[2]) * fraction)
        g = int(lo_color[1] + (hi_color[1] - lo_color[1]) * fraction)
        b = int(lo_color[0] + (hi_color[0] - lo_color[0]) * fraction)
        cv2.rectangle(img, (x, y), (x + fill_w, y + height), (b, g, r), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)


def draw_demo_overlay(
    frame:    "np.ndarray",
    score:    "EngagementScore",
    blink:    "Optional[BlinkAnalysis]" = None,
    *,
    origin:   tuple = (10, 10),
    width:    int   = 260,
) -> "np.ndarray":
    """Demo / presentation overlay — minimal, readable at a glance.

    Shows only the metrics a presenter or stakeholder needs:
      Top    : state badge + attention score bar
      Middle : confidence (or "Calibrating" during warmup)
      Lower  : gaze zone + on-screen status
      Bottom : blink rate + fatigue label

    Hides: raw EAR, H/V ratios, yaw/pitch/roll, reprojection error,
           stability stats, component sub-scores.
    """
    from src.signals.engagement import EngagementState  # local to avoid circular

    canvas = frame.copy()
    x, y   = origin
    lh     = 26
    font   = cv2.FONT_HERSHEY_SIMPLEX

    # Card background
    pad = 10
    card_h = lh * 7 + pad * 2 + 4
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - pad, y - pad),
                  (x + width + pad, y + card_h), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(canvas, (x - pad, y - pad),
                  (x + width + pad, y + card_h), (50, 50, 80), 1)

    # ── State badge (large) ───────────────────────────────────────────────────
    state_colors = {
        EngagementState.FOCUSED:    (0, 220, 80),
        EngagementState.DRIFTING:   (0, 200, 255),
        EngagementState.DISTRACTED: (0, 80,  255),
        EngagementState.FATIGUED:   (0, 50,  200),
        EngagementState.UNRELIABLE: (80, 80,  80),
    }
    state_col = state_colors.get(score.state, (200, 200, 200))
    _text(canvas, score.state.value.upper(), x, y + lh, font, 0.75, state_col, thickness=2)

    # ── Attention score bar ───────────────────────────────────────────────────
    row = y + lh + 10
    pct = int(score.smoothed_score * 100)
    score_col = (0, 220, 80) if pct >= 72 else (0, 200, 255) if pct >= 45 else (0, 80, 255)
    _text(canvas, f"Attention  {pct}%", x, row + lh, font, 0.50, score_col)
    _score_bar(canvas, x, row + lh + 4, width, 8, score.smoothed_score,
               lo_color=(0, 60, 230), hi_color=(0, 200, 60))

    # ── Confidence / calibrating ──────────────────────────────────────────────
    row += lh + 20
    cb = score.confidence_breakdown
    if cb.warmup_remaining_s > 0.5:
        remaining = int(cb.warmup_remaining_s)
        _text(canvas, f"Calibrating  ({remaining}s)", x, row + lh, font, 0.45, (120, 160, 200))
        _score_bar(canvas, x, row + lh + 4, width, 6, cb.history_warmup,
                   lo_color=(40, 40, 80), hi_color=(80, 140, 220))
    else:
        conf_pct = int(score.confidence * 100)
        conf_col = (0, 200, 60) if conf_pct >= 70 else (0, 180, 255) if conf_pct >= 40 else (80, 80, 80)
        _text(canvas, f"Confidence  {conf_pct}%", x, row + lh, font, 0.50, conf_col)
        _score_bar(canvas, x, row + lh + 4, width, 6, score.confidence,
                   lo_color=(60, 60, 80), hi_color=(0, 200, 160))

    # ── Divider ───────────────────────────────────────────────────────────────
    row += lh + 22
    cv2.line(canvas, (x, row), (x + width, row), (50, 50, 80), 1)
    row += 8

    # ── Gaze + screen status ──────────────────────────────────────────────────
    on_col = (0, 220, 80) if score.is_gaze_reliable else (80, 80, 80)
    _text(canvas, "Gaze", x, row + lh, font, 0.45, (160, 160, 160))
    # We don't have gaze zone directly on EngagementScore; use flags
    gaze_label = "ON SCREEN" if not score.is_looking_away else "OFF SCREEN"
    gaze_col   = (0, 220, 80) if not score.is_looking_away else (0, 80, 255)
    _text(canvas, gaze_label, x + 60, row + lh, font, 0.50, gaze_col)

    row += lh + 4
    # Head zone derived from is_looking_away flag
    head_label = "FOCUSED" if not score.is_looking_away else "AWAY"
    head_col   = (0, 220, 80) if not score.is_looking_away else (0, 80, 255)
    _text(canvas, "Head", x, row + lh, font, 0.45, (160, 160, 160))
    _text(canvas, head_label, x + 60, row + lh, font, 0.50, head_col)

    # ── Divider ───────────────────────────────────────────────────────────────
    row += lh + 4
    cv2.line(canvas, (x, row), (x + width, row), (50, 50, 80), 1)
    row += 8

    # ── Blink rate + fatigue ──────────────────────────────────────────────────
    if blink is not None:
        fatigue_colors = {"LOW": (0, 200, 80), "MODERATE": (0, 160, 255), "HIGH": (0, 50, 255)}
        fat_col = fatigue_colors.get(blink.fatigue_label, (200, 200, 200))
        if blink.has_sufficient_data:
            rate_str = f"{blink.blink_rate_per_min:.0f}/min"
        else:
            rate_str = "--/min"
        _text(canvas, "Blink", x, row + lh, font, 0.45, (160, 160, 160))
        _text(canvas, rate_str, x + 60, row + lh, font, 0.50, (200, 200, 200))
        _text(canvas, blink.fatigue_label, x + 140, row + lh, font, 0.50, fat_col)
    else:
        _text(canvas, "Blink  --/min", x, row + lh, font, 0.45, (100, 100, 100))

    return canvas


def draw_diagnostics_panel(
    frame:  "np.ndarray",
    score:  "EngagementScore",
    blink:  "Optional[BlinkAnalysis]" = None,
    *,
    origin: tuple = (10, 340),
) -> "np.ndarray":
    """Debug-only diagnostics panel: confidence breakdown + signal health.

    Shows WHY confidence is low. Typical cause: history_warmup is < 1.0
    because the session has been running for fewer seconds than min_history_s.
    Example: Attention=100%, Score=0.86, Confidence=0.41 means 4.1 s / 10 s
    elapsed — not a signal quality problem, just a warmup ramp.

    Visible only when the run script is invoked with --debug.
    """
    canvas = frame.copy()
    x, y   = origin
    lh     = 17
    font   = cv2.FONT_HERSHEY_SIMPLEX

    _text(canvas, "── Diagnostics ──────────────────", x, y, font, 0.38, (80, 140, 220))

    cb  = score.confidence_breakdown
    row = y + lh

    # History warmup row
    w_pct = int(cb.history_warmup * 100)
    w_col = (0, 200, 80) if w_pct >= 90 else (0, 180, 255) if w_pct >= 50 else (80, 80, 80)
    _text(canvas, f"Warmup  {w_pct:3d}%", x, row, font, 0.38, w_col)
    _score_bar(canvas, x + 100, row - 11, 80, 7, cb.history_warmup,
               lo_color=(40, 40, 80), hi_color=(80, 160, 220))
    if cb.warmup_remaining_s > 0.5:
        _text(canvas, f"{cb.warmup_remaining_s:.0f}s left", x + 190, row, font, 0.34, (100, 100, 130))

    row += lh
    gq_pct = int(cb.gaze_quality * 100)
    gq_col = (0, 200, 80) if gq_pct == 100 else (0, 160, 255)
    _text(canvas, f"Gaze Q  {gq_pct:3d}%", x, row, font, 0.38, gq_col)
    _score_bar(canvas, x + 100, row - 11, 80, 7, cb.gaze_quality,
               lo_color=(0, 60, 200), hi_color=(0, 200, 80))
    gaze_note = "reliable" if cb.gaze_quality >= 1.0 else "iris unreliable"
    _text(canvas, gaze_note, x + 190, row, font, 0.34, (100, 100, 130))

    row += lh
    pq_pct = int(cb.pose_quality * 100)
    pq_col = (0, 200, 80) if pq_pct >= 90 else (0, 160, 255)
    _text(canvas, f"Pose Q  {pq_pct:3d}%", x, row, font, 0.38, pq_col)
    _score_bar(canvas, x + 100, row - 11, 80, 7, cb.pose_quality,
               lo_color=(0, 60, 200), hi_color=(0, 200, 80))

    row += lh
    ov_pct  = int(cb.overall * 100)
    ov_col  = (0, 200, 80) if ov_pct >= 70 else (0, 160, 255) if ov_pct >= 40 else (80, 80, 80)
    _text(canvas, f"Conf    {ov_pct:3d}%", x, row, font, 0.40, ov_col)
    _score_bar(canvas, x + 100, row - 11, 80, 7, cb.overall,
               lo_color=(60, 60, 80), hi_color=(0, 200, 160))
    _text(canvas, f"= warmup×gaze×pose", x + 190, row, font, 0.32, (80, 80, 100))

    # Blink data quality row
    if blink is not None:
        row += lh
        if blink.has_sufficient_data:
            bd_col = (0, 200, 80)
            bd_txt = f"Blink  {blink.blink_rate_per_min:.1f}/min  data OK"
        else:
            bd_col = (0, 160, 255)
            bd_txt = "Blink  warming up"
        _text(canvas, bd_txt, x, row, font, 0.38, bd_col)

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
