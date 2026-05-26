"""Iris-based gaze direction estimation.

Why webcam gaze is only approximate
-------------------------------------
True gaze tracking (Tobii, SensoMotoric, etc.) uses near-infrared illumination
to locate the corneal reflection and compute the cornea-pupil vector with sub-
degree accuracy.  A plain RGB webcam gives us only the projected 2D position of
the iris within the eye region, which is affected by:

  • Head rotation (perspective changes the apparent iris position even if the
    eye is not moving — this is the biggest single error source)
  • Iris/pupil size variation with lighting
  • Eyelid occlusion when the eye is partly closed
  • Landmark noise from MediaPipe (especially at low resolution or bad lighting)
  • Individual anatomy (deep-set eyes, epicanthic folds, etc.)
  • Glasses and contact lenses

Expected accuracy: 4–10° in isolation, improving to 2–5° once head pose is
used to compensate for the largest error source.  This is sufficient to
distinguish ON_SCREEN from clearly OFF_SCREEN, and CENTER from LEFT/RIGHT at
the 20–25° level.  It cannot resolve fine within-screen gaze (e.g. which
paragraph the person is reading).

How iris position approximates gaze direction
----------------------------------------------
Six landmarks describe the eye opening: two horizontal corners and four lid
points.  Indices 468 (right eye) and 473 (left eye) are the iris centres —
the geometric centroids of the 4-point iris boundary rings.

Horizontal ratio:
    h_ratio = (iris_x − inner_corner_x) / (outer_corner_x − inner_corner_x)

    Both eyes produce a consistent ratio in image-pixel space:
      ≈ 0.0 → iris at temporal/outer extreme (gaze far to one side)
      ≈ 0.5 → iris centred horizontally (gaze roughly forward)
      ≈ 1.0 → iris at nasal/inner extreme (gaze far to other side)

Vertical ratio:
    v_ratio = (iris_y − lid_top_y) / (lid_bottom_y − lid_top_y)
      ≈ 0.0 → iris near upper lid (looking up)
      ≈ 0.5 → centred (looking ahead; eye-level screen)
      ≈ 0.7 → iris near lower lid (looking down — normal when reading)

Head pose and gaze can disagree
---------------------------------
  • Forward head + iris displaced RIGHT → person is turning head left while
    flicking eyes to the right — transitional posture, not meaningful.
  • Head RIGHT + iris centred → person is looking right-of-screen with their
    whole head (looking away entirely).
  • Head FORWARD + iris consistently off-centre → gaze signal.

The GazeDetector.update() accepts an optional head_zone string ('focused',
'glance', 'looking_away') so the caller can fold head pose into is_on_screen
without creating a circular dependency between modules.

Failure modes
-------------
  • EAR < 0.15: eyelid occludes iris; measurement is unreliable.
  • Extreme yaw (>45°): iris becomes foreshortened; ratios are unstable.
  • Face not detected: return None upstream; GazeDetector falls back to
    no_gaze_update() which returns the last valid analysis.
  • Missing iris landmarks (< 478 points): return None.

Connection to the EGRA analyzer (Day 6+)
-----------------------------------------
  Gaze adds the third spatial channel to the signal stack:
    Day 3: blink rate, EAR → ALERTNESS (temporal, eyes-open vs. closed)
    Day 4: head yaw/pitch → ORIENTATION (where the head is pointing)
    Day 5: iris ratio → GAZE (where the EYES are pointing, relative to head)
  Combined, these three signals approximate *actual visual attention*:
    on_screen = (head FOCUSED) AND (gaze CENTER) AND (eyes OPEN)
  That composite will feed the EGRA engagement score in Day 6.

MediaPipe landmark index conventions used here
------------------------------------------------
  With flip_horizontal=True (default), the frame given to MediaPipe is a
  mirrored selfie view.  MediaPipe labels landmarks from the SUBJECT'S
  perspective.  The mapping to image coordinates is:

    Subject's LEFT eye  → LEFT side of flipped image
      iris center:        473
      h_box:              outer=263 (leftmost), inner=362 (rightmost)
      v_box:              top=386, bottom=374

    Subject's RIGHT eye → RIGHT side of flipped image
      iris center:        468
      h_box:              inner=133 (leftmost), outer=33 (rightmost)
      v_box:              top=159, bottom=145

  h_ratio = (iris_x - left_px) / (right_px - left_px) is then consistent:
    both eyes' ratios decrease when the person looks to the viewer's LEFT and
    increase when looking to the viewer's RIGHT.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ── Iris and eye-box landmark indices ────────────────────────────────────────
# Subject's LEFT eye (LEFT side of flipped image)
_L_IRIS   = 473
_L_H_LEFT = 263   # outer/temporal corner — leftmost in image
_L_H_RIGHT = 362  # inner/nasal corner   — rightmost in image
_L_V_TOP   = 386
_L_V_BOT   = 374

# Subject's RIGHT eye (RIGHT side of flipped image)
_R_IRIS    = 468
_R_H_LEFT  = 133  # inner/nasal corner   — leftmost in image
_R_H_RIGHT = 33   # outer/temporal corner — rightmost in image
_R_V_TOP   = 159
_R_V_BOT   = 145

# Minimum landmark count for iris indices to be valid
IRIS_MIN_LANDMARKS = 478

# Minimum EAR for reliable iris measurement (below this, eyelid occludes iris)
_EAR_RELIABLE_MIN = 0.15


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GazeMeasurement:
    """Raw per-frame iris position — no behavioral inference.

    h_ratio: 0.0 = iris at image-left extreme, 0.5 = centred, 1.0 = image-right
    v_ratio: 0.0 = iris near upper lid (looking up), 1.0 = near lower lid (down)
    is_reliable: False when EAR is too low to trust the iris position.
    """
    left_h:      float   # left eye horizontal iris ratio
    left_v:      float   # left eye vertical iris ratio
    right_h:     float   # right eye horizontal iris ratio
    right_v:     float   # right eye vertical iris ratio
    mean_h:      float   # average horizontal ratio
    mean_v:      float   # average vertical ratio
    left_iris_px:  tuple   # (x, y) pixel coords of left iris centre
    right_iris_px: tuple   # (x, y) pixel coords of right iris centre
    is_reliable: bool    # False when eyes are too closed to trust iris position
    timestamp:   float
    frame_index: int


class GazeZone(str, Enum):
    CENTER     = "center"
    LEFT       = "left"
    RIGHT      = "right"
    UP         = "up"
    DOWN       = "down"
    UP_LEFT    = "up_left"
    UP_RIGHT   = "up_right"
    DOWN_LEFT  = "down_left"
    DOWN_RIGHT = "down_right"
    UNRELIABLE = "unreliable"   # eyes too closed or landmarks absent


@dataclass
class GazeAnalysis:
    zone:               GazeZone
    mean_h:             float     # iris horizontal ratio [0, 1]
    mean_v:             float     # iris vertical ratio [0, 1]
    is_on_screen:       bool      # gaze + head pose heuristic
    is_reliable:        bool      # False when eyes too closed
    stability_h:        float     # rolling std dev of horizontal ratio
    stability_v:        float     # rolling std dev of vertical ratio
    on_screen_fraction: float     # proportion of recent frames classified on-screen
    timestamp:          float
    frame_index:        int


# ── Raw measurement extraction ─────────────────────────────────────────────────

def extract_gaze_measurements(
    landmarks:   list,
    width:       int,
    height:      int,
    *,
    mean_ear:    float = 1.0,
    timestamp:   float = 0.0,
    frame_index: int   = 0,
    cfg:         Optional[dict] = None,
) -> Optional[GazeMeasurement]:
    """Extract iris-position ratios from a MediaPipe 478-point landmark list.

    Returns None if fewer than 478 landmarks are present (iris indices are
    only valid in the full 478-point model).

    mean_ear is used to set the is_reliable flag.  Pass the value from
    EyeMeasurement.mean_ear so the gaze module does not need to recompute EAR.
    """
    if len(landmarks) < IRIS_MIN_LANDMARKS:
        return None

    ear_min = float((cfg or {}).get("ear_reliable_min", _EAR_RELIABLE_MIN))
    reliable = mean_ear >= ear_min

    lx = landmarks[_L_IRIS].x * width
    ly = landmarks[_L_IRIS].y * height
    rx = landmarks[_R_IRIS].x * width
    ry = landmarks[_R_IRIS].y * height

    left_h  = _ratio(lx, landmarks[_L_H_LEFT].x * width,  landmarks[_L_H_RIGHT].x * width)
    left_v  = _ratio(ly, landmarks[_L_V_TOP].y  * height, landmarks[_L_V_BOT].y   * height)
    right_h = _ratio(rx, landmarks[_R_H_LEFT].x * width,  landmarks[_R_H_RIGHT].x * width)
    right_v = _ratio(ry, landmarks[_R_V_TOP].y  * height, landmarks[_R_V_BOT].y   * height)

    return GazeMeasurement(
        left_h=left_h, left_v=left_v,
        right_h=right_h, right_v=right_v,
        mean_h=(left_h + right_h) / 2.0,
        mean_v=(left_v + right_v) / 2.0,
        left_iris_px=(lx, ly),
        right_iris_px=(rx, ry),
        is_reliable=reliable,
        timestamp=timestamp,
        frame_index=frame_index,
    )


# ── Stateful detector ─────────────────────────────────────────────────────────

class GazeDetector:
    """Accumulates gaze measurements and produces per-frame GazeAnalysis."""

    def __init__(self, cfg: dict) -> None:
        g = cfg.get("gaze", {})
        self._h_lo  = float(g.get("center_h_lo", 0.35))
        self._h_hi  = float(g.get("center_h_hi", 0.65))
        self._v_lo  = float(g.get("center_v_lo", 0.25))
        self._v_hi  = float(g.get("center_v_hi", 0.70))
        # Head pose yaw/pitch thresholds for the is_on_screen override
        self._head_yaw_max   = float(g.get("head_yaw_max_on_screen",   25.0))
        self._head_pitch_max = float(g.get("head_pitch_max_on_screen", 20.0))
        self._window_s       = float(g.get("stability_window_s",        5.0))

        # Rolling history: (timestamp, mean_h, mean_v, is_on_screen)
        self._history: deque[tuple[float, float, float, bool]] = deque()
        self._last: Optional[GazeAnalysis] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        m: GazeMeasurement,
        *,
        head_zone: Optional[str] = None,
        head_yaw:  Optional[float] = None,
        head_pitch: Optional[float] = None,
    ) -> GazeAnalysis:
        """Process one GazeMeasurement and return GazeAnalysis.

        head_zone: 'focused'|'glance'|'looking_away' — from AttentionAnalysis.zone.value
        head_yaw, head_pitch: degrees — from PoseMeasurement.  If provided, used
            to gate is_on_screen even when iris is centred (head is turned away).
        """
        if not m.is_reliable:
            zone = GazeZone.UNRELIABLE
        else:
            zone = self._classify(m.mean_h, m.mean_v)

        on_screen = self._compute_on_screen(zone, head_zone, head_yaw, head_pitch)

        self._history.append((m.timestamp, m.mean_h, m.mean_v, on_screen))
        self._prune(m.timestamp)

        std_h, std_v = self._compute_stds()
        on_frac      = self._on_screen_fraction()

        analysis = GazeAnalysis(
            zone=zone,
            mean_h=m.mean_h, mean_v=m.mean_v,
            is_on_screen=on_screen,
            is_reliable=m.is_reliable,
            stability_h=std_h, stability_v=std_v,
            on_screen_fraction=on_frac,
            timestamp=m.timestamp,
            frame_index=m.frame_index,
        )
        self._last = analysis
        return analysis

    def no_gaze_update(self) -> Optional[GazeAnalysis]:
        """Return last analysis without advancing state (face/iris absent)."""
        return self._last

    def reset(self) -> None:
        self._history.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _classify(self, h: float, v: float) -> GazeZone:
        h_c = self._h_lo <= h <= self._h_hi
        v_c = self._v_lo <= v <= self._v_hi
        if h_c and v_c:
            return GazeZone.CENTER
        left  = h < self._h_lo
        right = h > self._h_hi
        up    = v < self._v_lo
        down  = v > self._v_hi
        if left  and up:   return GazeZone.UP_LEFT
        if left  and down: return GazeZone.DOWN_LEFT
        if right and up:   return GazeZone.UP_RIGHT
        if right and down: return GazeZone.DOWN_RIGHT
        if left:           return GazeZone.LEFT
        if right:          return GazeZone.RIGHT
        if up:             return GazeZone.UP
        return GazeZone.DOWN

    def _compute_on_screen(
        self,
        zone:        GazeZone,
        head_zone:   Optional[str],
        head_yaw:    Optional[float],
        head_pitch:  Optional[float],
    ) -> bool:
        if zone not in (GazeZone.CENTER, GazeZone.UP, GazeZone.DOWN):
            return False
        # If head is explicitly looking away, trust that over iris signal
        if head_zone == "looking_away":
            return False
        if head_yaw is not None and abs(head_yaw) > self._head_yaw_max:
            return False
        if head_pitch is not None and abs(head_pitch) > self._head_pitch_max:
            return False
        return True

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _compute_stds(self) -> tuple[float, float]:
        if len(self._history) < 2:
            return 0.0, 0.0
        hs = [e[1] for e in self._history]
        vs = [e[2] for e in self._history]
        return _std(hs), _std(vs)

    def _on_screen_fraction(self) -> float:
        if not self._history:
            return 0.0
        return sum(1 for e in self._history if e[3]) / len(self._history)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ratio(val: float, lo: float, hi: float) -> float:
    """Normalise val into [lo, hi], clamped to [0, 1]."""
    span = hi - lo
    if abs(span) < 1e-3:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / span))


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
