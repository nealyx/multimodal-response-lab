"""Raw eye measurement extraction: EAR and openness score.

This module computes *measurements* — quantities derived directly from landmark
geometry with no behavioral interpretation.  All state inference (is this a
blink? is the person drowsy?) lives in blink_detector.py.  Keeping the layers
separate means measurements can be logged, replayed, and tested independently
of any downstream state logic.

EAR — Eye Aspect Ratio
-----------------------
Proposed by Soukupová and Čech (2016) for real-time blink detection.

For each eye, six landmark points are selected:
    p1   p2  p3   p4
    ●────●───●────●      ← horizontal endpoints + upper lid
         |   |
    ●────●───●────●      ← lower lid
    p6   p5  (reuse p1/p4 for corners)

Formula:
    EAR = ( ‖p2 − p6‖ + ‖p3 − p5‖ ) / ( 2 · ‖p1 − p4‖ )

Numerator: sum of two vertical (lid separation) distances.
Denominator: twice the horizontal (eye width) distance.

The ratio cancels scale, so EAR is invariant to head size and camera distance.
Open eye typical range: 0.25 – 0.35
Closed eye typical range: 0.0 – 0.15
The gap makes 0.20 a stable detection threshold.

MediaPipe landmark indices
--------------------------
Indices below come from the canonical 478-point Face Mesh topology.
The outer/inner corner naming assumes a front-facing mirrored view.

    Left eye (viewer's left, subject's right):
        p1=33  p2=160  p3=158  p4=133  p5=153  p6=144

    Right eye:
        p1=362  p2=385  p3=387  p4=263  p5=373  p6=380

Using six points rather than a simple top/bottom pair captures the
asymmetric shape of the eyelid and improves stability near the corners.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

# ── EAR landmark indices (MediaPipe 478-point model) ─────────────────────────

LEFT_EYE_EAR_IDX  = (33,  160, 158, 133, 153, 144)   # p1…p6
RIGHT_EYE_EAR_IDX = (362, 385, 387, 263, 373, 380)

# Reference EAR values for normalising to a [0, 1] openness score.
# These are population averages; per-user calibration would improve accuracy.
EAR_CLOSED_REF: float = 0.15
EAR_OPEN_REF:   float = 0.35


# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EyeMeasurement:
    """Raw per-frame eye measurements — no behavioral inference.

    All EAR values are in [0.0, ~0.4]; all openness scores in [0.0, 1.0].
    """
    left_ear:       float
    right_ear:      float
    mean_ear:       float
    left_openness:  float   # normalised to [0 = closed, 1 = fully open]
    right_openness: float
    mean_openness:  float
    timestamp:      float   # time.perf_counter() at the point of measurement
    frame_index:    int


# ── Core computation ──────────────────────────────────────────────────────────

def compute_ear(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
    p5: tuple[float, float],
    p6: tuple[float, float],
) -> float:
    """Compute Eye Aspect Ratio from six (x, y) pixel-space points.

        EAR = ( ‖p2 − p6‖ + ‖p3 − p5‖ ) / ( 2 · ‖p1 − p4‖ )

    Returns 0.0 if the horizontal distance is degenerate (< 1e-6 px),
    which happens when the face is nearly edge-on to the camera.
    """
    vertical_a  = _dist(p2, p6)
    vertical_b  = _dist(p3, p5)
    horizontal  = _dist(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return (vertical_a + vertical_b) / (2.0 * horizontal)


def ear_to_openness(
    ear:        float,
    closed_ref: float = EAR_CLOSED_REF,
    open_ref:   float = EAR_OPEN_REF,
) -> float:
    """Map an EAR value to a normalised [0.0, 1.0] openness score.

    Values outside the [closed_ref, open_ref] range are clamped.
    """
    if open_ref <= closed_ref:
        return 0.0
    return max(0.0, min(1.0, (ear - closed_ref) / (open_ref - closed_ref)))


def extract_eye_measurements(
    landmarks:   list,    # list[NormalizedLandmark] for one face
    width:       int,
    height:      int,
    timestamp:   float,
    frame_index: int,
    cfg:         Optional[dict] = None,
) -> Optional[EyeMeasurement]:
    """Extract EAR and openness for both eyes from a MediaPipe landmark list.

    Returns None if the landmark list is too short (< 468 points), which
    can happen on the first few frames or when the face is partially occluded.

    cfg may contain:
        ear_closed_ref: float  (default EAR_CLOSED_REF)
        ear_open_ref:   float  (default EAR_OPEN_REF)
    """
    if len(landmarks) < 468:
        return None

    c_ref = (cfg or {}).get("ear_closed_ref", EAR_CLOSED_REF)
    o_ref = (cfg or {}).get("ear_open_ref",   EAR_OPEN_REF)

    left_pts  = [_lm_to_px(landmarks, i, width, height) for i in LEFT_EYE_EAR_IDX]
    right_pts = [_lm_to_px(landmarks, i, width, height) for i in RIGHT_EYE_EAR_IDX]

    left_ear  = compute_ear(*left_pts)
    right_ear = compute_ear(*right_pts)
    mean_ear  = (left_ear + right_ear) / 2.0

    return EyeMeasurement(
        left_ear=       left_ear,
        right_ear=      right_ear,
        mean_ear=       mean_ear,
        left_openness=  ear_to_openness(left_ear,  c_ref, o_ref),
        right_openness= ear_to_openness(right_ear, c_ref, o_ref),
        mean_openness=  ear_to_openness(mean_ear,  c_ref, o_ref),
        timestamp=      timestamp,
        frame_index=    frame_index,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _lm_to_px(
    landmarks: list,
    idx:       int,
    width:     int,
    height:    int,
) -> tuple[float, float]:
    lm = landmarks[idx]
    return lm.x * width, lm.y * height
