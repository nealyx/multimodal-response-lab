"""Attention and gaze-zone classification from head pose.

AttentionDetector ingests PoseMeasurement objects and produces AttentionAnalysis
on each update.  It maintains a rolling history of yaw/pitch values to compute
orientation stability (std dev) and an attention fraction (proportion of recent
frames spent in the FOCUSED zone).

Zone classification
-------------------
  FOCUSED       : |yaw| ≤ yaw_focus_deg  AND  |pitch| ≤ pitch_focus_deg
  GLANCE        : outside FOCUSED but within ±glance_deg
  LOOKING_AWAY  : beyond glance thresholds

Stability metric
----------------
yaw_std and pitch_std are the sample standard deviations over the last
stability_window_s seconds.  High std dev → restless / distracted head
movement.  Low std dev while LOOKING_AWAY → consistently off-screen.

Attention fraction
------------------
Proportion of frames in the rolling window that were classified as FOCUSED.
Provides a [0, 1] summary suitable for downstream scoring.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.signals.head_pose import PoseMeasurement


# ── Enums and data structures ─────────────────────────────────────────────────

class AttentionZone(str, Enum):
    FOCUSED      = "focused"
    GLANCE       = "glance"
    LOOKING_AWAY = "looking_away"


@dataclass
class AttentionAnalysis:
    zone:              AttentionZone
    yaw:               float     # degrees
    pitch:             float     # degrees
    roll:              float     # degrees
    yaw_std:           float     # rolling std dev (degrees)
    pitch_std:         float     # rolling std dev (degrees)
    attention_fraction: float    # [0, 1] — proportion FOCUSED in window
    reprojection_error: float
    timestamp:         float
    frame_index:       int


# ── Detector ──────────────────────────────────────────────────────────────────

class AttentionDetector:
    """Stateful accumulator for head-pose attention metrics."""

    def __init__(self, cfg: dict) -> None:
        hp_cfg = cfg.get("head_pose", {})
        att_cfg = hp_cfg.get("attention", {})

        self._yaw_focus   = float(att_cfg.get("yaw_focus_deg",   20.0))
        self._pitch_focus = float(att_cfg.get("pitch_focus_deg", 15.0))
        self._yaw_glance  = float(att_cfg.get("yaw_glance_deg",  35.0))
        self._pitch_glance = float(att_cfg.get("pitch_glance_deg", 25.0))
        self._stability_window_s = float(att_cfg.get("stability_window_s", 5.0))

        # Rolling history: each entry is (timestamp, yaw, pitch, zone)
        self._history: deque[tuple[float, float, float, AttentionZone]] = deque()
        self._last_analysis: Optional[AttentionAnalysis] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, pose: PoseMeasurement) -> AttentionAnalysis:
        """Process one PoseMeasurement and return AttentionAnalysis."""
        zone = self._classify(pose.yaw, pose.pitch)

        self._history.append((pose.timestamp, pose.yaw, pose.pitch, zone))
        self._prune(pose.timestamp)

        yaw_std, pitch_std = self._compute_stds()
        attention_fraction = self._compute_attention_fraction()

        analysis = AttentionAnalysis(
            zone=zone,
            yaw=pose.yaw,
            pitch=pose.pitch,
            roll=pose.roll,
            yaw_std=yaw_std,
            pitch_std=pitch_std,
            attention_fraction=attention_fraction,
            reprojection_error=pose.reprojection_error,
            timestamp=pose.timestamp,
            frame_index=pose.frame_index,
        )
        self._last_analysis = analysis
        return analysis

    def no_pose_update(self) -> Optional[AttentionAnalysis]:
        """Return last analysis without advancing state (face/pose missing)."""
        return self._last_analysis

    def reset(self) -> None:
        """Clear rolling history; last_analysis is preserved."""
        self._history.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _classify(self, yaw: float, pitch: float) -> AttentionZone:
        abs_yaw   = abs(yaw)
        abs_pitch = abs(pitch)
        if abs_yaw <= self._yaw_focus and abs_pitch <= self._pitch_focus:
            return AttentionZone.FOCUSED
        if abs_yaw <= self._yaw_glance and abs_pitch <= self._pitch_glance:
            return AttentionZone.GLANCE
        return AttentionZone.LOOKING_AWAY

    def _prune(self, now: float) -> None:
        cutoff = now - self._stability_window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _compute_stds(self) -> tuple[float, float]:
        if len(self._history) < 2:
            return 0.0, 0.0
        yaws   = [e[1] for e in self._history]
        pitches = [e[2] for e in self._history]
        return _std(yaws), _std(pitches)

    def _compute_attention_fraction(self) -> float:
        if not self._history:
            return 0.0
        focused = sum(1 for e in self._history if e[3] == AttentionZone.FOCUSED)
        return focused / len(self._history)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)
