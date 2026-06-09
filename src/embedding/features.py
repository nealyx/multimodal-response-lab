"""Sliding-window feature extraction from behavioral time series.

What a behavioral embedding is
--------------------------------
A behavioral embedding is a fixed-length numeric vector that summarizes
one window of behavioral signals — engagement score, gaze, head orientation,
blink patterns — into a representation suitable for machine learning.

Why fixed-length windows?
--------------------------
Session timelines are variable-length time series: some sessions are 2 minutes,
some are 10.  Most ML algorithms (clustering, PCA, anomaly detection) expect
fixed-length inputs.  A sliding window converts the variable-length series into
a sequence of fixed-dimension vectors, each capturing one behavioral "state" at
a specific point in the session.

Window aggregation strategy (why not use raw frames):
  - A single frame is too noisy — any individual frame can be a blink, a transient
    head movement, a landmark instability artifact.
  - A window of N frames produces aggregate statistics (mean, std, fractions)
    that are inherently smoothed and stable.
  - The resulting feature vector captures BEHAVIOR over a time interval, not
    instantaneous measurement noise.

Feature groups
--------------
  Engagement (6): score distribution + state fractions
  Gaze (4):       on-screen fraction + iris centering + gaze stability
  Head (4):       zone fraction + mean orientation + rotational stability
  Blink/eye (4):  rate + EAR + fatigue + rate variability
  Total: 18 normalized features ∈ [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.stimulus.events import BehavioralSample


# ── Feature schema ────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    # ── Engagement ──────────────────────────────────────────────────────────
    "engagement_mean",    # mean(smoothed_score)           already [0,1]
    "engagement_std",     # std(smoothed_score)            volatility
    "focused_frac",       # fraction of frames FOCUSED
    "distracted_frac",    # fraction DISTRACTED
    "fatigued_frac",      # fraction FATIGUED
    "unreliable_frac",    # fraction UNRELIABLE (data-quality proxy)
    # ── Gaze ────────────────────────────────────────────────────────────────
    "on_screen_frac",     # fraction is_on_screen
    "gaze_h_mean",        # mean iris horizontal position  already [0,1]
    "gaze_h_std",         # std of gaze_h                  stability proxy
    "gaze_v_std",         # std of gaze_v
    # ── Head pose ────────────────────────────────────────────────────────────
    "head_focused_frac",  # fraction head_zone == "focused"
    "head_yaw_mean",      # mean head yaw, normalized from [-45°, +45°]
    "head_yaw_std",       # yaw std-dev, normalized from [0°, 30°]
    "head_pitch_std",     # pitch std-dev, normalized from [0°, 30°]
    # ── Blink / eye ──────────────────────────────────────────────────────────
    "blink_rate_mean",    # mean blink rate, normalized from [0, 40/min]
    "ear_mean",           # mean EAR                       already [0,1]
    "fatigue_frac",       # fraction is_fatigued
    "blink_rate_std",     # rate variability, normalized from [0, 20/min]
]
N_FEATURES: int = len(FEATURE_NAMES)   # 18

# Normalization ranges for features that are not already in [0, 1]
# (lo, hi) → clamp to [lo, hi], then rescale to [0, 1]
_SCALE: Dict[str, tuple] = {
    "engagement_std":   (0.0,  0.5),
    "gaze_h_std":       (0.0,  0.3),
    "gaze_v_std":       (0.0,  0.3),
    "head_yaw_mean":    (-45.0, 45.0),
    "head_yaw_std":     (0.0,  30.0),
    "head_pitch_std":   (0.0,  30.0),
    "blink_rate_mean":  (0.0,  40.0),
    "blink_rate_std":   (0.0,  20.0),
}


def _norm(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi] and rescale to [0, 1]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


# ── BehavioralWindow ──────────────────────────────────────────────────────────

@dataclass
class BehavioralWindow:
    """A fixed-length feature vector extracted from a time window of frames.

    Attributes
    ----------
    window_id : str
        Unique identifier within the session, e.g. ``"win_0042"``.
    session_id : str
        Source session identifier for traceability.
    start_ts, end_ts : float
        Absolute monotonic timestamps of the window bounds.
    n_frames : int
        Number of BehavioralSample frames in this window.
    dominant_state : str
        The most common engagement state (excluding ``"unreliable"`` when
        other states are present).
    active_stimulus_ids : List[str]
        Any stimulus IDs that overlapped with this window.
    features : np.ndarray
        Normalized feature vector of shape ``(N_FEATURES,)`` ∈ [0, 1].
    metadata : Dict[str, Any]
        Per-window provenance and raw signal means for inspection.
    """
    window_id:           str
    session_id:          str
    start_ts:            float
    end_ts:              float
    n_frames:            int
    dominant_state:      str
    active_stimulus_ids: List[str]
    features:            np.ndarray        # shape (N_FEATURES,), dtype float32
    metadata:            Dict[str, Any] = field(default_factory=dict)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(samples: List[BehavioralSample]) -> np.ndarray:
    """Compute the 18-feature normalized vector from a list of samples.

    Returns a zero-centered neutral vector (all 0.5 → [0,1] centre) for
    empty input, so downstream code always receives a valid array.
    """
    if not samples:
        return np.full(N_FEATURES, 0.5, dtype=np.float32)

    n = len(samples)

    # ── Engagement ────────────────────────────────────────────────────────
    scores = [s.smoothed_score for s in samples]
    eng_mean = float(np.mean(scores))
    eng_std  = float(np.std(scores))
    focused_frac    = sum(1 for s in samples if s.engagement_state == "focused")    / n
    distracted_frac = sum(1 for s in samples if s.engagement_state == "distracted") / n
    fatigued_frac   = sum(1 for s in samples if s.engagement_state == "fatigued")   / n
    unreliable_frac = sum(1 for s in samples if s.engagement_state == "unreliable") / n

    # ── Gaze ──────────────────────────────────────────────────────────────
    on_screen_frac = sum(1 for s in samples if s.is_on_screen) / n
    gaze_h = [s.gaze_h for s in samples]
    gaze_v = [s.gaze_v for s in samples]
    gaze_h_mean = float(np.mean(gaze_h))
    gaze_h_std  = float(np.std(gaze_h))
    gaze_v_std  = float(np.std(gaze_v))

    # ── Head pose ─────────────────────────────────────────────────────────
    head_focused_frac = sum(1 for s in samples if s.head_zone == "focused") / n
    yaw   = [s.head_yaw   for s in samples]
    pitch = [s.head_pitch for s in samples]
    yaw_mean  = float(np.mean(yaw))
    yaw_std   = float(np.std(yaw))
    pitch_std = float(np.std(pitch))

    # ── Blink / eye ───────────────────────────────────────────────────────
    blink_rates = [s.blink_rate for s in samples]
    ears        = [s.mean_ear   for s in samples]
    blink_mean  = float(np.mean(blink_rates))
    blink_std   = float(np.std(blink_rates))
    ear_mean    = float(np.mean(ears))
    fatigue_frac = sum(1 for s in samples if s.is_fatigued) / n

    raw = {
        "engagement_mean":  eng_mean,
        "engagement_std":   eng_std,
        "focused_frac":     focused_frac,
        "distracted_frac":  distracted_frac,
        "fatigued_frac":    fatigued_frac,
        "unreliable_frac":  unreliable_frac,
        "on_screen_frac":   on_screen_frac,
        "gaze_h_mean":      gaze_h_mean,
        "gaze_h_std":       gaze_h_std,
        "gaze_v_std":       gaze_v_std,
        "head_focused_frac": head_focused_frac,
        "head_yaw_mean":    yaw_mean,
        "head_yaw_std":     yaw_std,
        "head_pitch_std":   pitch_std,
        "blink_rate_mean":  blink_mean,
        "ear_mean":         ear_mean,
        "fatigue_frac":     fatigue_frac,
        "blink_rate_std":   blink_std,
    }

    vec = np.empty(N_FEATURES, dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        v = raw[name]
        if name in _SCALE:
            lo, hi = _SCALE[name]
            vec[i] = _norm(v, lo, hi)
        else:
            vec[i] = float(np.clip(v, 0.0, 1.0))

    return vec


def dominant_state(samples: List[BehavioralSample]) -> str:
    """Return the most common engagement state, preferring meaningful states."""
    if not samples:
        return "unreliable"
    from collections import Counter
    counts = Counter(s.engagement_state for s in samples)
    for candidate in ("focused", "drifting", "distracted", "fatigued"):
        if counts.get(candidate, 0) > 0:
            preferred = {k: v for k, v in counts.items()
                         if k in ("focused", "drifting", "distracted", "fatigued")}
            return max(preferred, key=preferred.get)
    return "unreliable"


# ── Sliding-window extractor ──────────────────────────────────────────────────

class FeatureExtractor:
    """Extract a list of BehavioralWindows from a BehavioralSample time series.

    Parameters
    ----------
    window_s : float
        Window duration in seconds (default 2.0).
    stride_s : float
        Step between consecutive window starts in seconds (default 0.5).
        Overlapping windows (stride < window) capture gradual transitions.
    min_frames : int
        Minimum frames for a window to be included (default 5).
    """

    def __init__(
        self,
        window_s:   float = 2.0,
        stride_s:   float = 0.5,
        min_frames: int   = 5,
    ) -> None:
        self.window_s   = window_s
        self.stride_s   = stride_s
        self.min_frames = min_frames

    def extract(
        self,
        samples:    List[BehavioralSample],
        session_id: str = "unknown",
        start_ts:   float = 0.0,
    ) -> List[BehavioralWindow]:
        """Slide a window over *samples* and return one BehavioralWindow per step.

        Parameters
        ----------
        samples    — full session sample list, sorted by timestamp
        session_id — passed through for traceability
        start_ts   — session start (for relative-time metadata)
        """
        if not samples:
            return []

        # Sort defensively
        sorted_samples = sorted(samples, key=lambda s: s.timestamp)
        t_start = sorted_samples[0].timestamp
        t_end   = sorted_samples[-1].timestamp

        if t_end - t_start < self.window_s:
            # Entire session shorter than one window — emit a single window
            return [self._make_window(sorted_samples, 0, session_id, start_ts)]

        windows: List[BehavioralWindow] = []
        cursor    = t_start
        win_idx   = 0

        while cursor + self.window_s <= t_end + 1e-6:
            w_start = cursor
            w_end   = cursor + self.window_s
            in_win  = [s for s in sorted_samples if w_start <= s.timestamp < w_end]

            if len(in_win) >= self.min_frames:
                windows.append(self._make_window(in_win, win_idx, session_id, start_ts))
                win_idx += 1

            cursor += self.stride_s

        return windows

    @staticmethod
    def _make_window(
        samples:    List[BehavioralSample],
        idx:        int,
        session_id: str,
        session_start: float,
    ) -> BehavioralWindow:
        t0 = samples[0].timestamp
        t1 = samples[-1].timestamp
        stim_ids = list({
            s.active_stimulus_id for s in samples
            if s.active_stimulus_id is not None
        })
        return BehavioralWindow(
            window_id=           f"win_{idx:04d}",
            session_id=          session_id,
            start_ts=            t0,
            end_ts=              t1,
            n_frames=            len(samples),
            dominant_state=      dominant_state(samples),
            active_stimulus_ids= stim_ids,
            features=            extract_features(samples),
            metadata={
                "relative_start_s": round(t0 - session_start, 3),
                "relative_end_s":   round(t1 - session_start, 3),
            },
        )
