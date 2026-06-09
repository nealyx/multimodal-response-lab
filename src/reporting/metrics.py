"""Session-level metric schemas for offline behavioral analytics.

Why these are separate from live inference objects
---------------------------------------------------
Live objects (BlinkAnalysis, EngagementScore, GazeAnalysis) are optimised for
per-frame latency: they hold only the minimum state needed to produce the next
frame's output.  Session metrics require the full temporal record — you cannot
compute 'distracted_fraction over a 5-minute session' without accumulating all
5 minutes of frame data.

Keeping the reporting schema separate also means:
  - analyze_session.py never imports OpenCV or MediaPipe.
  - Metrics can be computed on exported JSON/CSV files on a different machine.
  - The schema can be versioned independently of the live pipeline.

Why session-level metrics matter more than single-frame labels
--------------------------------------------------------------
A single frame classified as DISTRACTED may be a transient glance away.
80% of frames classified as DISTRACTED means something fundamentally different.
Session-level fractions, reaction latency distributions, and per-stimulus
score deltas are the signals that a researcher, product team, or hiring manager
can actually interpret.

These are approximate behavioral engineering metrics, NOT medical or cognitive
diagnostic measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── Low-level distribution summary ───────────────────────────────────────────

@dataclass
class SignalSummary:
    """Descriptive statistics for one continuous signal channel over a session."""
    mean:    float
    std:     float
    min_val: float
    max_val: float
    p25:     float   # 25th percentile
    p75:     float   # 75th percentile
    n:       int     # number of valid samples


def summarize(values: List[float]) -> SignalSummary:
    """Compute SignalSummary from a list of floats.  Returns zeros for empty input."""
    if not values:
        return SignalSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    arr = np.asarray(values, dtype=float)
    return SignalSummary(
        mean=    float(np.mean(arr)),
        std=     float(np.std(arr)),
        min_val= float(np.min(arr)),
        max_val= float(np.max(arr)),
        p25=     float(np.percentile(arr, 25)),
        p75=     float(np.percentile(arr, 75)),
        n=       len(arr),
    )


# ── Per-channel metric groups ─────────────────────────────────────────────────

@dataclass
class EngagementMetrics:
    """Engagement score distribution and state fractions for the whole session."""
    score_summary:       SignalSummary
    focused_fraction:    float   # proportion of frames in FOCUSED state
    drifting_fraction:   float
    distracted_fraction: float
    fatigued_fraction:   float
    unreliable_fraction: float


@dataclass
class GazeMetrics:
    """Gaze direction distribution for the session."""
    on_screen_fraction: float
    gaze_h_summary:     SignalSummary   # horizontal iris ratio [0, 1]
    gaze_v_summary:     SignalSummary   # vertical iris ratio [0, 1]


@dataclass
class HeadMetrics:
    """Head pose distribution for the session."""
    head_focused_fraction: float   # proportion in FOCUSED attention zone
    yaw_summary:           SignalSummary   # degrees
    pitch_summary:         SignalSummary   # degrees


@dataclass
class BlinkMetrics:
    """Eye / blink behavioral summary for the session."""
    blink_rate_summary: SignalSummary   # blinks per minute, per-frame values
    fatigue_fraction:   float          # proportion of frames where is_fatigued
    ear_summary:        SignalSummary   # eye aspect ratio distribution


# ── Per-stimulus-type aggregate ───────────────────────────────────────────────

@dataclass
class StimulusTypeMetrics:
    """Aggregate across all trials of one stimulus type."""
    stimulus_type:       str
    n_trials:            int
    hit_rate:            float           # fraction of trials with any response
    mean_latency_ms:     Optional[float] # mean reaction latency (key-press)
    median_latency_ms:   Optional[float]
    std_latency_ms:      Optional[float]
    mean_baseline_score: float
    mean_during_score:   float
    score_delta:         float           # during − baseline (negative = disruption)
    mean_recovery_s:     Optional[float]


# ── Top-level session metrics ─────────────────────────────────────────────────

@dataclass
class SessionMetrics:
    """Complete session-level metrics for one experiment run.

    Derived offline from a SessionLog (JSON + samples.csv).
    All fields are either aggregate statistics or distribution summaries —
    no raw time-series data is embedded here.
    """
    # Identity
    session_id:      str
    experiment_name: str
    generated_at:    str   # ISO-8601 UTC

    # Session overview
    duration_s:  Optional[float]
    n_frames:    int
    n_trials:    int
    n_responses: int

    # Channel summaries
    engagement:  EngagementMetrics
    gaze:        GazeMetrics
    head:        HeadMetrics
    blink:       BlinkMetrics

    # Stimulus analytics (empty dicts when no stimuli were presented)
    stimulus_types:       Dict[str, StimulusTypeMetrics] = field(default_factory=dict)
    reaction_latencies_ms: List[float]                    = field(default_factory=list)

    # Provenance
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
