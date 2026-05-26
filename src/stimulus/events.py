"""Structured event schemas for stimulus-response experiments.

Why raw events are kept separate from interpreted metrics
----------------------------------------------------------
A StimulusEvent is a fact: "stimulus X appeared at monotonic time T."
A ResponseEvent is an observation: "behavioral signal Y changed Z ms later."
Interpretation (was this a genuine response? what does the latency mean?) is
deferred to analytics.py, which operates on the completed log.

This separation means:
  - The recorder can be lightweight (no computation on the hot path).
  - Analytics can be re-run offline with different parameters.
  - Events can be synced across machines if the same monotonic epoch is used.

All timestamps use time.monotonic() — immune to NTP corrections and wall-clock
adjustments that would corrupt latency measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Primitive event types ────────────────────────────────────────────────────

@dataclass
class StimulusEvent:
    """Record of a single stimulus presentation."""
    stimulus_id:  str
    trial_index:  int
    stimulus_type: str
    onset_ts:     float           # monotonic — stimulus first rendered
    offset_ts:    Optional[float] # monotonic — stimulus last rendered (None = ongoing)
    duration_ms:  float           # intended duration
    metadata:     Dict[str, Any]  # type-specific params (color, shape, …)

    @property
    def actual_duration_ms(self) -> Optional[float]:
        """Measured on-screen time; None while stimulus is still active."""
        if self.offset_ts is None:
            return None
        return (self.offset_ts - self.onset_ts) * 1000.0

    @property
    def is_active(self) -> bool:
        return self.offset_ts is None


@dataclass
class ResponseEvent:
    """An observable behavioral change detected after a stimulus onset."""
    stimulus_id:   str
    trial_index:   int
    response_type: str    # "key_press" | "gaze_shift" | "attention_change"
    response_ts:   float  # monotonic
    latency_ms:    float  # response_ts − stimulus onset_ts
    hit:           bool   # True if within a valid response window
    metadata:      Dict[str, Any]


@dataclass
class BehavioralSample:
    """Full multi-channel signal snapshot for one processed webcam frame.

    This is the time-series record that downstream analytics operate on.
    Every field is either raw (EAR, gaze ratios) or already-computed by an
    upstream module — no additional inference happens here.
    """
    frame_index:        int
    timestamp:          float         # monotonic
    active_stimulus_id: Optional[str] # which stimulus was on screen (None if ISI)

    # Engagement layer (Day 6)
    engagement_state:  str    # EngagementState.value
    smoothed_score:    float
    confidence:        float
    focused_fraction:  float

    # Gaze layer (Day 5)
    gaze_zone:   str    # GazeZone.value
    is_on_screen: bool
    gaze_h:      float  # mean horizontal iris ratio [0, 1]
    gaze_v:      float  # mean vertical iris ratio [0, 1]

    # Head pose layer (Day 4)
    head_zone:   str    # AttentionZone.value
    head_yaw:    float  # degrees
    head_pitch:  float  # degrees

    # Blink / eye layer (Day 3)
    blink_state: str    # BlinkState.value
    mean_ear:    float
    blink_rate:  float  # per minute
    is_fatigued: bool   # is_drowsy or is_prolonged_closure (mirrors engagement scorer)


# ── Per-trial summary (derived by analytics) ─────────────────────────────────

@dataclass
class TrialSummary:
    """Per-trial aggregate derived from BehavioralSamples and ResponseEvents."""
    stimulus_id:   str
    trial_index:   int
    stimulus_type: str
    onset_ts:      float
    duration_ms:   float

    # Response
    reaction_latency_ms: Optional[float]  # None if no response recorded
    response_type:       Optional[str]
    hit:                 bool

    # Engagement during the trial
    baseline_score: float  # mean score in window before onset
    during_score:   float  # mean score during stimulus
    recovery_s:     Optional[float]  # seconds to recover to ≥ 90% of baseline


# ── Session container ────────────────────────────────────────────────────────

@dataclass
class SessionLog:
    """Complete record of one experiment session."""
    session_id:      str
    experiment_name: str
    start_ts:        float          # monotonic session start
    end_ts:          Optional[float]
    config:          Dict[str, Any]

    stimulus_events: List[StimulusEvent]  = field(default_factory=list)
    response_events: List[ResponseEvent]  = field(default_factory=list)
    samples:         List[BehavioralSample] = field(default_factory=list)
    trial_summaries: List[TrialSummary]   = field(default_factory=list)

    @property
    def duration_s(self) -> Optional[float]:
        if self.end_ts is None:
            return None
        return self.end_ts - self.start_ts

    @property
    def n_trials(self) -> int:
        return len(self.stimulus_events)
