"""Composite engagement scoring: fuses blink, head pose, and gaze into one score.

Why fuse multiple weak signals?
---------------------------------
Each individual signal has a high false-positive rate when used alone:

  • Gaze alone: iris tracking degrades with glasses, bad lighting, or extreme
    head angles.  A person staring off to one side of the screen can still be
    genuinely engaged.
  • Head pose alone: someone can look straight at the camera while being
    completely mentally absent, or turn their head while still reading.
  • Blink rate alone: low rate may mean concentration or tiredness; high rate
    may mean eye irritation or active engagement.

Combining three orthogonal channels — WHERE the eyes are pointing (gaze),
WHERE the head is pointing (head pose), and WHETHER the eyes are behaviorally
healthy (blink/EAR) — makes the composite far more robust than any single
channel.  Individual channel noise is largely uncorrelated, so their product
correctly pushes toward zero only when all three agree that attention is absent.

Why temporal smoothing is necessary
--------------------------------------
Frame-to-frame scores jump significantly due to:
  • A single blink (EAR drops briefly, pulling eye_score to near zero)
  • MediaPipe jitter at the start of a detection
  • Transient head movement (adjusting posture)
  • Occasional iris landmark instability

An Exponential Moving Average (EMA) with a short half-life (~0.4 s at 25 FPS)
absorbs these transients without creating noticeable lag for genuine attention
changes (which take seconds to develop).

State debouncing adds a second layer: the new state must persist for N
consecutive frames before the label changes, preventing single-frame flips in
the displayed state badge.

Reliability vs. attention
---------------------------
These are independent dimensions:

  confidence  — how much we trust the measurement.  Low if:
                  • face absent (0.0)
                  • no iris landmarks (gaze_weight penalty)
                  • head pose reprojection error is high
                  • history too short to be statistically meaningful

  smoothed_score — the signal value (what we measured, given confidence > 0).

A high-confidence LOW score → person is distracted.
A low-confidence score    → we cannot tell.

The engagement state maps onto this 2-D space:
  FOCUSED     : high score, high confidence
  DRIFTING    : moderate score, any confidence
  DISTRACTED  : low score, adequate confidence
  FATIGUED    : fatigue flags active, regardless of score
  UNRELIABLE  : face absent or confidence < min_confidence

Failure modes that remain
---------------------------
  • Motivated deception: looking at camera while mentally absent — we do not
    read cognitive engagement, only behavioural orientation.
  • Screen off-center: someone with two monitors may look legitimately at a
    second monitor and be classified as DISTRACTED.
  • Calibration: all thresholds are population averages; per-user adaptation
    would reduce false alarms significantly.
  • Short sessions: confidence builds slowly; the first 10–15 s of a session
    should be treated as warm-up.

This is approximate behavioural inference, NOT a medical or cognitive
diagnostic tool.  Results should not be used to make individual assessments.

How this completes the human-response analyzer
------------------------------------------------
The pipeline is now:

  Raw pixels → face landmarks (Day 2)
           → EAR / blink FSM (Day 3)
           → head pose / attention zone (Day 4)
           → iris position / gaze zone (Day 5)
           → composite engagement score (Day 6)

Day 6 adds the synthesis layer.  The output is:
  EngagementScore.state          — categorical (FOCUSED/DRIFTING/DISTRACTED/…)
  EngagementScore.smoothed_score — continuous [0, 1] for time-series logging
  EngagementScore.focused_fraction  — rolling percentage (the EGRA headline metric)
  EngagementScore.confidence     — data-quality gate for downstream decisions

A system reading these fields can trigger adaptive responses:
  score < 0.4 for > 30 s → prompt the user
  state == FATIGUED        → suggest a break
  focused_fraction > 0.8  → log as high-engagement session
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple
from typing import Optional

from src.signals.attention import AttentionAnalysis
from src.signals.blink_detector import BlinkAnalysis
from src.signals.gaze import GazeAnalysis


# ── Confidence breakdown ──────────────────────────────────────────────────────

@dataclass
class ConfidenceBreakdown:
    """What drives the confidence score — for diagnostics and honest display.

    confidence = history_warmup × gaze_quality × pose_quality

    history_warmup  — ramps 0→1 as elapsed time grows toward min_history_s.
                      With min_history_s=10, it reaches 1.0 after 10 s.
                      This is WHY confidence can be 0.41 while all signals look
                      healthy: 4.1 s have elapsed out of 10 s required.

    gaze_quality    — 0.75 when iris landmarks are unreliable or absent;
                      1.00 when gaze is reliable.

    pose_quality    — decreases as head-pose reprojection error rises above
                      50% of pose_error_max.  Typically stays near 1.0.

    warmup_remaining_s — seconds until history_warmup reaches 1.0.
    """
    history_warmup:      float   # 0–1, time-based warmup fraction
    gaze_quality:        float   # 0–1, iris signal quality
    pose_quality:        float   # 0–1, head-pose quality
    overall:             float   # = history_warmup × gaze_quality × pose_quality
    warmup_remaining_s:  float   # seconds until fully warmed up (0 when done)


# ── State enum ────────────────────────────────────────────────────────────────

class EngagementState(str, Enum):
    FOCUSED    = "focused"      # gaze+head on screen, eyes open, no fatigue
    DRIFTING   = "drifting"     # partially engaged; score moderate
    DISTRACTED = "distracted"   # off-screen or head away, no fatigue flags
    FATIGUED   = "fatigued"     # drowsy / prolonged closure / high fatigue
    UNRELIABLE = "unreliable"   # face absent or confidence below minimum


# ── Output data structure ─────────────────────────────────────────────────────

@dataclass
class EngagementScore:
    """Composite per-frame engagement measurement.

    smoothed_score and focused_fraction are the primary outputs.
    raw_score and component scores are available for debugging/display.
    """
    # Headline outputs
    state:           EngagementState
    raw_score:       float   # instantaneous weighted sum [0, 1]
    smoothed_score:  float   # EMA-smoothed score [0, 1]
    confidence:      float   # trust in this reading [0, 1]

    # Rolling metric
    focused_fraction:  float   # proportion of recent frames in FOCUSED state
    engaged_fraction:  float   # proportion in FOCUSED or DRIFTING

    # Component scores (for HUD display and debugging)
    gaze_score:  float
    head_score:  float
    eye_score:   float

    # Active flags (derived from upstream signals)
    is_fatigued:      bool
    is_looking_away:  bool
    is_gaze_reliable: bool

    timestamp:   float
    frame_index: int

    # Confidence decomposition (for diagnostics display)
    confidence_breakdown: ConfidenceBreakdown = field(
        default_factory=lambda: ConfidenceBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)
    )


# ── Scorer ────────────────────────────────────────────────────────────────────

class AttentionScorer:
    """Fuses blink, head pose, and gaze analyses into a single engagement score.

    Call update() once per frame; pass None for any signal that was unavailable
    that frame.  The scorer handles missing inputs gracefully.
    """

    def __init__(self, cfg: dict) -> None:
        sc = cfg.get("engagement", {})

        w = sc.get("weights", {})
        self._w_gaze = float(w.get("gaze", 0.50))
        self._w_head = float(w.get("head", 0.20))
        self._w_eye  = float(w.get("eye",  0.30))

        th = sc.get("thresholds", {})
        self._thr_focused   = float(th.get("focused",   0.72))
        self._thr_drifting  = float(th.get("drifting",  0.45))
        self._min_confidence = float(th.get("min_confidence", 0.20))

        sm = sc.get("smoothing", {})
        self._alpha           = float(sm.get("alpha",            0.10))
        self._state_min_frames = int(sm.get("state_min_frames",  5))

        conf = sc.get("confidence", {})
        self._min_history_s  = float(conf.get("min_history_s",  10.0))
        self._pose_error_max = float(conf.get("pose_error_max", 10.0))

        self._window_s = float(sc.get("rolling_window_s", 60.0))

        # Internal state
        self._smoothed:       float = 0.5      # start at neutral
        self._current_state:  EngagementState = EngagementState.UNRELIABLE
        self._candidate_state: EngagementState = EngagementState.UNRELIABLE
        self._candidate_frames: int            = 0
        self._t_first:        Optional[float]  = None  # first update timestamp
        # Rolling history: (timestamp, state)
        self._history: deque[tuple[float, EngagementState]] = deque()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        *,
        blink:         Optional[BlinkAnalysis]     = None,
        head:          Optional[AttentionAnalysis]  = None,
        gaze:          Optional[GazeAnalysis]       = None,
        face_detected: bool  = True,
        timestamp:     float = 0.0,
        frame_index:   int   = 0,
    ) -> EngagementScore:
        """Fuse one frame of signal analyses into an EngagementScore.

        All signal arguments are optional so callers can pass whichever signals
        were available that frame without special-casing missing channels.
        """
        if self._t_first is None:
            self._t_first = timestamp

        # ── Component scores ──────────────────────────────────────────────
        gaze_s = self._gaze_component(gaze, head)
        head_s = self._head_component(head)
        eye_s  = self._eye_component(blink)

        # ── Raw and smoothed ──────────────────────────────────────────────
        raw = self._w_gaze * gaze_s + self._w_head * head_s + self._w_eye * eye_s
        self._smoothed = self._alpha * raw + (1.0 - self._alpha) * self._smoothed

        # ── Flags ─────────────────────────────────────────────────────────
        is_fatigued     = self._fatigue_flag(blink)
        is_looking_away = self._looking_away_flag(head)
        is_gaze_reliable = (gaze is not None and gaze.is_reliable)

        # ── Confidence ────────────────────────────────────────────────────
        confidence, conf_breakdown = self._compute_confidence(
            face_detected, gaze, head, timestamp,
        )

        # ── State classification with debounce ────────────────────────────
        candidate = self._classify(
            self._smoothed, is_fatigued, confidence, face_detected,
        )
        if candidate == self._candidate_state:
            self._candidate_frames += 1
        else:
            self._candidate_state  = candidate
            self._candidate_frames = 1

        if self._candidate_frames >= self._state_min_frames:
            self._current_state = candidate

        state = self._current_state

        # ── Rolling history ───────────────────────────────────────────────
        self._history.append((timestamp, state))
        self._prune(timestamp)

        focused_frac  = self._fraction(EngagementState.FOCUSED)
        engaged_frac  = self._fraction(
            EngagementState.FOCUSED, EngagementState.DRIFTING,
        )

        return EngagementScore(
            state=state,
            raw_score=round(raw, 4),
            smoothed_score=round(self._smoothed, 4),
            confidence=round(confidence, 4),
            focused_fraction=focused_frac,
            engaged_fraction=engaged_frac,
            gaze_score=round(gaze_s, 4),
            head_score=round(head_s, 4),
            eye_score=round(eye_s,  4),
            is_fatigued=is_fatigued,
            is_looking_away=is_looking_away,
            is_gaze_reliable=is_gaze_reliable,
            timestamp=timestamp,
            frame_index=frame_index,
            confidence_breakdown=conf_breakdown,
        )

    def reset(self) -> None:
        """Reset temporal state.  Component state (history) is preserved."""
        self._smoothed          = 0.5
        self._current_state     = EngagementState.UNRELIABLE
        self._candidate_state   = EngagementState.UNRELIABLE
        self._candidate_frames  = 0
        self._t_first           = None

    # ── Component scorers (pure functions of their inputs) ─────────────────

    @staticmethod
    def _gaze_component(
        gaze: Optional[GazeAnalysis],
        head: Optional[AttentionAnalysis],
    ) -> float:
        """Primary on-screen signal.  Gaze already incorporates head-pose gate."""
        if gaze is not None and gaze.is_reliable:
            return gaze.on_screen_fraction
        # Iris unreliable or absent — fall back to head-only, lower ceiling
        if head is not None:
            return head.attention_fraction * 0.65
        return 0.0

    @staticmethod
    def _head_component(head: Optional[AttentionAnalysis]) -> float:
        """Supplementary head stability signal."""
        if head is None:
            return 0.5   # neutral: absence of evidence is not evidence of absence
        score = head.attention_fraction
        # Penalise high rotational jitter (fidgety = lower engagement quality)
        std_sum = head.yaw_std + head.pitch_std
        stability_penalty = min(0.5, std_sum / 30.0)
        return max(0.0, score * (1.0 - stability_penalty))

    @staticmethod
    def _eye_component(blink: Optional[BlinkAnalysis]) -> float:
        """Alertness/eye-health signal from blink and EAR metrics."""
        if blink is None:
            return 0.5   # neutral
        score = blink.mean_openness
        if blink.is_prolonged_closure:
            score *= 0.15   # heavy: eye has been closed a long time
        elif blink.is_drowsy:
            score *= 0.35   # moderate: multiple drowsy indicators
        if blink.is_low_blink_rate:
            score *= 0.80   # mild: staring / early fatigue
        if blink.is_high_blink_rate:
            score *= 0.85   # mild: eye strain
        return max(0.0, min(1.0, score))

    # ── State classification ───────────────────────────────────────────────

    def _classify(
        self,
        score:         float,
        is_fatigued:   bool,
        confidence:    float,
        face_detected: bool,
    ) -> EngagementState:
        if not face_detected or confidence < self._min_confidence:
            return EngagementState.UNRELIABLE
        if is_fatigued:
            return EngagementState.FATIGUED
        if score >= self._thr_focused:
            return EngagementState.FOCUSED
        if score >= self._thr_drifting:
            return EngagementState.DRIFTING
        return EngagementState.DISTRACTED

    # ── Confidence computation ─────────────────────────────────────────────

    def _compute_confidence(
        self,
        face_detected: bool,
        gaze:          Optional[GazeAnalysis],
        head:          Optional[AttentionAnalysis],
        timestamp:     float,
    ) -> tuple:
        """Return (confidence: float, breakdown: ConfidenceBreakdown).

        confidence = history_warmup × gaze_quality × pose_quality

        Callers see a low confidence primarily because of history_warmup:
        the score starts at 0 and ramps to 1 over min_history_s seconds.
        This prevents false-positive FOCUSED classification in the first
        few seconds before the EMA has stabilised.
        """
        if not face_detected:
            bd = ConfidenceBreakdown(
                history_warmup=0.0, gaze_quality=0.0, pose_quality=0.0,
                overall=0.0, warmup_remaining_s=max(0.0, self._min_history_s),
            )
            return 0.0, bd

        # ── History warmup ────────────────────────────────────────────────
        if self._t_first is not None:
            if self._min_history_s <= 0.0:
                history_conf = 1.0
                remaining_s  = 0.0
            else:
                elapsed_s    = max(0.0, timestamp - self._t_first)
                history_conf = min(1.0, elapsed_s / self._min_history_s)
                remaining_s  = max(0.0, self._min_history_s - elapsed_s)
        else:
            history_conf = 0.0
            remaining_s  = float(self._min_history_s)

        # ── Gaze quality ──────────────────────────────────────────────────
        gaze_quality = 1.0 if (gaze is not None and gaze.is_reliable) else 0.75

        # ── Pose quality ──────────────────────────────────────────────────
        pose_quality = 1.0
        if head is not None:
            excess = max(0.0, head.reprojection_error - self._pose_error_max * 0.5)
            pose_penalty = min(0.3, excess / max(self._pose_error_max, 1e-6))
            pose_quality = 1.0 - pose_penalty

        confidence = history_conf * gaze_quality * pose_quality
        confidence  = max(0.0, min(1.0, confidence))

        bd = ConfidenceBreakdown(
            history_warmup=round(history_conf, 4),
            gaze_quality=  round(gaze_quality, 4),
            pose_quality=  round(pose_quality, 4),
            overall=       round(confidence,   4),
            warmup_remaining_s=round(remaining_s, 2),
        )
        return confidence, bd

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _fatigue_flag(blink: Optional[BlinkAnalysis]) -> bool:
        if blink is None:
            return False
        return blink.is_drowsy or blink.is_prolonged_closure or blink.fatigue_label == "HIGH"

    @staticmethod
    def _looking_away_flag(head: Optional[AttentionAnalysis]) -> bool:
        if head is None:
            return False
        from src.signals.attention import AttentionZone
        return head.zone == AttentionZone.LOOKING_AWAY

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _fraction(self, *states: EngagementState) -> float:
        if not self._history:
            return 0.0
        count = sum(1 for _, s in self._history if s in states)
        return count / len(self._history)
