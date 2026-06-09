"""SessionReporter — derives SessionMetrics from a completed SessionLog.

This module has no dependency on OpenCV, MediaPipe, or any live-pipeline
module.  It only imports from:
  - src.stimulus.events  (pure dataclasses)
  - src.stimulus.analytics (pure computation)
  - src.reporting.metrics (pure dataclasses)
  - standard library + numpy

This separation means the reporting layer can run on any machine that has
numpy, without the full webcam stack installed.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.reporting.metrics import (
    BlinkMetrics,
    EngagementMetrics,
    GazeMetrics,
    HeadMetrics,
    SessionMetrics,
    SignalSummary,
    StimulusTypeMetrics,
    summarize,
)
from src.stimulus.analytics import SessionAnalytics
from src.stimulus.events import BehavioralSample, SessionLog, TrialSummary


class SessionReporter:
    """Computes a SessionMetrics from a SessionLog."""

    def compute(self, log: SessionLog) -> SessionMetrics:
        """Derive all session-level metrics from *log*.

        Works with any combination of available data:
          - samples only (no stimuli) → engagement/gaze/head/blink metrics
          - stimuli + responses + samples → full report including per-type stats
          - empty log → returns a valid object with zero values
        """
        samples = log.samples

        # Ensure trial summaries are available
        trial_summaries = list(log.trial_summaries)
        if not trial_summaries and (log.stimulus_events or log.samples):
            analytics      = SessionAnalytics()
            trial_summaries = analytics.compute_trial_summaries(log)

        return SessionMetrics(
            session_id=      log.session_id,
            experiment_name= log.experiment_name,
            generated_at=    datetime.now(timezone.utc).isoformat(),
            duration_s=      log.duration_s,
            n_frames=        len(samples),
            n_trials=        len(log.stimulus_events),
            n_responses=     len(log.response_events),
            engagement=      self._engagement(samples),
            gaze=            self._gaze(samples),
            head=            self._head(samples),
            blink=           self._blink(samples),
            stimulus_types=  self._stimulus_types(trial_summaries),
            reaction_latencies_ms=self._key_latencies(log),
            config_snapshot= log.config,
        )

    # ── Per-channel metrics ───────────────────────────────────────────────────

    @staticmethod
    def _engagement(samples: List[BehavioralSample]) -> EngagementMetrics:
        if not samples:
            empty = summarize([])
            return EngagementMetrics(empty, 0.0, 0.0, 0.0, 0.0, 0.0)

        scores = [s.smoothed_score for s in samples]
        n      = len(samples)

        def _frac(state: str) -> float:
            return sum(1 for s in samples if s.engagement_state == state) / n

        return EngagementMetrics(
            score_summary=       summarize(scores),
            focused_fraction=    _frac("focused"),
            drifting_fraction=   _frac("drifting"),
            distracted_fraction= _frac("distracted"),
            fatigued_fraction=   _frac("fatigued"),
            unreliable_fraction= _frac("unreliable"),
        )

    @staticmethod
    def _gaze(samples: List[BehavioralSample]) -> GazeMetrics:
        if not samples:
            empty = summarize([])
            return GazeMetrics(0.0, empty, empty)

        on_scr = sum(1 for s in samples if s.is_on_screen) / len(samples)
        return GazeMetrics(
            on_screen_fraction= on_scr,
            gaze_h_summary=     summarize([s.gaze_h for s in samples]),
            gaze_v_summary=     summarize([s.gaze_v for s in samples]),
        )

    @staticmethod
    def _head(samples: List[BehavioralSample]) -> HeadMetrics:
        if not samples:
            empty = summarize([])
            return HeadMetrics(0.0, empty, empty)

        focused = sum(1 for s in samples if s.head_zone == "focused") / len(samples)
        return HeadMetrics(
            head_focused_fraction= focused,
            yaw_summary=           summarize([s.head_yaw   for s in samples]),
            pitch_summary=         summarize([s.head_pitch for s in samples]),
        )

    @staticmethod
    def _blink(samples: List[BehavioralSample]) -> BlinkMetrics:
        if not samples:
            empty = summarize([])
            return BlinkMetrics(empty, 0.0, empty)

        fatigued = sum(1 for s in samples if s.is_fatigued) / len(samples)
        return BlinkMetrics(
            blink_rate_summary= summarize([s.blink_rate for s in samples]),
            fatigue_fraction=   fatigued,
            ear_summary=        summarize([s.mean_ear   for s in samples]),
        )

    @staticmethod
    def _stimulus_types(
        summaries: List[TrialSummary],
    ) -> Dict[str, StimulusTypeMetrics]:
        if not summaries:
            return {}

        groups: Dict[str, List[TrialSummary]] = {}
        for t in summaries:
            groups.setdefault(t.stimulus_type, []).append(t)

        result: Dict[str, StimulusTypeMetrics] = {}
        for stype, trials in groups.items():
            latencies  = [t.reaction_latency_ms for t in trials
                          if t.reaction_latency_ms is not None]
            recoveries = [t.recovery_s for t in trials if t.recovery_s is not None]
            baselines  = [t.baseline_score for t in trials]
            durings    = [t.during_score   for t in trials]

            mean_lat  = statistics.mean(latencies)   if latencies           else None
            med_lat   = statistics.median(latencies) if latencies           else None
            std_lat   = statistics.stdev(latencies)  if len(latencies) > 1  else None
            mean_rec  = statistics.mean(recoveries)  if recoveries          else None
            mean_base = statistics.mean(baselines)   if baselines           else 0.0
            mean_dur  = statistics.mean(durings)     if durings             else 0.0

            result[stype] = StimulusTypeMetrics(
                stimulus_type=       stype,
                n_trials=            len(trials),
                hit_rate=            sum(t.hit for t in trials) / len(trials),
                mean_latency_ms=     mean_lat,
                median_latency_ms=   med_lat,
                std_latency_ms=      std_lat,
                mean_baseline_score= mean_base,
                mean_during_score=   mean_dur,
                score_delta=         mean_dur - mean_base,
                mean_recovery_s=     mean_rec,
            )
        return result

    @staticmethod
    def _key_latencies(log: SessionLog) -> List[float]:
        return [
            r.latency_ms for r in log.response_events
            if r.response_type == "key_press"
        ]
