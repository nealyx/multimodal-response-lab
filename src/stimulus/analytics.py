"""Session analytics — reaction latency, recovery timing, aggregate statistics.

Why compute analytics post-hoc rather than in real time?
---------------------------------------------------------
Real-time computation on the hot path adds latency risk and complicates the
recorder (which should be a simple accumulator).  Analytics only need to be
available at session end for export and display, so computing them once over
the completed SessionLog is simpler and cheaper.

The functions here are pure: they read from SessionLog and return new objects.
Nothing is mutated.  This makes unit testing trivial and allows re-running
analytics with different parameter choices without re-running the experiment.

Reaction latency vs. engagement score
--------------------------------------
These measure different things:

  reaction_latency_ms — time from stimulus onset to a discrete behavioral
                        event (key press or gaze shift).  Comparable to
                        classic RT paradigms.  Fast = attentive + responsive.

  during_score        — mean engagement score while the stimulus is on screen.
                        Continuous rather than discrete.  Captures sustained
                        attention, not just the initial response.

  recovery_s          — time after stimulus offset for engagement score to
                        return to 90% of pre-stimulus baseline.  Long recovery
                        suggests the stimulus was disruptive.

Sources of measurement error in webcam latency estimation
-----------------------------------------------------------
  1. Frame rate granularity — at 30 FPS, the smallest detectable latency
     change is ~33 ms.  Reported latencies are accurate to ±33 ms.
  2. Gaze-based response detection suffers from EMA smoothing delay
     (alpha=0.10 → ~200–400 ms additional lag in engagement state changes).
  3. Key-press responses are the most accurate; limited by keyboard polling
     frequency (typically 1–8 ms at OS level, negligible vs. frame granularity).
  4. Display latency (~8–32 ms on a typical LCD) is NOT subtracted from
     key-press latencies; reported values are stimulus-to-key, not
     stimulus-on-screen-to-key.  Document this as a known systematic offset.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from src.stimulus.events import (
    BehavioralSample, ResponseEvent, SessionLog, StimulusEvent, TrialSummary,
)


class SessionAnalytics:
    """Derives metrics from a completed SessionLog.

    Usage:
        analytics = SessionAnalytics()
        summaries = analytics.compute_trial_summaries(log)
        agg       = analytics.aggregate_by_type(summaries)
        overall   = analytics.overall_stats(log)
    """

    BASELINE_WINDOW_S = 5.0   # seconds before onset used as baseline
    RECOVERY_WINDOW_S = 10.0  # seconds after offset searched for recovery

    # ── Primary API ───────────────────────────────────────────────────────────

    def compute_trial_summaries(self, log: SessionLog) -> List[TrialSummary]:
        """Build one TrialSummary per StimulusEvent in the log."""
        summaries = []
        for ev in log.stimulus_events:
            resp      = self._first_response(log.response_events, ev.stimulus_id)
            baseline  = self._baseline_score(log.samples, ev.onset_ts)
            during    = self._during_score(log.samples, ev)
            recovery  = self._recovery_time(log.samples, ev, baseline)

            summaries.append(TrialSummary(
                stimulus_id=         ev.stimulus_id,
                trial_index=         ev.trial_index,
                stimulus_type=       ev.stimulus_type,
                onset_ts=            ev.onset_ts,
                duration_ms=         ev.duration_ms,
                reaction_latency_ms= resp.latency_ms if resp else None,
                response_type=       resp.response_type if resp else None,
                hit=                 resp.hit if resp else False,
                baseline_score=      baseline,
                during_score=        during,
                recovery_s=          recovery,
            ))
        return summaries

    def aggregate_by_type(
        self,
        summaries: List[TrialSummary],
    ) -> Dict[str, Dict]:
        """Group TrialSummaries by stimulus_type and compute per-type statistics."""
        groups: Dict[str, List[TrialSummary]] = {}
        for s in summaries:
            groups.setdefault(s.stimulus_type, []).append(s)

        result: Dict[str, Dict] = {}
        for stype, trials in groups.items():
            latencies  = [t.reaction_latency_ms for t in trials
                          if t.reaction_latency_ms is not None]
            recoveries = [t.recovery_s for t in trials if t.recovery_s is not None]
            during     = [t.during_score for t in trials]

            result[stype] = {
                "n_trials":          len(trials),
                "n_responses":       len(latencies),
                "hit_rate":          sum(t.hit for t in trials) / len(trials),
                "mean_latency_ms":   statistics.mean(latencies)   if latencies          else None,
                "median_latency_ms": statistics.median(latencies) if latencies          else None,
                "std_latency_ms":    statistics.stdev(latencies)  if len(latencies) > 1 else None,
                "mean_recovery_s":   statistics.mean(recoveries)  if recoveries         else None,
                "mean_during_score": statistics.mean(during)      if during             else None,
            }
        return result

    def overall_stats(self, log: SessionLog) -> Dict:
        """Session-level summary statistics."""
        if not log.samples:
            return {}

        scores  = [s.smoothed_score    for s in log.samples]
        focused = [s for s in log.samples if s.engagement_state == "focused"]
        on_scr  = [s for s in log.samples if s.is_on_screen]

        latencies = [r.latency_ms for r in log.response_events
                     if r.response_type == "key_press"]

        return {
            "duration_s":          log.duration_s,
            "n_frames":            len(log.samples),
            "n_trials":            len(log.stimulus_events),
            "n_responses":         len(log.response_events),
            "mean_score":          statistics.mean(scores),
            "focused_pct":         100.0 * len(focused) / len(log.samples),
            "on_screen_pct":       100.0 * len(on_scr)  / len(log.samples),
            "mean_key_latency_ms": statistics.mean(latencies) if latencies else None,
        }

    def rolling_baseline(
        self,
        samples: List[BehavioralSample],
        before_ts: float,
        window_s: float = 5.0,
    ) -> float:
        """Mean engagement score in [before_ts - window_s, before_ts)."""
        window = [
            s for s in samples
            if (before_ts - window_s) <= s.timestamp < before_ts
        ]
        if not window:
            return 0.5
        return statistics.mean(s.smoothed_score for s in window)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _first_response(
        self,
        responses: List[ResponseEvent],
        stimulus_id: str,
    ) -> Optional[ResponseEvent]:
        for r in responses:
            if r.stimulus_id == stimulus_id:
                return r
        return None

    def _window(
        self,
        samples: List[BehavioralSample],
        t_start: float,
        t_end: float,
    ) -> List[BehavioralSample]:
        return [s for s in samples if t_start <= s.timestamp < t_end]

    def _baseline_score(
        self,
        samples: List[BehavioralSample],
        onset_ts: float,
    ) -> float:
        w = self._window(samples, onset_ts - self.BASELINE_WINDOW_S, onset_ts)
        return statistics.mean(s.smoothed_score for s in w) if w else 0.5

    def _during_score(
        self,
        samples: List[BehavioralSample],
        ev: StimulusEvent,
    ) -> float:
        offset = ev.offset_ts if ev.offset_ts else ev.onset_ts + ev.duration_ms / 1000.0
        w = self._window(samples, ev.onset_ts, offset)
        return statistics.mean(s.smoothed_score for s in w) if w else 0.0

    def _recovery_time(
        self,
        samples: List[BehavioralSample],
        ev: StimulusEvent,
        baseline: float,
    ) -> Optional[float]:
        offset    = ev.offset_ts if ev.offset_ts else ev.onset_ts + ev.duration_ms / 1000.0
        threshold = baseline * 0.90
        post      = self._window(samples, offset, offset + self.RECOVERY_WINDOW_S)
        for s in post:
            if s.smoothed_score >= threshold:
                return s.timestamp - offset
        return None
