"""Unit tests for SessionAnalytics, EventRecorder response detection, and
SessionExporter (smoke tests — no filesystem writes needed for most checks).

All tests use synthetic in-memory data; no webcam or config files required.
"""

import json
import os
import tempfile

import pytest

from src.stimulus.analytics import SessionAnalytics
from src.stimulus.events import (
    BehavioralSample, ResponseEvent, SessionLog, StimulusEvent, TrialSummary,
)
from src.stimulus.exporter import SessionExporter
from src.stimulus.recorder import EventRecorder


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _sample(
    ts: float,
    score: float = 0.75,
    state: str = "focused",
    on_screen: bool = True,
    stim_id: str = None,
) -> BehavioralSample:
    return BehavioralSample(
        frame_index=0, timestamp=ts,
        active_stimulus_id=stim_id,
        engagement_state=state, smoothed_score=score, confidence=0.8,
        focused_fraction=0.7,
        gaze_zone="center", is_on_screen=on_screen, gaze_h=0.5, gaze_v=0.5,
        head_zone="focused", head_yaw=0.0, head_pitch=0.0,
        blink_state="open", mean_ear=0.30, blink_rate=14.0, is_fatigued=False,
    )


def _stim(
    stim_id: str = "s0",
    onset: float = 10.0,
    offset: float = 10.5,
    duration_ms: float = 500.0,
    stim_type: str = "ColorFlash",
) -> StimulusEvent:
    return StimulusEvent(
        stimulus_id=stim_id, trial_index=0,
        stimulus_type=stim_type,
        onset_ts=onset, offset_ts=offset,
        duration_ms=duration_ms, metadata={},
    )


def _resp(
    stim_id: str = "s0",
    latency_ms: float = 300.0,
    rtype: str = "key_press",
    hit: bool = True,
) -> ResponseEvent:
    return ResponseEvent(
        stimulus_id=stim_id, trial_index=0,
        response_type=rtype,
        response_ts=10.0 + latency_ms / 1000.0,
        latency_ms=latency_ms,
        hit=hit, metadata={},
    )


def _log(samples=None, stimuli=None, responses=None) -> SessionLog:
    log = SessionLog(
        session_id="test", experiment_name="test",
        start_ts=0.0, end_ts=60.0, config={},
    )
    log.samples        = samples   or []
    log.stimulus_events = stimuli  or []
    log.response_events = responses or []
    return log


# ── SessionAnalytics: baseline score ─────────────────────────────────────────

class TestBaselineScore:
    def test_no_samples_in_window_returns_neutral(self):
        a = SessionAnalytics()
        # Samples all after the onset — none in baseline window
        samples = [_sample(ts=20.0, score=0.8) for _ in range(5)]
        result  = a._baseline_score(samples, onset_ts=10.0)
        assert result == pytest.approx(0.5)

    def test_uses_samples_before_onset(self):
        a       = SessionAnalytics()
        samples = [_sample(ts=float(i), score=0.9) for i in range(8)]
        result  = a._baseline_score(samples, onset_ts=8.0)
        assert result == pytest.approx(0.9, abs=0.01)

    def test_excludes_samples_after_onset(self):
        a = SessionAnalytics()
        before = [_sample(ts=float(i), score=0.9) for i in range(5)]
        after  = [_sample(ts=float(i + 20), score=0.1) for i in range(5)]
        result = a._baseline_score(before + after, onset_ts=8.0)
        assert result > 0.8   # dominated by the before-samples


# ── SessionAnalytics: during score ────────────────────────────────────────────

class TestDuringScore:
    def test_uses_samples_during_stimulus(self):
        a    = SessionAnalytics()
        ev   = _stim(onset=10.0, offset=12.0)
        samp = (
            [_sample(ts=t, score=0.3) for t in [10.1, 10.5, 11.0, 11.5, 11.9]]
            + [_sample(ts=t, score=0.9) for t in [9.0, 12.5]]   # outside window
        )
        result = a._during_score(samp, ev)
        assert result == pytest.approx(0.3, abs=0.01)

    def test_no_during_samples_returns_zero(self):
        a    = SessionAnalytics()
        ev   = _stim(onset=100.0, offset=101.0)
        samp = [_sample(ts=0.0, score=0.8)]
        result = a._during_score(samp, ev)
        assert result == pytest.approx(0.0)


# ── SessionAnalytics: recovery time ───────────────────────────────────────────

class TestRecoveryTime:
    def test_recovery_found(self):
        a    = SessionAnalytics()
        ev   = _stim(onset=10.0, offset=10.5)
        # Samples post-offset with low score first, then recovering
        samp = (
            [_sample(ts=10.6 + i * 0.5, score=0.4) for i in range(3)]
            + [_sample(ts=12.2, score=0.85)]   # baseline was 0.9 → 0.9*0.9=0.81 threshold
        )
        baseline = 0.9
        result   = a._recovery_time(samp, ev, baseline)
        assert result is not None
        assert result == pytest.approx(12.2 - 10.5, abs=0.05)

    def test_recovery_not_found_returns_none(self):
        a    = SessionAnalytics()
        ev   = _stim(onset=10.0, offset=10.5)
        # Score never recovers within RECOVERY_WINDOW_S=10
        samp = [_sample(ts=10.6 + i, score=0.2) for i in range(8)]
        result = a._recovery_time(samp, ev, baseline=0.9)
        assert result is None

    def test_immediate_recovery(self):
        a    = SessionAnalytics()
        ev   = _stim(onset=10.0, offset=10.5)
        # Very first post-offset sample is already above threshold
        samp = [_sample(ts=10.6, score=0.95)]
        result = a._recovery_time(samp, ev, baseline=0.9)
        assert result is not None
        assert result == pytest.approx(0.1, abs=0.05)


# ── SessionAnalytics: compute_trial_summaries ────────────────────────────────

class TestComputeTrialSummaries:
    def test_one_summary_per_stimulus(self):
        a   = SessionAnalytics()
        log = _log(
            stimuli=[_stim("s0"), _stim("s1", onset=20.0, offset=20.5)],
        )
        summaries = a.compute_trial_summaries(log)
        assert len(summaries) == 2

    def test_latency_from_response(self):
        a    = SessionAnalytics()
        log  = _log(
            stimuli=[_stim("s0", onset=10.0, offset=10.5)],
            responses=[_resp("s0", latency_ms=320.0)],
        )
        summaries = a.compute_trial_summaries(log)
        assert summaries[0].reaction_latency_ms == pytest.approx(320.0)

    def test_no_response_gives_none_latency(self):
        a    = SessionAnalytics()
        log  = _log(stimuli=[_stim("s0")])
        summaries = a.compute_trial_summaries(log)
        assert summaries[0].reaction_latency_ms is None

    def test_hit_false_when_no_response(self):
        a    = SessionAnalytics()
        log  = _log(stimuli=[_stim("s0")])
        summaries = a.compute_trial_summaries(log)
        assert summaries[0].hit is False

    def test_empty_log_returns_empty(self):
        a   = SessionAnalytics()
        log = _log()
        assert a.compute_trial_summaries(log) == []


# ── SessionAnalytics: aggregate_by_type ──────────────────────────────────────

class TestAggregateByType:
    def _summaries(self):
        return [
            TrialSummary("a0", 0, "ColorFlash", 10.0, 250.0,  300.0, "key_press", True,  0.8, 0.6, 1.0),
            TrialSummary("a1", 1, "ColorFlash", 20.0, 250.0,  400.0, "key_press", True,  0.8, 0.5, 1.5),
            TrialSummary("b0", 2, "ShapeStimulus", 30.0, 500.0, None, None, False, 0.8, 0.7, None),
        ]

    def test_groups_by_type(self):
        a   = SessionAnalytics()
        agg = a.aggregate_by_type(self._summaries())
        assert set(agg.keys()) == {"ColorFlash", "ShapeStimulus"}

    def test_n_trials_per_group(self):
        a   = SessionAnalytics()
        agg = a.aggregate_by_type(self._summaries())
        assert agg["ColorFlash"]["n_trials"]    == 2
        assert agg["ShapeStimulus"]["n_trials"] == 1

    def test_mean_latency_computed(self):
        a   = SessionAnalytics()
        agg = a.aggregate_by_type(self._summaries())
        assert agg["ColorFlash"]["mean_latency_ms"] == pytest.approx(350.0)

    def test_no_responses_gives_none_mean(self):
        a   = SessionAnalytics()
        agg = a.aggregate_by_type(self._summaries())
        assert agg["ShapeStimulus"]["mean_latency_ms"] is None

    def test_hit_rate(self):
        a   = SessionAnalytics()
        agg = a.aggregate_by_type(self._summaries())
        assert agg["ColorFlash"]["hit_rate"] == pytest.approx(1.0)
        assert agg["ShapeStimulus"]["hit_rate"] == pytest.approx(0.0)

    def test_empty_summaries_returns_empty(self):
        a   = SessionAnalytics()
        agg = a.aggregate_by_type([])
        assert agg == {}


# ── SessionAnalytics: overall_stats ──────────────────────────────────────────

class TestOverallStats:
    def test_empty_log_returns_empty(self):
        a   = SessionAnalytics()
        log = _log()
        assert a.overall_stats(log) == {}

    def test_duration_s_in_stats(self):
        a    = SessionAnalytics()
        samp = [_sample(ts=float(i)) for i in range(10)]
        log  = _log(samples=samp, stimuli=[_stim("s0")])
        stats = a.overall_stats(log)
        assert stats["duration_s"] == pytest.approx(60.0)
        assert stats["n_frames"]   == 10
        assert stats["n_trials"]   == 1

    def test_focused_pct(self):
        a    = SessionAnalytics()
        samp = (
            [_sample(ts=float(i), state="focused") for i in range(8)]
            + [_sample(ts=float(i + 10), state="distracted") for i in range(2)]
        )
        log   = _log(samples=samp)
        stats = a.overall_stats(log)
        assert stats["focused_pct"] == pytest.approx(80.0)

    def test_on_screen_pct(self):
        a    = SessionAnalytics()
        samp = (
            [_sample(ts=float(i), on_screen=True)  for i in range(6)]
            + [_sample(ts=float(i + 10), on_screen=False) for i in range(4)]
        )
        log   = _log(samples=samp)
        stats = a.overall_stats(log)
        assert stats["on_screen_pct"] == pytest.approx(60.0)

    def test_mean_key_latency_none_when_no_key_responses(self):
        a    = SessionAnalytics()
        samp = [_sample(ts=float(i)) for i in range(5)]
        log  = _log(samples=samp, responses=[_resp(rtype="gaze_shift")])
        stats = a.overall_stats(log)
        assert stats["mean_key_latency_ms"] is None


# ── EventRecorder: response detection ────────────────────────────────────────

class TestEventRecorderResponses:
    def _make_log(self):
        return SessionLog(
            session_id="r", experiment_name="test",
            start_ts=0.0, end_ts=None, config={},
        )

    def test_gaze_shift_detected_on_screen_to_off(self):
        log = self._make_log()
        rec = EventRecorder(log)
        ev  = _stim("s0", onset=5.0, offset=None)

        # First sample: on-screen, during stimulus
        s1 = _sample(ts=5.1, on_screen=True, stim_id="s0")
        rec.record_sample(s1, ev)
        assert len(log.response_events) == 0

        # Second sample: off-screen — should detect gaze_shift
        s2 = _sample(ts=5.3, on_screen=False, stim_id="s0")
        rec.record_sample(s2, ev)
        assert len(log.response_events) == 1
        assert log.response_events[0].response_type == "gaze_shift"
        assert log.response_events[0].latency_ms == pytest.approx(300.0, abs=5.0)

    def test_no_gaze_shift_without_active_stimulus(self):
        log = self._make_log()
        rec = EventRecorder(log)

        s1 = _sample(ts=1.0, on_screen=True)
        rec.record_sample(s1, None)   # no active stimulus
        s2 = _sample(ts=1.1, on_screen=False)
        rec.record_sample(s2, None)
        assert len(log.response_events) == 0

    def test_attention_change_detected(self):
        log = self._make_log()
        rec = EventRecorder(log)
        ev  = _stim("s0", onset=10.0, offset=None)

        s1 = _sample(ts=10.1, state="focused",    stim_id="s0")
        rec.record_sample(s1, ev)
        s2 = _sample(ts=10.4, state="distracted", stim_id="s0")
        rec.record_sample(s2, ev)

        attn_changes = [r for r in log.response_events if r.response_type == "attention_change"]
        assert len(attn_changes) == 1
        assert attn_changes[0].metadata["from_state"] == "focused"
        assert attn_changes[0].metadata["to_state"]   == "distracted"

    def test_key_response_recorded(self):
        log = self._make_log()
        rec = EventRecorder(log)
        ev  = _stim("react_00", onset=10.0, offset=None)

        resp = rec.record_key_response(ev, ts=10.350)
        assert resp.response_type == "key_press"
        assert resp.latency_ms    == pytest.approx(350.0, abs=1.0)
        assert len(log.response_events) == 1


# ── SessionExporter: smoke tests ──────────────────────────────────────────────

class TestSessionExporter:
    def _full_log(self):
        log = _log(
            stimuli=[_stim("s0"), _stim("s1", onset=20.0, offset=20.5)],
            responses=[_resp("s0", latency_ms=280.0)],
            samples=[_sample(ts=float(i)) for i in range(5)],
        )
        return log

    def test_to_json_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "session.json")
            SessionExporter.to_json(self._full_log(), path)
            assert os.path.exists(path)

    def test_to_json_valid_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.json")
            SessionExporter.to_json(self._full_log(), path)
            with open(path) as fh:
                doc = json.load(fh)
            assert "stimulus_events" in doc
            assert "response_events" in doc
            assert len(doc["stimulus_events"]) == 2

    def test_to_csv_creates_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "samples.csv")
            ep = os.path.join(tmp, "events.csv")
            SessionExporter.to_csv(self._full_log(), sp, ep)
            assert os.path.exists(sp)
            assert os.path.exists(ep)

    def test_to_csv_samples_row_count(self):
        import csv as csv_mod
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "samples.csv")
            ep = os.path.join(tmp, "events.csv")
            SessionExporter.to_csv(self._full_log(), sp, ep)
            with open(sp) as fh:
                rows = list(csv_mod.DictReader(fh))
            assert len(rows) == 5   # 5 samples in _full_log
