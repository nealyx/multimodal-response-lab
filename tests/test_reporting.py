"""Unit tests for the session reporting layer (Day 8).

All tests are offline — no webcam, no config files, no matplotlib display.
Tests that exercise chart generation are skipped if matplotlib is unavailable
(pytest.importorskip), though matplotlib IS available in the project venv.

Synthetic data helpers build minimal SessionLog / BehavioralSample objects
so tests are self-contained and fast.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.reporting.metrics import (
    SessionMetrics, SignalSummary, summarize,
    EngagementMetrics, GazeMetrics, HeadMetrics, BlinkMetrics, StimulusTypeMetrics,
)
from src.reporting.reporter import SessionReporter
from src.reporting.loader import load_session_dir, load_session_json, _bool, _nonempty
from src.reporting.html_report import generate_html, _dominant_state
from src.stimulus.events import (
    BehavioralSample, ResponseEvent, SessionLog, StimulusEvent, TrialSummary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample(
    ts: float = 0.0,
    score: float = 0.75,
    state: str = "focused",
    on_screen: bool = True,
    stim_id: str = None,
    yaw: float = 2.0,
    pitch: float = 1.0,
    blink_rate: float = 14.0,
    ear: float = 0.30,
    fatigued: bool = False,
    gaze_h: float = 0.5,
    gaze_v: float = 0.5,
    head_zone: str = "focused",
) -> BehavioralSample:
    return BehavioralSample(
        frame_index=0, timestamp=ts,
        active_stimulus_id=stim_id,
        engagement_state=state, smoothed_score=score, confidence=0.8,
        focused_fraction=0.7,
        gaze_zone="center", is_on_screen=on_screen, gaze_h=gaze_h, gaze_v=gaze_v,
        head_zone=head_zone, head_yaw=yaw, head_pitch=pitch,
        blink_state="open", mean_ear=ear, blink_rate=blink_rate,
        is_fatigued=fatigued,
    )


def _log(
    samples=None, stimuli=None, responses=None, summaries=None,
    session_id="test", start_ts=0.0, end_ts=60.0,
) -> SessionLog:
    log = SessionLog(
        session_id=session_id, experiment_name="test",
        start_ts=start_ts, end_ts=end_ts, config={},
    )
    log.samples         = samples   or []
    log.stimulus_events = stimuli   or []
    log.response_events = responses or []
    log.trial_summaries = summaries or []
    return log


def _make_metrics(
    focused=0.6, drifting=0.2, distracted=0.1, fatigued=0.05, unreliable=0.05,
    mean_score=0.70,
) -> "EngagementMetrics":
    return EngagementMetrics(
        score_summary=summarize([mean_score] * 100),
        focused_fraction=focused, drifting_fraction=drifting,
        distracted_fraction=distracted, fatigued_fraction=fatigued,
        unreliable_fraction=unreliable,
    )


# ── SignalSummary / summarize ─────────────────────────────────────────────────

class TestSummarize:
    def test_empty_returns_zeros(self):
        s = summarize([])
        assert s.mean == 0.0
        assert s.std  == 0.0
        assert s.n    == 0

    def test_constant_array(self):
        s = summarize([5.0] * 10)
        assert s.mean == pytest.approx(5.0)
        assert s.std  == pytest.approx(0.0)
        assert s.min_val == pytest.approx(5.0)
        assert s.max_val == pytest.approx(5.0)
        assert s.n == 10

    def test_known_values(self):
        s = summarize([0.0, 0.25, 0.50, 0.75, 1.0])
        assert s.mean    == pytest.approx(0.5)
        assert s.min_val == pytest.approx(0.0)
        assert s.max_val == pytest.approx(1.0)
        assert s.p25     == pytest.approx(0.25, abs=0.01)
        assert s.p75     == pytest.approx(0.75, abs=0.01)
        assert s.n == 5

    def test_single_element(self):
        s = summarize([42.0])
        assert s.mean == pytest.approx(42.0)
        assert s.n    == 1


# ── SessionReporter ───────────────────────────────────────────────────────────

class TestSessionReporter:
    def _reporter(self):
        return SessionReporter()

    def test_returns_session_metrics(self):
        r   = self._reporter()
        log = _log(samples=[_sample(ts=float(i)) for i in range(10)])
        m   = r.compute(log)
        assert isinstance(m, SessionMetrics)

    def test_empty_log_does_not_crash(self):
        r = self._reporter()
        m = r.compute(_log())
        assert m.n_frames == 0
        assert m.engagement.focused_fraction == pytest.approx(0.0)

    def test_duration_s_matches_log(self):
        r   = self._reporter()
        log = _log(start_ts=100.0, end_ts=160.0)
        m   = r.compute(log)
        assert m.duration_s == pytest.approx(60.0)

    def test_n_frames_matches_samples(self):
        r    = self._reporter()
        samp = [_sample(ts=float(i)) for i in range(15)]
        m    = r.compute(_log(samples=samp))
        assert m.n_frames == 15

    def test_n_trials_matches_stimulus_events(self):
        r   = self._reporter()
        ev  = StimulusEvent("s0", 0, "ColorFlash", 10.0, 10.25, 250.0, {})
        log = _log(stimuli=[ev])
        m   = r.compute(log)
        assert m.n_trials == 1

    def test_focused_fraction_correct(self):
        r    = self._reporter()
        samp = (
            [_sample(ts=float(i), state="focused")    for i in range(8)]
            + [_sample(ts=float(i+10), state="distracted") for i in range(2)]
        )
        m = r.compute(_log(samples=samp))
        assert m.engagement.focused_fraction == pytest.approx(0.8, abs=0.01)

    def test_distracted_fraction_correct(self):
        r    = self._reporter()
        samp = (
            [_sample(ts=float(i), state="focused")    for i in range(6)]
            + [_sample(ts=float(i+10), state="distracted") for i in range(4)]
        )
        m = r.compute(_log(samples=samp))
        assert m.engagement.distracted_fraction == pytest.approx(0.4, abs=0.01)

    def test_fatigued_fraction_correct(self):
        r    = self._reporter()
        samp = (
            [_sample(ts=float(i), state="focused")  for i in range(7)]
            + [_sample(ts=float(i+10), state="fatigued", fatigued=True) for i in range(3)]
        )
        m = r.compute(_log(samples=samp))
        assert m.engagement.fatigued_fraction == pytest.approx(0.3, abs=0.01)
        assert m.blink.fatigue_fraction       == pytest.approx(0.3, abs=0.01)

    def test_on_screen_fraction_correct(self):
        r    = self._reporter()
        samp = (
            [_sample(ts=float(i), on_screen=True)  for i in range(7)]
            + [_sample(ts=float(i+10), on_screen=False) for i in range(3)]
        )
        m = r.compute(_log(samples=samp))
        assert m.gaze.on_screen_fraction == pytest.approx(0.7, abs=0.01)

    def test_gaze_h_summary_mean(self):
        r    = self._reporter()
        samp = [_sample(ts=float(i), gaze_h=0.6) for i in range(10)]
        m    = r.compute(_log(samples=samp))
        assert m.gaze.gaze_h_summary.mean == pytest.approx(0.6, abs=0.01)

    def test_head_focused_fraction(self):
        r    = self._reporter()
        samp = (
            [_sample(ts=float(i), head_zone="focused")      for i in range(8)]
            + [_sample(ts=float(i+10), head_zone="looking_away") for i in range(2)]
        )
        m = r.compute(_log(samples=samp))
        assert m.head.head_focused_fraction == pytest.approx(0.8, abs=0.01)

    def test_blink_rate_summary(self):
        r    = self._reporter()
        samp = [_sample(ts=float(i), blink_rate=15.0) for i in range(10)]
        m    = r.compute(_log(samples=samp))
        assert m.blink.blink_rate_summary.mean == pytest.approx(15.0, abs=0.01)

    def test_per_stimulus_type_computed(self):
        r   = self._reporter()
        ts_obj = TrialSummary("s0", 0, "ColorFlash", 10.0, 250.0,
                              300.0, "key_press", True, 0.8, 0.7, 1.0)
        log = _log(summaries=[ts_obj])
        m   = r.compute(log)
        assert "ColorFlash" in m.stimulus_types
        assert m.stimulus_types["ColorFlash"].n_trials == 1

    def test_stimulus_score_delta(self):
        r   = self._reporter()
        ts1 = TrialSummary("a0", 0, "ColorFlash", 10.0, 250.0,
                           None, None, False, 0.80, 0.65, None)
        ts2 = TrialSummary("a1", 1, "ColorFlash", 20.0, 250.0,
                           None, None, False, 0.70, 0.55, None)
        m   = r.compute(_log(summaries=[ts1, ts2]))
        cf  = m.stimulus_types["ColorFlash"]
        # mean_during=0.60, mean_baseline=0.75, delta=-0.15
        assert cf.score_delta == pytest.approx(0.60 - 0.75, abs=0.01)

    def test_key_latencies_extracted(self):
        r    = self._reporter()
        resp = ResponseEvent("s0", 0, "key_press", 10.3, 300.0, True, {})
        log  = _log(responses=[resp])
        m    = r.compute(log)
        assert m.reaction_latencies_ms == pytest.approx([300.0])

    def test_gaze_latencies_not_in_key_list(self):
        r    = self._reporter()
        resp = ResponseEvent("s0", 0, "gaze_shift", 10.2, 200.0, True, {})
        log  = _log(responses=[resp])
        m    = r.compute(log)
        assert m.reaction_latencies_ms == []


# ── Loader utilities ──────────────────────────────────────────────────────────

class TestLoaderUtils:
    def test_bool_true_variants(self):
        for v in ("True", "true", "1", "yes"):
            assert _bool(v) is True

    def test_bool_false_variants(self):
        for v in ("False", "false", "0", "no", ""):
            assert _bool(v) is False

    def test_nonempty_returns_none_for_empty(self):
        assert _nonempty("") is None
        assert _nonempty("  ") is None

    def test_nonempty_returns_string_when_present(self):
        assert _nonempty("flash_00") == "flash_00"


class TestLoaderFromDisk:
    def _write_session(self, tmp: str, n_samples: int = 5) -> str:
        """Write a minimal session dir and return its path."""
        session_dir = os.path.join(tmp, "test_session")
        os.makedirs(session_dir)

        doc = {
            "session_id":      "test_session",
            "experiment_name": "test",
            "start_ts":        100.0,
            "end_ts":          160.0,
            "duration_s":      60.0,
            "config":          {},
            "stimulus_events": [],
            "response_events": [],
            "trial_summaries": [],
            "aggregate_stats": {},
            "overall_stats":   {},
        }
        with open(os.path.join(session_dir, "session_log.json"), "w") as fh:
            json.dump(doc, fh)

        with open(os.path.join(session_dir, "samples.csv"), "w", newline="") as fh:
            fh.write(
                "frame_index,timestamp,active_stimulus_id,engagement_state,"
                "smoothed_score,confidence,focused_fraction,gaze_zone,is_on_screen,"
                "gaze_h,gaze_v,head_zone,head_yaw,head_pitch,blink_state,"
                "mean_ear,blink_rate,is_fatigued\n"
            )
            for i in range(n_samples):
                fh.write(
                    f"{i},{100.0 + i * 0.033},,focused,0.82,0.9,0.7,center,"
                    f"True,0.5,0.5,focused,2.0,1.0,open,0.30,14.0,False\n"
                )
        return session_dir

    def test_load_session_dir_returns_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            sd  = self._write_session(tmp)
            log = load_session_dir(sd)
            assert log.session_id == "test_session"
            assert len(log.samples) == 5

    def test_load_samples_csv_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            sd  = self._write_session(tmp, n_samples=3)
            log = load_session_dir(sd)
            assert log.samples[0].engagement_state == "focused"
            assert log.samples[0].is_on_screen is True
            assert log.samples[0].smoothed_score == pytest.approx(0.82)

    def test_load_json_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            sd  = self._write_session(tmp)
            log = load_session_json(os.path.join(sd, "session_log.json"))
            assert log.session_id == "test_session"
            assert log.samples    == []  # no CSV loaded

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_session_dir("/nonexistent/path/that/does/not/exist")


# ── HTML report ───────────────────────────────────────────────────────────────

class TestHtmlReport:
    def _metrics(self) -> SessionMetrics:
        eng = _make_metrics()
        empty_sig = summarize([])
        return SessionMetrics(
            session_id="test_s", experiment_name="demo",
            generated_at="2025-01-01T00:00:00+00:00",
            duration_s=60.0, n_frames=1800, n_trials=5, n_responses=3,
            engagement=eng,
            gaze=GazeMetrics(0.82, empty_sig, empty_sig),
            head=HeadMetrics(0.78, empty_sig, empty_sig),
            blink=BlinkMetrics(empty_sig, 0.02, empty_sig),
            stimulus_types={},
            reaction_latencies_ms=[320.0, 410.0, 280.0],
            config_snapshot={},
        )

    def test_returns_html_string(self):
        m    = self._metrics()
        html = generate_html(m, {})
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")

    def test_contains_session_id(self):
        m    = self._metrics()
        html = generate_html(m, {})
        assert "test_s" in html

    def test_contains_metrics_values(self):
        m    = self._metrics()
        html = generate_html(m, {})
        assert "60.0" in html   # duration
        assert "1800" in html   # n_frames

    def test_no_charts_produces_valid_html(self):
        m    = self._metrics()
        html = generate_html(m, {})
        assert "</html>" in html

    def test_chart_embedded_as_base64(self):
        m    = self._metrics()
        html = generate_html(m, {"timeline": b"FAKEPNG"})
        assert "data:image/png;base64," in html

    def test_dominant_state_focused(self):
        e = _make_metrics(focused=0.7, drifting=0.1, distracted=0.1, fatigued=0.05, unreliable=0.05)
        assert _dominant_state(e) == "focused"

    def test_dominant_state_distracted(self):
        e = _make_metrics(focused=0.1, drifting=0.1, distracted=0.6, fatigued=0.1, unreliable=0.1)
        assert _dominant_state(e) == "distracted"


# ── Charts (requires matplotlib) ─────────────────────────────────────────────

class TestCharts:
    @pytest.fixture(autouse=True)
    def _require_mpl(self):
        pytest.importorskip("matplotlib")

    def test_engagement_timeline_returns_figure(self):
        from src.reporting.charts import engagement_timeline
        import matplotlib.pyplot as plt
        samp = [_sample(ts=float(i) * 0.033, score=0.7 + 0.1 * (i % 3)) for i in range(30)]
        fig  = engagement_timeline(samp, start_ts=0.0)
        assert fig is not None
        plt.close(fig)

    def test_state_distribution_returns_figure(self):
        from src.reporting.charts import state_distribution
        import matplotlib.pyplot as plt
        e   = _make_metrics()
        fig = state_distribution(e)
        assert fig is not None
        plt.close(fig)

    def test_signal_channels_returns_figure(self):
        from src.reporting.charts import signal_channels
        import matplotlib.pyplot as plt
        samp = [_sample(ts=float(i) * 0.033) for i in range(30)]
        fig  = signal_channels(samp, start_ts=0.0)
        assert fig is not None
        plt.close(fig)

    def test_engagement_timeline_empty_returns_none(self):
        from src.reporting.charts import engagement_timeline
        assert engagement_timeline([]) is None

    def test_stimulus_comparison_empty_returns_none(self):
        from src.reporting.charts import stimulus_comparison
        assert stimulus_comparison({}) is None

    def test_reaction_latency_no_data_returns_none(self):
        from src.reporting.charts import reaction_latency
        empty = {"ColorFlash": StimulusTypeMetrics(
            "ColorFlash", 3, 0.0, None, None, None, 0.7, 0.6, -0.1, None
        )}
        assert reaction_latency(empty) is None

    def test_figure_to_png_bytes_returns_bytes(self):
        from src.reporting.charts import engagement_timeline, figure_to_png_bytes
        import matplotlib.pyplot as plt
        samp = [_sample(ts=float(i) * 0.033) for i in range(20)]
        fig  = engagement_timeline(samp)
        png  = figure_to_png_bytes(fig)
        assert isinstance(png, bytes)
        assert len(png) > 100    # non-trivial PNG
        plt.close(fig)

    def test_generate_all_returns_dict(self):
        from src.reporting.charts import generate_all
        import matplotlib.pyplot as plt
        samp = [_sample(ts=float(i) * 0.033) for i in range(30)]
        log  = _log(samples=samp)
        r    = SessionReporter()
        m    = r.compute(log)
        figs = generate_all(log, m)
        assert isinstance(figs, dict)
        for fig in figs.values():
            plt.close(fig)
