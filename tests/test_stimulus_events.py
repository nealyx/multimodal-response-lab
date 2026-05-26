"""Unit tests for event schemas, stimulus rendering, and schedule building.

All tests are offline — no webcam, no MediaPipe, no config files required.
Rendering tests use small dummy frames (np.zeros) to verify shape contracts
and that the input is never modified.
"""

import random

import numpy as np
import pytest

from src.stimulus.events import (
    BehavioralSample, ResponseEvent, SessionLog, StimulusEvent, TrialSummary,
)
from src.stimulus.stimuli import (
    ColorFlash, MovingTarget, ReactionPrompt, ShapeStimulus,
)
from src.stimulus.presenter import build_schedule, ScheduledTrial


# ── Helpers ───────────────────────────────────────────────────────────────────

_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

def _frame():
    return _FRAME.copy()

def _stim_event(stim_id="s0", onset=10.0, offset=None, duration_ms=500.0):
    return StimulusEvent(
        stimulus_id=stim_id,
        trial_index=0,
        stimulus_type="ColorFlash",
        onset_ts=onset,
        offset_ts=offset,
        duration_ms=duration_ms,
        metadata={},
    )


# ── StimulusEvent ─────────────────────────────────────────────────────────────

class TestStimulusEvent:
    def test_actual_duration_ms_when_offset_set(self):
        ev = _stim_event(onset=10.0, offset=10.5)
        assert ev.actual_duration_ms == pytest.approx(500.0, abs=1.0)

    def test_actual_duration_ms_none_when_active(self):
        ev = _stim_event(onset=10.0, offset=None)
        assert ev.actual_duration_ms is None

    def test_is_active_without_offset(self):
        ev = _stim_event(offset=None)
        assert ev.is_active is True

    def test_is_active_with_offset(self):
        ev = _stim_event(offset=10.5)
        assert ev.is_active is False

    def test_fields_accessible(self):
        ev = _stim_event(stim_id="x", onset=5.0, duration_ms=250.0)
        assert ev.stimulus_id  == "x"
        assert ev.onset_ts     == pytest.approx(5.0)
        assert ev.duration_ms  == pytest.approx(250.0)
        assert ev.trial_index  == 0
        assert ev.stimulus_type == "ColorFlash"


# ── ResponseEvent ─────────────────────────────────────────────────────────────

class TestResponseEvent:
    def test_latency_ms_stored_correctly(self):
        ev = ResponseEvent(
            stimulus_id="s0", trial_index=0,
            response_type="key_press",
            response_ts=10.350, latency_ms=350.0, hit=True, metadata={},
        )
        assert ev.latency_ms == pytest.approx(350.0)
        assert ev.hit is True

    def test_fields_accessible(self):
        ev = ResponseEvent(
            stimulus_id="s0", trial_index=1,
            response_type="gaze_shift",
            response_ts=20.0, latency_ms=200.0, hit=True,
            metadata={"direction": "off_screen"},
        )
        assert ev.response_type == "gaze_shift"
        assert ev.metadata["direction"] == "off_screen"


# ── BehavioralSample ──────────────────────────────────────────────────────────

class TestBehavioralSample:
    def test_all_fields_accessible(self):
        s = BehavioralSample(
            frame_index=5, timestamp=12.3,
            active_stimulus_id="s0",
            engagement_state="focused", smoothed_score=0.85, confidence=0.90,
            focused_fraction=0.80,
            gaze_zone="center", is_on_screen=True, gaze_h=0.5, gaze_v=0.5,
            head_zone="focused", head_yaw=2.0, head_pitch=1.0,
            blink_state="open", mean_ear=0.30, blink_rate=14.0, is_fatigued=False,
        )
        assert s.engagement_state == "focused"
        assert s.is_on_screen is True
        assert s.blink_rate == pytest.approx(14.0)

    def test_none_stimulus_id_allowed(self):
        s = BehavioralSample(
            frame_index=0, timestamp=0.0,
            active_stimulus_id=None,
            engagement_state="unreliable", smoothed_score=0.5, confidence=0.0,
            focused_fraction=0.0,
            gaze_zone="unreliable", is_on_screen=False, gaze_h=0.5, gaze_v=0.5,
            head_zone="focused", head_yaw=0.0, head_pitch=0.0,
            blink_state="open", mean_ear=0.28, blink_rate=0.0, is_fatigued=False,
        )
        assert s.active_stimulus_id is None


# ── SessionLog ────────────────────────────────────────────────────────────────

class TestSessionLog:
    def test_duration_s_computed(self):
        log = SessionLog(
            session_id="t1", experiment_name="test",
            start_ts=100.0, end_ts=160.0, config={},
        )
        assert log.duration_s == pytest.approx(60.0)

    def test_duration_s_none_when_no_end(self):
        log = SessionLog(
            session_id="t2", experiment_name="test",
            start_ts=100.0, end_ts=None, config={},
        )
        assert log.duration_s is None

    def test_n_trials_counts_stimulus_events(self):
        log = SessionLog(
            session_id="t3", experiment_name="test",
            start_ts=0.0, end_ts=None, config={},
        )
        log.stimulus_events.append(_stim_event("a"))
        log.stimulus_events.append(_stim_event("b"))
        assert log.n_trials == 2

    def test_default_lists_empty(self):
        log = SessionLog(
            session_id="t4", experiment_name="test",
            start_ts=0.0, end_ts=None, config={},
        )
        assert log.stimulus_events == []
        assert log.response_events == []
        assert log.samples         == []


# ── ColorFlash rendering ──────────────────────────────────────────────────────

class TestColorFlashRender:
    def test_output_shape_matches_input(self):
        f = ColorFlash("id", 250.0, {}, color=(0, 100, 255), alpha=0.4)
        out = f.render(_frame(), 0.0)
        assert out.shape == _FRAME.shape

    def test_input_not_modified(self):
        f   = ColorFlash("id", 250.0, {})
        inp = _frame()
        ref = inp.copy()
        f.render(inp, 0.0)
        np.testing.assert_array_equal(inp, ref)

    def test_metadata_has_color(self):
        f = ColorFlash("id", 250.0, {}, color=(10, 20, 30))
        assert "color" in f.metadata()
        assert f.metadata()["color"] == [10, 20, 30]

    def test_region_none_full_frame(self):
        f = ColorFlash("id", 250.0, {}, region=None)
        assert f.metadata()["region"] is None

    def test_partial_region_render(self):
        f   = ColorFlash("id", 250.0, {}, region=(0, 0, 100, 100))
        out = f.render(_frame(), 0.0)
        assert out.shape == _FRAME.shape


# ── ShapeStimulus rendering ───────────────────────────────────────────────────

class TestShapeStimulusRender:
    @pytest.mark.parametrize("shape", ["circle", "square", "cross", "arrow_up", "arrow_right"])
    def test_shape_renders_correct_size(self, shape):
        s   = ShapeStimulus("id", 500.0, {}, shape=shape)
        out = s.render(_frame(), 0.0)
        assert out.shape == _FRAME.shape

    def test_input_not_modified(self):
        s   = ShapeStimulus("id", 500.0, {})
        inp = _frame()
        ref = inp.copy()
        s.render(inp, 0.0)
        np.testing.assert_array_equal(inp, ref)

    def test_metadata_has_shape(self):
        s = ShapeStimulus("id", 500.0, {}, shape="square")
        assert s.metadata()["shape"] == "square"

    @pytest.mark.parametrize("pos", ["center", "top", "bottom", "left", "right"])
    def test_all_positions_render(self, pos):
        s   = ShapeStimulus("id", 500.0, {}, position=pos)
        out = s.render(_frame(), 0.0)
        assert out.shape == _FRAME.shape


# ── MovingTarget rendering and position math ──────────────────────────────────

class TestMovingTargetRender:
    def test_output_shape_matches(self):
        t   = MovingTarget("id", 2000.0, {})
        out = t.render(_frame(), 500.0)
        assert out.shape == _FRAME.shape

    def test_input_not_modified(self):
        t   = MovingTarget("id", 2000.0, {})
        inp = _frame()
        ref = inp.copy()
        t.render(inp, 100.0)
        np.testing.assert_array_equal(inp, ref)

    @pytest.mark.parametrize("path", ["horizontal", "vertical", "diagonal", "circular"])
    def test_position_within_frame_bounds(self, path):
        t  = MovingTarget("id", 5000.0, {}, path=path, speed_px_per_s=100.0)
        w, h = 640, 480
        for ms in [0, 500, 1000, 2000, 4000]:
            cx, cy = t._position(ms, w, h)
            assert 0 <= cx < w, f"cx={cx} out of bounds at ms={ms}"
            assert 0 <= cy < h, f"cy={cy} out of bounds at ms={ms}"

    def test_metadata_has_path(self):
        t = MovingTarget("id", 2000.0, {}, path="circular")
        assert t.metadata()["path"] == "circular"


# ── ReactionPrompt rendering ──────────────────────────────────────────────────

class TestReactionPromptRender:
    def test_output_shape_matches(self):
        p   = ReactionPrompt("id", 2000.0, {})
        out = p.render(_frame(), 100.0)
        assert out.shape == _FRAME.shape

    def test_input_not_modified(self):
        p   = ReactionPrompt("id", 2000.0, {})
        inp = _frame()
        ref = inp.copy()
        p.render(inp, 0.0)
        np.testing.assert_array_equal(inp, ref)

    def test_metadata_has_response_key(self):
        p = ReactionPrompt("id", 2000.0, {}, response_key="space")
        assert p.metadata()["response_key"] == "space"

    def test_different_elapsed_renders_cleanly(self):
        p = ReactionPrompt("id", 2000.0, {})
        for ms in [0, 200, 800, 1500, 1999]:
            out = p.render(_frame(), float(ms))
            assert out.shape == _FRAME.shape


# ── build_schedule ────────────────────────────────────────────────────────────

_SCHED_CFG = {
    "experiment": {
        "baseline_duration_s":       2.0,
        "inter_stimulus_interval_s": 1.0,
        "isi_jitter_s":              0.0,   # deterministic for tests
        "randomise_order":           False,
        "stimuli": [
            {"type": "color_flash", "id_prefix": "flash", "n_trials": 3, "duration_ms": 250},
            {"type": "shape",       "id_prefix": "shape", "n_trials": 2, "duration_ms": 500},
        ],
    }
}


class TestBuildSchedule:
    def test_trial_count(self):
        random.seed(42)
        trials = build_schedule(_SCHED_CFG, t0=0.0)
        assert len(trials) == 5  # 3 + 2

    def test_timestamps_monotonically_increasing(self):
        random.seed(42)
        trials = build_schedule(_SCHED_CFG, t0=0.0)
        ts = [t.start_ts for t in trials]
        assert ts == sorted(ts)

    def test_first_trial_after_baseline(self):
        random.seed(42)
        trials = build_schedule(_SCHED_CFG, t0=100.0)
        baseline = _SCHED_CFG["experiment"]["baseline_duration_s"]
        assert trials[0].start_ts >= 100.0 + baseline

    def test_trial_indices_assigned(self):
        random.seed(42)
        trials = build_schedule(_SCHED_CFG, t0=0.0)
        indices = [t.trial_index for t in trials]
        assert indices == list(range(5))

    def test_stimulus_ids_unique(self):
        random.seed(42)
        trials = build_schedule(_SCHED_CFG, t0=0.0)
        ids = [t.stimulus.stimulus_id for t in trials]
        assert len(ids) == len(set(ids))

    def test_empty_stimuli_returns_empty(self):
        cfg = {"experiment": {"stimuli": [], "baseline_duration_s": 2.0,
                              "inter_stimulus_interval_s": 1.0, "isi_jitter_s": 0.0}}
        trials = build_schedule(cfg, t0=0.0)
        assert trials == []

    def test_unknown_stimulus_type_skipped(self):
        cfg = {
            "experiment": {
                "baseline_duration_s": 2.0, "inter_stimulus_interval_s": 1.0,
                "isi_jitter_s": 0.0, "stimuli": [
                    {"type": "unknown_xyz", "n_trials": 1, "duration_ms": 100},
                    {"type": "color_flash", "n_trials": 1, "duration_ms": 100},
                ],
            }
        }
        trials = build_schedule(cfg, t0=0.0)
        assert len(trials) == 1
        assert trials[0].stimulus.stimulus_type == "ColorFlash"
