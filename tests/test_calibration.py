"""Tests for Day 12: calibration, survey, labeler, profiler, and comparison.

Coverage:
  TestBaseline          ( 8 tests) — UserProfile serialization, save/load, properties
  TestCollector         ( 7 tests) — add, valid_samples, open_eye_samples, duration
  TestProfiler          (12 tests) — compute profile, defaults, blink warmup, edge cases
  TestSurvey            ( 8 tests) — SurveyResponse derived properties, serialization
  TestLabeler           ( 8 tests) — SessionLabeler mapping, edge cases, save/load
  TestComparison        ( 8 tests) — SignalDelta, z-score, direction, magnitude
  Total                 (51 tests)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.calibration.baseline import (
    BlinkBaseline,
    EngagementBaseline,
    GazeBaseline,
    HeadPoseBaseline,
    UserProfile,
)
from src.calibration.collector import CalibrationCollector, CalibrationSample
from src.calibration.comparison import BaselineComparator, SignalDelta
from src.calibration.labeler import SessionLabel, SessionLabeler
from src.calibration.profiler import CalibrationProfiler
from src.calibration.survey import SurveyResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_sample(
    timestamp: float = 0.0,
    ear: float = 0.28,
    blink_rate: float = 15.0,
    head_yaw: float = 0.0,
    head_pitch: float = -2.0,
    gaze_h: float = 0.50,
    gaze_v: float = 0.50,
    engagement_score: float = 0.72,
    is_face_detected: bool = True,
    is_eyes_open: bool = True,
) -> CalibrationSample:
    return CalibrationSample(
        timestamp=        timestamp,
        ear=              ear,
        blink_rate=       blink_rate,
        head_yaw=         head_yaw,
        head_pitch=       head_pitch,
        gaze_h=           gaze_h,
        gaze_v=           gaze_v,
        engagement_score= engagement_score,
        is_face_detected= is_face_detected,
        is_eyes_open=     is_eyes_open,
    )


def _make_collector(n: int = 60, warmup_offset: float = 6.0) -> CalibrationCollector:
    """Return a collector with *n* valid face-detected samples spread over ~n seconds.

    warmup_offset ensures the first sample is past the 5 s warm-up threshold so
    blink rate is captured correctly.
    """
    c = CalibrationCollector()
    for i in range(n):
        c.add(_make_sample(timestamp=warmup_offset + i))
    return c


def _make_profile() -> UserProfile:
    return UserProfile(
        profile_id="test-abc",
        created_at="2026-01-01T00:00:00+00:00",
        n_frames=300,
        duration_s=30.0,
        blink=      BlinkBaseline(rate_per_min=15.0, rate_std=3.0, open_ear=0.28, open_ear_std=0.02),
        head=       HeadPoseBaseline(yaw_center=0.0, pitch_center=-2.0, yaw_range=5.0, pitch_range=4.0),
        gaze=       GazeBaseline(h_center=0.50, v_center=0.50, h_range=0.08, v_range=0.06),
        engagement= EngagementBaseline(score_mean=0.72, score_std=0.08),
    )


def _make_survey(
    eng=4, fat=2, dis=2, dif=3, recall=4
) -> SurveyResponse:
    return SurveyResponse(
        engagement_q=  eng,
        fatigue_q=     fat,
        distraction_q= dis,
        difficulty_q=  dif,
        recall_conf_q= recall,
    )


# ── TestBaseline ──────────────────────────────────────────────────────────────

class TestBaseline:
    def test_to_dict_round_trips(self):
        p = _make_profile()
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        assert p2.profile_id   == p.profile_id
        assert p2.blink.rate_per_min == p.blink.rate_per_min
        assert p2.gaze.h_center     == p.gaze.h_center

    def test_save_load_roundtrip(self, tmp_path):
        p = _make_profile()
        out = str(tmp_path / "profile.json")
        p.save(out)
        p2 = UserProfile.load(out)
        assert p2.profile_id == p.profile_id
        assert p2.engagement.score_mean == pytest.approx(p.engagement.score_mean)

    def test_try_load_returns_none_for_missing(self):
        assert UserProfile.try_load("/nonexistent/path.json") is None

    def test_try_load_returns_none_for_none(self):
        assert UserProfile.try_load(None) is None

    def test_personalized_low_blink_threshold(self):
        p = _make_profile()
        thr = p.personalized_low_blink_threshold
        # mean=15, std=3 → 15 - 2*3 = 9
        assert thr == pytest.approx(9.0)

    def test_personalized_low_blink_threshold_clips_to_zero(self):
        p = _make_profile()
        p.blink.rate_per_min = 2.0
        p.blink.rate_std     = 5.0
        assert p.personalized_low_blink_threshold >= 0.0

    def test_personalized_close_ear(self):
        p = _make_profile()
        ear = p.personalized_close_ear
        # mean=0.28, std=0.02 → 0.28 - 2.5*0.02 = 0.23
        assert ear == pytest.approx(0.23)

    def test_from_dict_missing_n_frames_defaults_to_zero(self):
        p = _make_profile()
        d = p.to_dict()
        del d["n_frames"]
        p2 = UserProfile.from_dict(d)
        assert p2.n_frames == 0


# ── TestCollector ─────────────────────────────────────────────────────────────

class TestCollector:
    def test_add_increases_n_samples(self):
        c = CalibrationCollector()
        c.add(_make_sample(timestamp=0.0))
        c.add(_make_sample(timestamp=1.0))
        assert c.n_samples == 2

    def test_valid_samples_filters_no_face(self):
        c = CalibrationCollector()
        c.add(_make_sample(is_face_detected=True))
        c.add(_make_sample(is_face_detected=False))
        assert len(c.valid_samples()) == 1

    def test_open_eye_samples_filters_closed_eyes(self):
        c = CalibrationCollector()
        c.add(_make_sample(is_face_detected=True, is_eyes_open=True))
        c.add(_make_sample(is_face_detected=True, is_eyes_open=False))
        c.add(_make_sample(is_face_detected=False, is_eyes_open=True))
        assert len(c.open_eye_samples()) == 1

    def test_duration_s_empty(self):
        c = CalibrationCollector()
        assert c.duration_s == 0.0

    def test_duration_s_two_samples(self):
        c = CalibrationCollector()
        c.add(_make_sample(timestamp=10.0))
        c.add(_make_sample(timestamp=40.0))
        assert c.duration_s == pytest.approx(30.0)

    def test_face_detection_rate(self):
        c = CalibrationCollector()
        c.add(_make_sample(is_face_detected=True))
        c.add(_make_sample(is_face_detected=False))
        assert c.face_detection_rate == pytest.approx(0.5)

    def test_clear_empties_samples(self):
        c = _make_collector(n=10)
        c.clear()
        assert c.n_samples == 0


# ── TestProfiler ──────────────────────────────────────────────────────────────

class TestProfiler:
    def test_compute_returns_user_profile(self):
        c = _make_collector(n=60)
        p = CalibrationProfiler.compute(c)
        assert isinstance(p, UserProfile)

    def test_profile_id_non_empty(self):
        c = _make_collector(n=60)
        p = CalibrationProfiler.compute(c)
        assert len(p.profile_id) > 0

    def test_custom_profile_id_preserved(self):
        c = _make_collector(n=60)
        p = CalibrationProfiler.compute(c, profile_id="myid")
        assert p.profile_id == "myid"

    def test_blink_rate_mean_near_injected_value(self):
        c = _make_collector(n=100)
        # All samples have blink_rate=15.0, after warmup_offset=6s
        p = CalibrationProfiler.compute(c)
        assert p.blink.rate_per_min == pytest.approx(15.0, abs=1.0)

    def test_blink_rate_warmup_excluded(self):
        # Samples before 5 s: rate=0 (warmup period); after: rate=15
        c = CalibrationCollector()
        for i in range(5):
            c.add(_make_sample(timestamp=float(i), blink_rate=0.0))    # warmup
        for i in range(5, 65):
            c.add(_make_sample(timestamp=float(i), blink_rate=15.0))   # post-warmup
        p = CalibrationProfiler.compute(c)
        # Must use 15.0, not a blend with 0.0 samples
        assert p.blink.rate_per_min == pytest.approx(15.0, abs=0.5)

    def test_open_ear_mean_near_injected_value(self):
        c = _make_collector(n=60)
        p = CalibrationProfiler.compute(c)
        assert p.blink.open_ear == pytest.approx(0.28, abs=0.01)

    def test_head_yaw_center_near_zero(self):
        c = _make_collector(n=60)
        p = CalibrationProfiler.compute(c)
        assert abs(p.head.yaw_center) < 1.0

    def test_gaze_h_center_near_half(self):
        c = _make_collector(n=60)
        p = CalibrationProfiler.compute(c)
        assert p.gaze.h_center == pytest.approx(0.50, abs=0.05)

    def test_engagement_score_mean_near_injected(self):
        c = _make_collector(n=60)
        p = CalibrationProfiler.compute(c)
        assert p.engagement.score_mean == pytest.approx(0.72, abs=0.01)

    def test_insufficient_frames_uses_defaults(self):
        c = CalibrationCollector()
        # Only 5 samples — below MIN_VALID_FRAMES=30
        for i in range(5):
            c.add(_make_sample(timestamp=float(i + 10)))
        p = CalibrationProfiler.compute(c)
        # Should return a profile without crashing; blink uses population default
        assert isinstance(p, UserProfile)
        assert p.blink.rate_per_min > 0

    def test_no_open_eye_samples_uses_ear_defaults(self):
        c = CalibrationCollector()
        for i in range(60):
            c.add(_make_sample(timestamp=float(i + 6), is_eyes_open=False))
        p = CalibrationProfiler.compute(c)
        # Population default EAR
        assert 0.20 <= p.blink.open_ear <= 0.35

    def test_n_frames_matches_collector(self):
        c = _make_collector(n=60)
        p = CalibrationProfiler.compute(c)
        assert p.n_frames == 60


# ── TestSurvey ────────────────────────────────────────────────────────────────

class TestSurvey:
    def test_engagement_level_high(self):
        assert _make_survey(eng=4).engagement_level == "HIGH"
        assert _make_survey(eng=5).engagement_level == "HIGH"

    def test_engagement_level_low(self):
        assert _make_survey(eng=1).engagement_level == "LOW"
        assert _make_survey(eng=2).engagement_level == "LOW"

    def test_engagement_level_none_for_middle(self):
        assert _make_survey(eng=3).engagement_level is None

    def test_fatigue_present(self):
        assert _make_survey(fat=4).fatigue_present is True
        assert _make_survey(fat=3).fatigue_present is False

    def test_high_distraction(self):
        assert _make_survey(dis=5).high_distraction is True
        assert _make_survey(dis=3).high_distraction is False

    def test_is_high_confidence(self):
        assert _make_survey(recall=4).is_high_confidence is True
        assert _make_survey(recall=3).is_high_confidence is False

    def test_to_dict_round_trips(self):
        s = _make_survey()
        d = s.to_dict()
        s2 = SurveyResponse.from_dict(d)
        assert s2.engagement_q  == s.engagement_q
        assert s2.recall_conf_q == s.recall_conf_q

    def test_save_load_roundtrip(self, tmp_path):
        s = _make_survey(eng=5, fat=1)
        out = str(tmp_path / "survey.json")
        s.save(out)
        s2 = SurveyResponse.load(out)
        assert s2.engagement_q == 5
        assert s2.fatigue_q    == 1


# ── TestLabeler ───────────────────────────────────────────────────────────────

class TestLabeler:
    def test_from_survey_engagement_high(self):
        label = SessionLabeler.from_survey(_make_survey(eng=5))
        assert label.label_for("engagement") == "HIGH"

    def test_from_survey_engagement_low(self):
        label = SessionLabeler.from_survey(_make_survey(eng=1))
        assert label.label_for("engagement") == "LOW"

    def test_from_survey_engagement_none_middle(self):
        label = SessionLabeler.from_survey(_make_survey(eng=3))
        assert label.label_for("engagement") is None

    def test_from_survey_fatigue_high(self):
        label = SessionLabeler.from_survey(_make_survey(fat=4))
        assert label.label_for("fatigue") == "HIGH"

    def test_from_survey_distraction_none_middle(self):
        label = SessionLabeler.from_survey(_make_survey(dis=3))
        assert label.label_for("distraction") is None

    def test_recall_conf_stored(self):
        label = SessionLabeler.from_survey(_make_survey(recall=2))
        assert label.recall_conf == 2
        assert label.is_high_confidence is False

    def test_save_load_roundtrip(self, tmp_path):
        s = _make_survey(eng=4, fat=1, dis=2)
        label = SessionLabeler.from_survey(s, session_dir="/tmp/sess1")
        out = str(tmp_path / "labels.json")
        label.save(out)
        label2 = SessionLabel.load(out)
        assert label2.label_for("engagement") == "HIGH"
        assert label2.session_dir == "/tmp/sess1"

    def test_try_load_missing_returns_none(self):
        assert SessionLabel.try_load("/nonexistent/labels.json") is None


# ── TestComparison ────────────────────────────────────────────────────────────

class TestComparison:
    def test_compare_produces_deltas(self):
        profile = _make_profile()
        cmp = BaselineComparator.compare(
            profile,
            session_dir="sess1",
            blink_rate=15.0,
            engagement_score=0.72,
        )
        assert "blink_rate"       in cmp.deltas
        assert "engagement_score" in cmp.deltas

    def test_z_score_zero_when_at_mean(self):
        profile = _make_profile()
        cmp = BaselineComparator.compare(
            profile, session_dir="x",
            blink_rate=15.0,   # exactly at mean
        )
        z = cmp.deltas["blink_rate"].z_score
        assert z == pytest.approx(0.0, abs=0.01)

    def test_z_score_positive_above_mean(self):
        profile = _make_profile()
        cmp = BaselineComparator.compare(
            profile, session_dir="x",
            blink_rate=18.0,   # above mean=15, std=3 → z=1.0
        )
        z = cmp.deltas["blink_rate"].z_score
        assert z == pytest.approx(1.0, abs=0.01)

    def test_z_score_negative_below_mean(self):
        profile = _make_profile()
        cmp = BaselineComparator.compare(
            profile, session_dir="x",
            blink_rate=9.0,   # below mean=15, std=3 → z=-2.0
        )
        z = cmp.deltas["blink_rate"].z_score
        assert z == pytest.approx(-2.0, abs=0.01)

    def test_direction_above(self):
        profile = _make_profile()
        cmp = BaselineComparator.compare(
            profile, session_dir="x",
            blink_rate=25.0,  # far above → "above"
        )
        assert cmp.deltas["blink_rate"].direction == "above"

    def test_direction_below(self):
        profile = _make_profile()
        cmp = BaselineComparator.compare(
            profile, session_dir="x",
            blink_rate=5.0,   # far below → "below"
        )
        assert cmp.deltas["blink_rate"].direction == "below"

    def test_magnitude_label_normal(self):
        delta = SignalDelta("x", 1.0, 1.0, 1.0, 0.3, "neutral")
        assert delta.magnitude == "normal"

    def test_none_signal_excluded(self):
        profile = _make_profile()
        cmp = BaselineComparator.compare(
            profile, session_dir="x",
            blink_rate=None,  # excluded
        )
        assert "blink_rate" not in cmp.deltas


class TestRunCalibrationImports:
    """Regression test: run_calibration.py must import cleanly with the current API."""

    def test_module_imports_cleanly(self):
        import importlib
        import run_calibration
        importlib.reload(run_calibration)
        assert hasattr(run_calibration, "run_webcam_calibration")
        assert hasattr(run_calibration, "run_survey_mode")

    def test_no_stale_engagement_scorer_import(self):
        import ast, pathlib
        src = pathlib.Path("run_calibration.py").read_text()
        assert "EngagementScorer" not in src
        assert "EyeMetricExtractor" not in src
        assert "GazeAnalyzer" not in src
        assert "HeadPoseEstimator" not in src

    def test_correct_api_references_present(self):
        import pathlib
        src = pathlib.Path("run_calibration.py").read_text()
        assert "AttentionScorer" in src
        assert "AttentionDetector" in src
        assert "extract_eye_measurements" in src
        assert "GazeDetector" in src
        assert "build_camera_matrix" in src
        assert "estimate_head_pose" in src
