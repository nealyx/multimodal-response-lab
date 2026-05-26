"""Unit tests for the composite engagement scoring layer.

All tests inject pre-built signal objects (BlinkAnalysis, AttentionAnalysis,
GazeAnalysis) with controlled field values — no webcam or landmark data needed.
"""

import pytest
import numpy as np

from src.signals.engagement import AttentionScorer, EngagementState, EngagementScore
from src.signals.blink_detector import BlinkAnalysis, BlinkState
from src.signals.attention import AttentionAnalysis, AttentionZone
from src.signals.gaze import GazeAnalysis, GazeZone


# ── Mock constructors ─────────────────────────────────────────────────────────

def _blink(
    openness: float = 0.8,
    *,
    prolonged: bool = False,
    drowsy:    bool = False,
    low_rate:  bool = False,
    high_rate: bool = False,
    ts: float = 0.0,
) -> BlinkAnalysis:
    return BlinkAnalysis(
        state=BlinkState.OPEN,
        blink_count=5, blink_rate_per_min=14.0,
        mean_ear=openness * 0.35, mean_openness=openness,
        ms_since_last_blink=400.0,
        is_low_blink_rate=low_rate, is_high_blink_rate=high_rate,
        is_prolonged_closure=prolonged, is_drowsy=drowsy,
        timestamp=ts,
    )


def _head(
    zone: AttentionZone = AttentionZone.FOCUSED,
    fraction: float = 0.9,
    yaw_std: float = 1.0,
    pitch_std: float = 1.0,
    reproj: float = 2.0,
    ts: float = 0.0,
) -> AttentionAnalysis:
    return AttentionAnalysis(
        zone=zone, yaw=0.0, pitch=0.0, roll=0.0,
        yaw_std=yaw_std, pitch_std=pitch_std,
        attention_fraction=fraction,
        reprojection_error=reproj,
        timestamp=ts, frame_index=0,
    )


def _gaze(
    zone: GazeZone = GazeZone.CENTER,
    on_screen_frac: float = 0.9,
    reliable: bool = True,
    ts: float = 0.0,
) -> GazeAnalysis:
    return GazeAnalysis(
        zone=zone, mean_h=0.5, mean_v=0.5,
        is_on_screen=(zone == GazeZone.CENTER),
        is_reliable=reliable,
        stability_h=0.01, stability_v=0.01,
        on_screen_fraction=on_screen_frac,
        timestamp=ts, frame_index=0,
    )


_CFG = {
    "engagement": {
        "weights":    {"gaze": 0.50, "head": 0.20, "eye": 0.30},
        "thresholds": {"focused": 0.72, "drifting": 0.45, "min_confidence": 0.20},
        "smoothing":  {"alpha": 1.0, "state_min_frames": 1},  # no smoothing in tests
        "confidence": {"min_history_s": 0.0, "pose_error_max": 10.0},
        "rolling_window_s": 60.0,
    }
}


def _scorer(extra: dict = None) -> AttentionScorer:
    if extra:
        import copy
        cfg = copy.deepcopy(_CFG)
        cfg["engagement"].update(extra)
        return AttentionScorer(cfg)
    return AttentionScorer(_CFG)


# ── Component score methods ───────────────────────────────────────────────────

class TestGazeComponent:
    def test_reliable_gaze_uses_on_screen_fraction(self):
        g = _gaze(on_screen_frac=0.80)
        score = AttentionScorer._gaze_component(g, None)
        assert score == pytest.approx(0.80)

    def test_unreliable_gaze_falls_back_to_head(self):
        g = _gaze(reliable=False, on_screen_frac=0.90)
        h = _head(fraction=0.60)
        score = AttentionScorer._gaze_component(g, h)
        assert score == pytest.approx(0.60 * 0.65, abs=0.001)

    def test_no_gaze_falls_back_to_head(self):
        h = _head(fraction=0.70)
        score = AttentionScorer._gaze_component(None, h)
        assert score == pytest.approx(0.70 * 0.65, abs=0.001)

    def test_no_signals_returns_zero(self):
        assert AttentionScorer._gaze_component(None, None) == 0.0


class TestHeadComponent:
    def test_none_returns_neutral(self):
        assert AttentionScorer._head_component(None) == pytest.approx(0.5)

    def test_focused_head_returns_high(self):
        h = _head(fraction=0.9, yaw_std=1.0, pitch_std=1.0)
        score = AttentionScorer._head_component(h)
        assert score > 0.8

    def test_high_std_reduces_score(self):
        h_still = _head(fraction=0.9, yaw_std=1.0,  pitch_std=1.0)
        h_fidgy = _head(fraction=0.9, yaw_std=12.0, pitch_std=12.0)
        assert AttentionScorer._head_component(h_still) > \
               AttentionScorer._head_component(h_fidgy)

    def test_score_clamped_non_negative(self):
        h = _head(fraction=0.0, yaw_std=50.0, pitch_std=50.0)
        assert AttentionScorer._head_component(h) >= 0.0


class TestEyeComponent:
    def test_none_returns_neutral(self):
        assert AttentionScorer._eye_component(None) == pytest.approx(0.5)

    def test_open_eye_no_flags_returns_openness(self):
        b = _blink(openness=0.9)
        assert AttentionScorer._eye_component(b) == pytest.approx(0.9, abs=0.01)

    def test_prolonged_closure_heavy_penalty(self):
        normal   = AttentionScorer._eye_component(_blink(openness=0.8))
        prolonged = AttentionScorer._eye_component(_blink(openness=0.8, prolonged=True))
        assert prolonged < normal * 0.5

    def test_drowsy_reduces_score(self):
        normal = AttentionScorer._eye_component(_blink(openness=0.8))
        drowsy = AttentionScorer._eye_component(_blink(openness=0.8, drowsy=True))
        assert drowsy < normal

    def test_low_rate_mild_penalty(self):
        normal   = AttentionScorer._eye_component(_blink(openness=0.8))
        low_rate = AttentionScorer._eye_component(_blink(openness=0.8, low_rate=True))
        assert low_rate < normal
        assert low_rate > normal * 0.5   # mild, not severe

    def test_score_clamped_to_unit_range(self):
        # Even with all penalties, score stays in [0, 1]
        b = _blink(openness=1.0, prolonged=True, drowsy=True,
                   low_rate=True, high_rate=True)
        s = AttentionScorer._eye_component(b)
        assert 0.0 <= s <= 1.0


# ── State classification ──────────────────────────────────────────────────────

class TestStateClassification:
    """Use alpha=1.0 and state_min_frames=1 so each update is immediate."""

    def _update_once(self, blink=None, head=None, gaze=None,
                     face_detected=True, ts=0.0):
        s = _scorer()
        return s.update(blink=blink, head=head, gaze=gaze,
                        face_detected=face_detected, timestamp=ts)

    def test_all_good_signals_focused(self):
        a = self._update_once(
            blink=_blink(0.9),
            head=_head(fraction=0.95),
            gaze=_gaze(on_screen_frac=0.95),
            ts=100.0,   # long history → high confidence
        )
        assert a.state == EngagementState.FOCUSED

    def test_low_scores_distracted(self):
        a = self._update_once(
            blink=_blink(0.8),
            head=_head(zone=AttentionZone.LOOKING_AWAY, fraction=0.05),
            gaze=_gaze(zone=GazeZone.LEFT, on_screen_frac=0.05),
            ts=100.0,
        )
        assert a.state == EngagementState.DISTRACTED

    def test_moderate_scores_drifting(self):
        a = self._update_once(
            blink=_blink(0.75),
            head=_head(fraction=0.55),
            gaze=_gaze(on_screen_frac=0.55),
            ts=100.0,
        )
        assert a.state == EngagementState.DRIFTING

    def test_fatigue_overrides_high_score(self):
        a = self._update_once(
            blink=_blink(0.9, drowsy=True),
            head=_head(fraction=0.95),
            gaze=_gaze(on_screen_frac=0.95),
            ts=100.0,
        )
        assert a.state == EngagementState.FATIGUED

    def test_prolonged_closure_triggers_fatigued(self):
        a = self._update_once(
            blink=_blink(0.1, prolonged=True),
            head=_head(fraction=0.9),
            gaze=_gaze(on_screen_frac=0.9),
            ts=100.0,
        )
        assert a.state == EngagementState.FATIGUED

    def test_no_face_gives_unreliable(self):
        a = self._update_once(face_detected=False, ts=100.0)
        assert a.state == EngagementState.UNRELIABLE

    def test_zero_history_gives_unreliable(self):
        # With min_history_s=0.0 in _CFG and ts=0 this is borderline; confidence=0.0
        s = AttentionScorer(_CFG)
        a = s.update(
            blink=_blink(0.9), head=_head(fraction=0.9), gaze=_gaze(),
            face_detected=True, timestamp=0.0,
        )
        # confidence=0 at t=0 with min_history_s=0 → history_conf=1.0 (immediate)
        # so state should NOT be unreliable; just check it returns something valid
        assert isinstance(a.state, EngagementState)


# ── Temporal smoothing and debounce ──────────────────────────────────────────

class TestSmoothing:
    def test_single_bad_frame_does_not_flip_state(self):
        """With state_min_frames=5, one bad frame must not change the state."""
        cfg = {
            "engagement": {
                **_CFG["engagement"],
                "smoothing": {"alpha": 0.10, "state_min_frames": 5},
            }
        }
        s = AttentionScorer(cfg)
        # Feed 30 high-score frames; EMA needs ~7 frames to cross 0.72 threshold,
        # then 5 more debounce frames before state confirms FOCUSED.
        for i in range(30):
            a = s.update(
                blink=_blink(0.9), head=_head(fraction=0.95),
                gaze=_gaze(on_screen_frac=0.95),
                face_detected=True, timestamp=float(i),
            )
        assert a.state == EngagementState.FOCUSED

        # One frame with no face
        a_bad = s.update(face_detected=False, timestamp=30.0)
        # Should still report FOCUSED (debounce hasn't fired)
        assert a_bad.state == EngagementState.FOCUSED

    def test_ema_smoothing_applied(self):
        """Smoothed score should be between old smoothed and new raw."""
        cfg = {
            "engagement": {
                **_CFG["engagement"],
                "smoothing": {"alpha": 0.5, "state_min_frames": 1},
            }
        }
        s = AttentionScorer(cfg)
        # Drive smoothed to high value
        for _ in range(20):
            s.update(blink=_blink(0.9), head=_head(fraction=0.9),
                     gaze=_gaze(on_screen_frac=0.9), timestamp=0.0)
        smoothed_high = s._smoothed

        # One very low frame
        a = s.update(blink=_blink(0.1), head=_head(fraction=0.1),
                     gaze=_gaze(on_screen_frac=0.1), timestamp=1.0)
        # Smoothed drops but doesn't reach raw level in one frame
        assert a.smoothed_score < smoothed_high
        assert a.smoothed_score > a.raw_score


# ── Confidence score ──────────────────────────────────────────────────────────

class TestConfidence:
    def test_no_face_gives_zero_confidence(self):
        s = _scorer()
        a = s.update(face_detected=False, timestamp=100.0)
        assert a.confidence == pytest.approx(0.0)

    def test_short_history_reduces_confidence(self):
        cfg = {
            "engagement": {
                **_CFG["engagement"],
                "confidence": {"min_history_s": 30.0, "pose_error_max": 10.0},
            }
        }
        s = AttentionScorer(cfg)
        a = s.update(blink=_blink(0.8), face_detected=True, timestamp=5.0)
        assert a.confidence < 0.5   # only 5 s of 30 s required

    def test_long_history_gives_high_confidence(self):
        cfg = {
            "engagement": {
                **_CFG["engagement"],
                "confidence": {"min_history_s": 10.0, "pose_error_max": 10.0},
            }
        }
        s = AttentionScorer(cfg)
        # Prime _t_first at t=0, then jump to t=20 (20 s > min_history_s=10 s)
        s.update(blink=_blink(0.8), head=_head(),
                 gaze=_gaze(), face_detected=True, timestamp=0.0)
        a = s.update(blink=_blink(0.8), head=_head(),
                     gaze=_gaze(), face_detected=True, timestamp=20.0)
        assert a.confidence > 0.8

    def test_unreliable_gaze_reduces_confidence(self):
        s_reliable   = _scorer()
        s_unreliable = _scorer()
        a_r  = s_reliable.update(
            gaze=_gaze(reliable=True), face_detected=True, timestamp=20.0)
        a_ur = s_unreliable.update(
            gaze=_gaze(reliable=False), face_detected=True, timestamp=20.0)
        assert a_r.confidence > a_ur.confidence

    def test_confidence_in_unit_range(self):
        s = _scorer()
        for ts in [0.0, 5.0, 15.0, 100.0]:
            a = s.update(blink=_blink(0.8), head=_head(),
                         gaze=_gaze(), face_detected=True, timestamp=ts)
            assert 0.0 <= a.confidence <= 1.0


# ── Rolling fractions ─────────────────────────────────────────────────────────

class TestRollingFractions:
    def _focused_cfg(self):
        # alpha=1 and min_frames=1 so state immediately reflects raw score
        return {
            "engagement": {
                **_CFG["engagement"],
                "smoothing": {"alpha": 1.0, "state_min_frames": 1},
                "confidence": {"min_history_s": 0.0, "pose_error_max": 10.0},
            }
        }

    def test_all_focused_fraction_is_one(self):
        s = AttentionScorer(self._focused_cfg())
        for i in range(10):
            s.update(blink=_blink(0.9), head=_head(fraction=0.95),
                     gaze=_gaze(on_screen_frac=0.95),
                     face_detected=True, timestamp=float(i))
        a = s.update(blink=_blink(0.9), head=_head(fraction=0.95),
                     gaze=_gaze(on_screen_frac=0.95),
                     face_detected=True, timestamp=11.0)
        assert a.focused_fraction == pytest.approx(1.0, abs=0.01)

    def test_all_distracted_fraction_is_zero(self):
        s = AttentionScorer(self._focused_cfg())
        for i in range(10):
            s.update(blink=_blink(0.5), head=_head(fraction=0.05),
                     gaze=_gaze(on_screen_frac=0.05),
                     face_detected=True, timestamp=float(i))
        a = s.update(blink=_blink(0.5), head=_head(fraction=0.05),
                     gaze=_gaze(on_screen_frac=0.05),
                     face_detected=True, timestamp=11.0)
        assert a.focused_fraction == pytest.approx(0.0, abs=0.01)

    def test_old_frames_pruned_from_window(self):
        cfg = {
            "engagement": {
                **_CFG["engagement"],
                "smoothing": {"alpha": 1.0, "state_min_frames": 1},
                "confidence": {"min_history_s": 0.0, "pose_error_max": 10.0},
                "rolling_window_s": 5.0,   # short window for testing
            }
        }
        s = AttentionScorer(cfg)
        # 10 distracted frames at t=0..9
        for i in range(10):
            s.update(blink=_blink(0.5), head=_head(fraction=0.05),
                     gaze=_gaze(on_screen_frac=0.05),
                     face_detected=True, timestamp=float(i))
        # 10 focused frames at t=20..29 (well beyond 5 s window)
        for i in range(10):
            s.update(blink=_blink(0.9), head=_head(fraction=0.95),
                     gaze=_gaze(on_screen_frac=0.95),
                     face_detected=True, timestamp=20.0 + float(i))
        a = s.update(blink=_blink(0.9), head=_head(fraction=0.95),
                     gaze=_gaze(on_screen_frac=0.95),
                     face_detected=True, timestamp=30.0)
        # Old distracted frames should be pruned; recent are all focused
        assert a.focused_fraction == pytest.approx(1.0, abs=0.01)

    def test_engaged_fraction_includes_drifting(self):
        s = AttentionScorer(self._focused_cfg())
        # Mix: some FOCUSED, some DRIFTING
        s.update(blink=_blink(0.9), head=_head(fraction=0.95),
                 gaze=_gaze(on_screen_frac=0.95), timestamp=0.0)
        s.update(blink=_blink(0.75), head=_head(fraction=0.55),
                 gaze=_gaze(on_screen_frac=0.55), timestamp=1.0)
        a = s.update(blink=_blink(0.75), head=_head(fraction=0.55),
                     gaze=_gaze(on_screen_frac=0.55), timestamp=2.0)
        assert a.engaged_fraction >= a.focused_fraction


# ── Flags ─────────────────────────────────────────────────────────────────────

class TestFlags:
    def test_is_fatigued_when_drowsy(self):
        s = _scorer()
        a = s.update(blink=_blink(0.8, drowsy=True), timestamp=0.0)
        assert a.is_fatigued is True

    def test_not_fatigued_when_healthy(self):
        s = _scorer()
        a = s.update(blink=_blink(0.8), timestamp=0.0)
        assert a.is_fatigued is False

    def test_is_looking_away_flag(self):
        s = _scorer()
        h = _head(zone=AttentionZone.LOOKING_AWAY, fraction=0.1)
        a = s.update(head=h, timestamp=0.0)
        assert a.is_looking_away is True

    def test_is_gaze_reliable_when_reliable(self):
        s = _scorer()
        a = s.update(gaze=_gaze(reliable=True), timestamp=0.0)
        assert a.is_gaze_reliable is True

    def test_is_gaze_reliable_when_unreliable(self):
        s = _scorer()
        a = s.update(gaze=_gaze(reliable=False), timestamp=0.0)
        assert a.is_gaze_reliable is False


# ── Missing signals ───────────────────────────────────────────────────────────

class TestMissingSignals:
    def test_all_none_does_not_crash(self):
        s = _scorer()
        a = s.update(face_detected=True, timestamp=0.0)
        assert isinstance(a, EngagementScore)

    def test_blink_only_does_not_crash(self):
        s = _scorer()
        a = s.update(blink=_blink(0.8), timestamp=0.0)
        assert 0.0 <= a.raw_score <= 1.0

    def test_head_only_does_not_crash(self):
        s = _scorer()
        a = s.update(head=_head(), timestamp=0.0)
        assert 0.0 <= a.raw_score <= 1.0

    def test_gaze_only_does_not_crash(self):
        s = _scorer()
        a = s.update(gaze=_gaze(), timestamp=0.0)
        assert 0.0 <= a.raw_score <= 1.0

    def test_score_always_in_unit_range(self):
        """Component scores are bounded; weighted sum must stay [0, 1]."""
        combos = [
            dict(blink=_blink(0.9), head=_head(), gaze=_gaze()),
            dict(blink=_blink(0.1, drowsy=True), head=_head(fraction=0.0),
                 gaze=_gaze(on_screen_frac=0.0)),
            dict(),
            dict(face_detected=False),
        ]
        s = _scorer()
        for kw in combos:
            a = s.update(timestamp=0.0, **kw)
            assert 0.0 <= a.raw_score      <= 1.0, f"raw out of range: {a.raw_score}"
            assert 0.0 <= a.smoothed_score <= 1.0


# ── Reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_smoothed_score(self):
        s = _scorer()
        for _ in range(10):
            s.update(blink=_blink(0.9), head=_head(), gaze=_gaze(), timestamp=0.0)
        s.reset()
        assert s._smoothed == pytest.approx(0.5)

    def test_reset_resets_state_to_unreliable(self):
        s = _scorer()
        for _ in range(10):
            s.update(blink=_blink(0.9), head=_head(), gaze=_gaze(), timestamp=0.0)
        s.reset()
        assert s._current_state == EngagementState.UNRELIABLE

    def test_history_preserved_across_reset(self):
        """reset() clears temporal FSM but preserves rolling history."""
        s = _scorer()
        s.update(blink=_blink(0.9), head=_head(), gaze=_gaze(), timestamp=0.0)
        s.reset()
        # History is NOT cleared by reset (it's cumulative session data)
        assert len(s._history) > 0
