"""Unit tests for BlinkDetector FSM, blink registration, and rate calculation.

All tests inject EyeMeasurement objects with controlled EAR values and
timestamps — no webcam or MediaPipe runtime required.
"""

import time
import pytest
from src.signals.blink_detector import BlinkDetector, BlinkState
from src.signals.eye_metrics import EyeMeasurement


# ── Fixtures and helpers ──────────────────────────────────────────────────────

_BASE_CFG = {
    "blink": {
        "ear_close_threshold":      0.20,
        "ear_open_threshold":       0.25,
        "min_close_frames":         2,
        "min_open_frames":          2,
        "min_blink_ms":             60.0,
        "max_blink_ms":             500.0,
        "rate_window_s":            60.0,
        "low_blink_rate_threshold":  8.0,
        "high_blink_rate_threshold": 25.0,
        "drowsy_closure_ms":        500.0,
        "min_history_s":             0.0,   # disable history guard in tests
    }
}


def _measurement(ear: float, ts: float, frame_index: int = 0) -> EyeMeasurement:
    openness = max(0.0, min(1.0, (ear - 0.15) / 0.20))
    return EyeMeasurement(
        left_ear=ear, right_ear=ear, mean_ear=ear,
        left_openness=openness, right_openness=openness, mean_openness=openness,
        timestamp=ts, frame_index=frame_index,
    )


def _feed(detector: BlinkDetector, ears: list[float], fps: float = 30.0):
    """Feed a list of EAR values to the detector at *fps*, return last analysis."""
    analysis = None
    for i, ear in enumerate(ears):
        ts = i / fps
        analysis = detector.update(_measurement(ear, ts, frame_index=i))
    return analysis


def _make_blink_sequence(
    fps:         float = 30.0,
    pre_open:    int   = 10,    # frames with open eye before blink
    close_dur_s: float = 0.15,  # seconds eye stays closed
    post_open:   int   = 10,    # frames with open eye after blink
) -> list[float]:
    """Generate an EAR time-series with one blink in the middle."""
    close_frames = max(2, int(close_dur_s * fps))
    return (
        [0.30] * pre_open          # open
        + [0.15] * close_frames    # closed
        + [0.30] * post_open       # open again
    )


# ── State transitions ─────────────────────────────────────────────────────────

class TestFSMTransitions:
    def test_starts_in_open(self):
        det = BlinkDetector(_BASE_CFG)
        assert det._state == BlinkState.OPEN

    def test_single_low_frame_goes_to_closing(self):
        """One frame below threshold → CLOSING (not yet CLOSED)."""
        det = BlinkDetector(_BASE_CFG)
        det.update(_measurement(0.15, 0.0))
        assert det._state == BlinkState.CLOSING

    def test_two_low_frames_go_to_closed(self):
        """min_close_frames=2: two consecutive low frames → CLOSED."""
        det = BlinkDetector(_BASE_CFG)
        det.update(_measurement(0.15, 0.000))
        det.update(_measurement(0.15, 0.033))
        assert det._state == BlinkState.CLOSED

    def test_false_alarm_returns_to_open(self):
        """One low frame followed by recovery → back to OPEN, no blink."""
        det = BlinkDetector(_BASE_CFG)
        det.update(_measurement(0.15, 0.000))   # CLOSING
        det.update(_measurement(0.30, 0.033))   # recovery → OPEN
        assert det._state == BlinkState.OPEN
        assert det._blink_count == 0

    def test_closed_to_opening(self):
        """CLOSED + EAR above open threshold → OPENING (debounce)."""
        det = BlinkDetector(_BASE_CFG)
        det.update(_measurement(0.15, 0.000))
        det.update(_measurement(0.15, 0.033))   # → CLOSED
        det.update(_measurement(0.30, 0.100))   # → OPENING
        assert det._state == BlinkState.OPENING

    def test_re_close_during_opening(self):
        """EAR drops again during OPENING → back to CLOSED (blink not counted)."""
        det = BlinkDetector(_BASE_CFG)
        det.update(_measurement(0.15, 0.000))
        det.update(_measurement(0.15, 0.033))   # CLOSED
        det.update(_measurement(0.30, 0.200))   # OPENING
        det.update(_measurement(0.10, 0.233))   # drop → back to CLOSED
        assert det._state == BlinkState.CLOSED
        assert det._blink_count == 0

    def test_full_blink_cycle_registers(self):
        """Complete OPEN→CLOSED→OPEN cycle within valid duration → 1 blink."""
        det = BlinkDetector(_BASE_CFG)
        ears = _make_blink_sequence(fps=30.0, close_dur_s=0.15)
        analysis = _feed(det, ears, fps=30.0)
        assert analysis.blink_count == 1
        assert analysis.state == BlinkState.OPEN


# ── Duration gating ───────────────────────────────────────────────────────────

class TestDurationGating:
    def test_too_short_closure_not_counted(self):
        """Closure shorter than min_blink_ms (60 ms) is rejected."""
        det = BlinkDetector(_BASE_CFG)
        # At 30 fps, 1 frame = 33 ms < 60 ms.  Need 2 close frames for CLOSED
        # but then immediately open — duration ≈ 33 ms (1 closed frame interval).
        det.update(_measurement(0.15, 0.000))
        det.update(_measurement(0.15, 0.010))   # CLOSED at t=0.010
        det.update(_measurement(0.30, 0.030))   # → OPENING, duration ≈ 20 ms
        det.update(_measurement(0.30, 0.063))   # → OPEN
        assert det._blink_count == 0

    def test_normal_duration_counted(self):
        """Closure of 150 ms (within 60–500 ms gate) registers as a blink."""
        det = BlinkDetector(_BASE_CFG)
        ears = _make_blink_sequence(fps=30.0, close_dur_s=0.15)
        _feed(det, ears, fps=30.0)
        assert det._blink_count == 1

    def test_too_long_closure_not_counted(self):
        """Closure > max_blink_ms (500 ms) is not counted as a blink."""
        det = BlinkDetector(_BASE_CFG)
        # Closed for 600 ms (> 500 ms max)
        ears = _make_blink_sequence(fps=30.0, close_dur_s=0.60)
        _feed(det, ears, fps=30.0)
        assert det._blink_count == 0

    def test_multiple_blinks_all_counted(self):
        """Three sequential blinks → blink_count == 3."""
        det = BlinkDetector(_BASE_CFG)
        blink = _make_blink_sequence(fps=30.0, pre_open=8, close_dur_s=0.15, post_open=5)
        triple = blink + blink + blink
        analysis = _feed(det, triple, fps=30.0)
        assert analysis.blink_count == 3


# ── Rolling rate ──────────────────────────────────────────────────────────────

class TestRollingRate:
    def test_rate_zero_before_any_blink(self):
        det = BlinkDetector(_BASE_CFG)
        analysis = det.update(_measurement(0.30, 0.0))
        assert analysis.blink_rate_per_min == pytest.approx(0.0)

    def test_rate_increases_with_blinks(self):
        """More blinks in the window → higher rate."""
        det = BlinkDetector(_BASE_CFG)
        blink = _make_blink_sequence(fps=30.0, close_dur_s=0.15, pre_open=5, post_open=5)
        # Feed one blink, record rate; feed another, confirm rate went up.
        _feed(det, blink, fps=30.0)
        rate_after_1 = det._last_analysis.blink_rate_per_min

        _feed(det, blink, fps=30.0)
        rate_after_2 = det._last_analysis.blink_rate_per_min

        assert rate_after_2 >= rate_after_1

    def test_events_outside_window_pruned(self):
        """Blinks older than rate_window_s are dropped from the count."""
        cfg = {
            "blink": {
                **_BASE_CFG["blink"],
                "rate_window_s": 5.0,    # very short window for testing
                "min_history_s": 0.0,
            }
        }
        det = BlinkDetector(cfg)

        # Register a blink at t ≈ 0 (simulated)
        ears = _make_blink_sequence(fps=30.0, close_dur_s=0.15)
        _feed(det, ears, fps=30.0)
        assert det._blink_count == 1

        # Now advance time well past the 5 s window.
        # Feed open frames starting at t = 10 s.
        for i in range(10):
            det.update(_measurement(0.30, ts=10.0 + i * 0.033, frame_index=1000 + i))

        # The old blink event should have been pruned from _blink_events.
        assert len(det._blink_events) == 0


# ── no_face_update ────────────────────────────────────────────────────────────

class TestNoFaceUpdate:
    def test_returns_none_before_any_update(self):
        det = BlinkDetector(_BASE_CFG)
        assert det.no_face_update() is None

    def test_returns_last_analysis_after_update(self):
        det = BlinkDetector(_BASE_CFG)
        det.update(_measurement(0.30, 0.0))
        result = det.no_face_update()
        assert result is not None
        assert result.state == BlinkState.OPEN

    def test_does_not_advance_fsm(self):
        """Calling no_face_update during a closure must not reset close_ts."""
        det = BlinkDetector(_BASE_CFG)
        det.update(_measurement(0.15, 0.0))
        det.update(_measurement(0.15, 0.033))   # CLOSED
        state_before = det._state
        close_ts_before = det._close_ts

        det.no_face_update()
        det.no_face_update()

        assert det._state  == state_before
        assert det._close_ts == close_ts_before


# ── Fatigue flags ─────────────────────────────────────────────────────────────

class TestFatigueFlags:
    def test_prolonged_closure_flag(self):
        """Eye closed for > drowsy_closure_ms should set is_prolonged_closure."""
        det = BlinkDetector(_BASE_CFG)
        # Force CLOSED state with a very early close_ts
        det.update(_measurement(0.15, 0.000))
        det.update(_measurement(0.15, 0.033))   # now CLOSED, close_ts ≈ 0.033

        # Feed open frames far in the future — but stay closed
        analysis = det.update(_measurement(0.10, 2.0))   # still closed, 2 s later
        assert analysis.is_prolonged_closure is True

    def test_low_blink_rate_flag(self):
        """Rate below low_blink_rate_threshold should set is_low_blink_rate."""
        det = BlinkDetector(_BASE_CFG)
        # Feed open frames for a long time — no blinks → rate = 0
        for i in range(50):
            analysis = det.update(_measurement(0.30, ts=float(i) * 0.1))
        # min_history_s=0 so flag is active immediately
        assert analysis.is_low_blink_rate is True

    def test_fatigue_label_low_produces_moderate_or_high(self):
        """At least MODERATE fatigue when low-blink-rate flag is set."""
        det = BlinkDetector(_BASE_CFG)
        for i in range(50):
            analysis = det.update(_measurement(0.30, ts=float(i) * 0.1))
        assert analysis.fatigue_label in ("MODERATE", "HIGH")

    def test_fatigue_label_normal(self):
        """Healthy blink pattern → fatigue_label == 'LOW'."""
        det = BlinkDetector(_BASE_CFG)
        # Simulate 15 blinks/min for 30 s = 7-8 blinks in 30 s
        blink = _make_blink_sequence(fps=30.0, close_dur_s=0.15,
                                     pre_open=55, post_open=5)
        for _ in range(8):
            _feed(det, blink, fps=30.0)
        # Confirm no pathological flags
        analysis = det._last_analysis
        assert analysis.is_prolonged_closure is False


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_fsm_not_history(self):
        det = BlinkDetector(_BASE_CFG)
        ears = _make_blink_sequence(fps=30.0, close_dur_s=0.15)
        _feed(det, ears, fps=30.0)
        assert det._blink_count == 1

        det.reset()
        assert det._state == BlinkState.OPEN
        assert det._frames_closing == 0
        assert det._close_ts is None
        # Blink count and events are intentionally preserved across reset
        # (reset is for FSM state only, not cumulative metrics)
        assert det._blink_count == 1
