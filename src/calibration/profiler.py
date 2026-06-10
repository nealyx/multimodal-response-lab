"""Compute a UserProfile from raw CalibrationSamples.

Statistical choices
--------------------
mean ± std is used rather than percentile-based ranges because:
  1. We have a small sample (300–600 frames in a 10–20 s calibration).
  2. The signals are approximately Gaussian during quiet rest.
  3. The downstream use is z-score computation, which expects mean and std.

The main edge case is blink rate: since the rolling rate starts at 0 and ramps
up during the calibration warm-up, we exclude the first `_RATE_WARMUP_S` seconds
from the blink-rate average to avoid downward bias.

Minimum sample requirements
-----------------------------
With fewer than MIN_VALID_FRAMES valid face frames, the profiler emits a warning
but still returns a profile — populated with safe population-average defaults for
any signal that had insufficient data.  This is better than crashing so the rest
of the pipeline can degrade gracefully.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import List

import numpy as np

from src.calibration.baseline import (
    BlinkBaseline,
    EngagementBaseline,
    GazeBaseline,
    HeadPoseBaseline,
    UserProfile,
)
from src.calibration.collector import CalibrationCollector, CalibrationSample

log = logging.getLogger(__name__)

MIN_VALID_FRAMES = 30   # fewer → warn, use defaults for that signal
_RATE_WARMUP_S   = 5.0  # exclude early frames from blink-rate mean


# Population-average defaults used when calibration data is insufficient
_DEFAULTS = {
    "blink_rate":  15.0,
    "blink_std":    4.0,
    "open_ear":     0.28,
    "open_ear_std": 0.02,
    "head_yaw":     0.0,
    "head_pitch":  -2.0,
    "head_range":   5.0,
    "gaze_h":       0.50,
    "gaze_v":       0.50,
    "gaze_range":   0.10,
    "eng_score":    0.70,
    "eng_std":      0.08,
}


def _mean_std(values: List[float]) -> tuple:
    """Return (mean, std) for *values*, or (0.0, 1.0) if empty."""
    if not values:
        return 0.0, 1.0
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0


class CalibrationProfiler:
    """Compute a UserProfile from a CalibrationCollector's samples."""

    @classmethod
    def compute(
        cls,
        collector:  CalibrationCollector,
        profile_id: str = "",
        notes:      str = "",
    ) -> UserProfile:
        """Compute and return a UserProfile from collected samples.

        Parameters
        ----------
        collector  : CalibrationCollector with ≥ MIN_VALID_FRAMES samples
        profile_id : optional user-supplied ID; a random hex is used if empty
        notes      : optional free-text note stored in the profile
        """
        pid = profile_id.strip() or secrets.token_hex(6)
        now = datetime.now(timezone.utc).isoformat()

        valid   = collector.valid_samples()
        n_valid = len(valid)

        if n_valid < MIN_VALID_FRAMES:
            log.warning(
                "Only %d valid frames for calibration (minimum %d). "
                "Population-average defaults will be used for insufficient signals.",
                n_valid, MIN_VALID_FRAMES,
            )

        # ── Blink baseline ────────────────────────────────────────────────────
        blink = cls._blink_baseline(valid, collector.duration_s)

        # ── Open-eye EAR baseline ─────────────────────────────────────────────
        open_eye = collector.open_eye_samples()
        if len(open_eye) >= 10:
            ears = [s.ear for s in open_eye]
            ear_mean, ear_std = _mean_std(ears)
        else:
            log.warning("Insufficient open-eye samples; using population EAR defaults.")
            ear_mean = _DEFAULTS["open_ear"]
            ear_std  = _DEFAULTS["open_ear_std"]
        blink.open_ear     = round(ear_mean, 4)
        blink.open_ear_std = round(ear_std,  4)

        # ── Head pose baseline ────────────────────────────────────────────────
        head = cls._head_baseline(valid)

        # ── Gaze baseline ─────────────────────────────────────────────────────
        gaze = cls._gaze_baseline(valid)

        # ── Engagement baseline ───────────────────────────────────────────────
        eng = cls._engagement_baseline(valid)

        return UserProfile(
            profile_id= pid,
            created_at= now,
            n_frames=   collector.n_samples,
            duration_s= round(collector.duration_s, 2),
            blink=      blink,
            head=       head,
            gaze=       gaze,
            engagement= eng,
            notes=      notes,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @classmethod
    def _blink_baseline(
        cls,
        valid:      List[CalibrationSample],
        duration_s: float,
    ) -> BlinkBaseline:
        # Skip the first _RATE_WARMUP_S seconds so the rolling rate window
        # has had time to fill before we sample it.
        if valid:
            t0 = valid[0].timestamp
            rates = [
                s.blink_rate for s in valid
                if (s.timestamp - t0) >= _RATE_WARMUP_S
            ]
        else:
            rates = []

        if len(rates) >= 10:
            rate_mean, rate_std = _mean_std(rates)
        else:
            log.warning("Insufficient post-warmup frames for blink rate; using defaults.")
            rate_mean = _DEFAULTS["blink_rate"]
            rate_std  = _DEFAULTS["blink_std"]

        return BlinkBaseline(
            rate_per_min= round(rate_mean, 2),
            rate_std=     round(max(rate_std, 0.5), 2),
            open_ear=     _DEFAULTS["open_ear"],      # filled in by caller
            open_ear_std= _DEFAULTS["open_ear_std"],
        )

    @classmethod
    def _head_baseline(cls, valid: List[CalibrationSample]) -> HeadPoseBaseline:
        if len(valid) >= 10:
            yaws   = [s.head_yaw   for s in valid]
            pitches = [s.head_pitch for s in valid]
            yaw_mean,   yaw_std   = _mean_std(yaws)
            pitch_mean, pitch_std = _mean_std(pitches)
        else:
            yaw_mean,   yaw_std   = _DEFAULTS["head_yaw"],  _DEFAULTS["head_range"]
            pitch_mean, pitch_std = _DEFAULTS["head_pitch"], _DEFAULTS["head_range"]

        return HeadPoseBaseline(
            yaw_center=  round(yaw_mean,   2),
            pitch_center=round(pitch_mean, 2),
            yaw_range=   round(max(yaw_std,   0.5), 2),
            pitch_range= round(max(pitch_std, 0.5), 2),
        )

    @classmethod
    def _gaze_baseline(cls, valid: List[CalibrationSample]) -> GazeBaseline:
        if len(valid) >= 10:
            hs = [s.gaze_h for s in valid]
            vs = [s.gaze_v for s in valid]
            h_mean, h_std = _mean_std(hs)
            v_mean, v_std = _mean_std(vs)
        else:
            h_mean, h_std = _DEFAULTS["gaze_h"], _DEFAULTS["gaze_range"]
            v_mean, v_std = _DEFAULTS["gaze_v"], _DEFAULTS["gaze_range"]

        return GazeBaseline(
            h_center=round(h_mean, 4),
            v_center=round(v_mean, 4),
            h_range= round(max(h_std, 0.01), 4),
            v_range= round(max(v_std, 0.01), 4),
        )

    @classmethod
    def _engagement_baseline(cls, valid: List[CalibrationSample]) -> EngagementBaseline:
        if len(valid) >= 10:
            scores = [s.engagement_score for s in valid]
            s_mean, s_std = _mean_std(scores)
        else:
            s_mean = _DEFAULTS["eng_score"]
            s_std  = _DEFAULTS["eng_std"]

        return EngagementBaseline(
            score_mean=round(s_mean, 4),
            score_std= round(max(s_std, 0.01), 4),
        )
