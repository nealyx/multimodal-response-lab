"""Compare a session's signals against a personal baseline profile.

Why z-scores rather than absolute deltas
------------------------------------------
A blink rate of 10/min means nothing without context:
  - Person A's baseline: 18/min → z = (10 - 18) / 4 = -2.0 → notable suppression
  - Person B's baseline:  9/min → z = (10 - 9)  / 2 =  0.5 → essentially normal

Absolute thresholds (e.g. "low if < 8/min") fail for both: they flag Person B
even though 10/min is their normal, and they let Person A pass even though their
rate has dropped significantly.  Z-scores solve both cases.

Interpreting z-scores
-----------------------
  |z| < 1.0  : within normal range for this person
  |z| 1–2    : mild deviation
  |z| 2–3    : notable deviation
  |z| > 3    : strong deviation — may warrant attention

Direction meaning per signal
------------------------------
  blink_rate      : z < 0 → suppressed (potential screen focus/early fatigue)
                    z > 0 → elevated (potential eye strain)
  engagement_score: z < 0 → below personal norm (possible disengagement)
                    z > 0 → above norm (heightened engagement)
  head_yaw/pitch  : z < 0 → tilted toward lower/left relative to calibration
                    z > 0 → tilted toward higher/right
  gaze_h/v        : z < 0 → shifted left/up of personal screen center
                    z > 0 → shifted right/down
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from src.calibration.baseline import UserProfile


@dataclass
class SignalDelta:
    """Deviation of a single signal from its personal baseline.

    Attributes
    ----------
    signal_name   : Human-readable signal identifier (e.g. "blink_rate")
    raw_value     : The observed session-aggregate value
    baseline_mean : Personal calibration mean
    baseline_std  : Personal calibration standard deviation
    z_score       : (raw - mean) / std; None if std == 0
    direction     : "above", "below", or "neutral" (|z| < 0.5)
    """
    signal_name:    str
    raw_value:      float
    baseline_mean:  float
    baseline_std:   float
    z_score:        Optional[float]
    direction:      str   # "above" | "below" | "neutral"

    @property
    def magnitude(self) -> str:
        """Narrative label for |z|: 'normal', 'mild', 'notable', 'strong'."""
        if self.z_score is None:
            return "unknown"
        az = abs(self.z_score)
        if az < 1.0:
            return "normal"
        if az < 2.0:
            return "mild"
        if az < 3.0:
            return "notable"
        return "strong"

    def to_dict(self) -> dict:
        return {
            "signal_name":   self.signal_name,
            "raw_value":     round(self.raw_value, 4),
            "baseline_mean": round(self.baseline_mean, 4),
            "baseline_std":  round(self.baseline_std, 4),
            "z_score":       round(self.z_score, 3) if self.z_score is not None else None,
            "direction":     self.direction,
            "magnitude":     self.magnitude,
        }


@dataclass
class SessionComparison:
    """Aggregated baseline comparison for one session.

    ``deltas`` maps signal_name → SignalDelta.
    ``profile_id`` records which calibration profile was used.
    ``session_dir`` identifies the session being compared.
    """
    profile_id:  str
    session_dir: str
    deltas:      Dict[str, SignalDelta]

    def summary_lines(self) -> list:
        """Return a list of printable summary strings, one per signal."""
        lines = []
        for name, d in sorted(self.deltas.items()):
            z = f"{d.z_score:+.2f}" if d.z_score is not None else "N/A"
            lines.append(
                f"  {name:<22}  raw={d.raw_value:>7.3f}  "
                f"baseline={d.baseline_mean:.3f}±{d.baseline_std:.3f}  "
                f"z={z:<7}  [{d.magnitude}]"
            )
        return lines

    def to_dict(self) -> dict:
        return {
            "profile_id":  self.profile_id,
            "session_dir": self.session_dir,
            "deltas":      {k: v.to_dict() for k, v in self.deltas.items()},
        }


class BaselineComparator:
    """Compute a SessionComparison from session aggregates and a UserProfile."""

    @classmethod
    def compare(
        cls,
        profile:     UserProfile,
        session_dir: str,
        *,
        blink_rate:      Optional[float] = None,
        engagement_score: Optional[float] = None,
        head_yaw:        Optional[float] = None,
        head_pitch:      Optional[float] = None,
        gaze_h:          Optional[float] = None,
        gaze_v:          Optional[float] = None,
    ) -> SessionComparison:
        """Compute z-score deltas between session aggregates and the profile.

        Pass None for any signal that is unavailable; it will be omitted from
        the comparison.
        """
        deltas: Dict[str, SignalDelta] = {}

        signal_map = {
            "blink_rate":       (blink_rate,       profile.blink.rate_per_min,  profile.blink.rate_std),
            "engagement_score": (engagement_score,  profile.engagement.score_mean, profile.engagement.score_std),
            "head_yaw":         (head_yaw,          profile.head.yaw_center,     profile.head.yaw_range),
            "head_pitch":       (head_pitch,        profile.head.pitch_center,   profile.head.pitch_range),
            "gaze_h":           (gaze_h,            profile.gaze.h_center,       profile.gaze.h_range),
            "gaze_v":           (gaze_v,            profile.gaze.v_center,       profile.gaze.v_range),
        }

        for name, (value, mean, std) in signal_map.items():
            if value is None:
                continue
            deltas[name] = cls._make_delta(name, value, mean, std)

        return SessionComparison(
            profile_id=  profile.profile_id,
            session_dir= session_dir,
            deltas=      deltas,
        )

    @staticmethod
    def _make_delta(
        name:  str,
        value: float,
        mean:  float,
        std:   float,
    ) -> SignalDelta:
        if std > 0.0:
            z: Optional[float] = (value - mean) / std
        else:
            z = None

        if z is None or abs(z) < 0.5:
            direction = "neutral"
        elif z > 0:
            direction = "above"
        else:
            direction = "below"

        return SignalDelta(
            signal_name=   name,
            raw_value=     value,
            baseline_mean= mean,
            baseline_std=  std,
            z_score=       z,
            direction=     direction,
        )
