"""User-specific baseline profile: personal signal norms established during calibration.

What each baseline measures
----------------------------
blink : BlinkBaseline
    Personal resting blink rate and open-eye EAR.  Population average is ~15/min
    and EAR ~0.28, but individuals deviate by ±40%.  A person with baseline rate
    of 8/min is not abnormally low — they are just a slow blinker.  Flagging them
    at the population threshold of 8/min would produce constant false alarms.

head : HeadPoseBaseline
    Personal neutral head orientation and typical rotational range during rest.
    Subtle head tilts (glasses asymmetry, posture habits) cause systematic yaw/pitch
    offsets that shift the gaze-center estimate and trigger spurious "looking away."

gaze : GazeBaseline
    Personal center of gaze when looking at the screen.  The iris centering
    formula assumes gaze_h ≈ 0.5 = looking at screen center, but people with
    offset camera or wide-set eyes have systematic horizontal offsets.

engagement : EngagementBaseline
    Personal resting composite score during relaxed screen viewing.  Some people
    score 0.65 when focused (lower baseline EAR, mild habitual head tilt);
    others score 0.85 for the same behavioral quality.  Deviation from personal
    mean is more meaningful than absolute value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class BlinkBaseline:
    rate_per_min:  float   # personal resting blink rate
    rate_std:      float   # 1-sigma variability
    open_ear:      float   # mean EAR when eyes fully open
    open_ear_std:  float   # variability in open-eye EAR


@dataclass
class HeadPoseBaseline:
    yaw_center:   float   # neutral yaw (degrees; typically near 0)
    pitch_center: float   # neutral pitch (degrees; often slightly negative)
    yaw_range:    float   # 1-sigma yaw spread during rest
    pitch_range:  float   # 1-sigma pitch spread during rest


@dataclass
class GazeBaseline:
    h_center: float   # mean horizontal iris ratio during screen viewing [0,1]
    v_center: float   # mean vertical iris ratio during screen viewing [0,1]
    h_range:  float   # 1-sigma horizontal spread
    v_range:  float   # 1-sigma vertical spread


@dataclass
class EngagementBaseline:
    score_mean: float   # mean composite engagement during calibration
    score_std:  float   # variability


@dataclass
class UserProfile:
    """Personal behavioral baseline established during a calibration session.

    Attributes
    ----------
    profile_id : str
        Unique identifier.  Defaults to a short random hex string.
    created_at : str
        ISO-8601 UTC timestamp when the profile was saved.
    n_frames : int
        Number of frames collected during calibration.
    duration_s : float
        Calibration session length in seconds.
    blink : BlinkBaseline
    head : HeadPoseBaseline
    gaze : GazeBaseline
    engagement : EngagementBaseline
    notes : str
        Optional free-text note (device description, lighting conditions, etc.).
    """
    profile_id:  str
    created_at:  str
    n_frames:    int
    duration_s:  float
    blink:       BlinkBaseline
    head:        HeadPoseBaseline
    gaze:        GazeBaseline
    engagement:  EngagementBaseline
    notes:       str = ""

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "created_at": self.created_at,
            "n_frames":   self.n_frames,
            "duration_s": self.duration_s,
            "notes":      self.notes,
            "blink": asdict(self.blink),
            "head":  asdict(self.head),
            "gaze":  asdict(self.gaze),
            "engagement": asdict(self.engagement),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        return cls(
            profile_id=  d["profile_id"],
            created_at=  d["created_at"],
            n_frames=    int(d.get("n_frames", 0)),
            duration_s=  float(d.get("duration_s", 0.0)),
            notes=       d.get("notes", ""),
            blink=       BlinkBaseline(**d["blink"]),
            head=        HeadPoseBaseline(**d["head"]),
            gaze=        GazeBaseline(**d["gaze"]),
            engagement=  EngagementBaseline(**d["engagement"]),
        )

    def save(self, path: str) -> None:
        """Write profile to *path* as JSON (creates parent directories)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "UserProfile":
        """Load a profile from *path*.  Raises FileNotFoundError if missing."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def try_load(cls, path: Optional[str]) -> Optional["UserProfile"]:
        """Load a profile if *path* is provided and exists, else return None."""
        if path is None:
            return None
        try:
            return cls.load(path)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def personalized_low_blink_threshold(self) -> float:
        """Blink rate below this → low-rate flag for THIS person (2 sigma below mean)."""
        return max(0.0, self.blink.rate_per_min - 2.0 * max(self.blink.rate_std, 1.0))

    @property
    def personalized_close_ear(self) -> float:
        """EAR close threshold personalised to this person (mean - 2.5 sigma)."""
        return max(0.10, self.blink.open_ear - 2.5 * max(self.blink.open_ear_std, 0.01))
