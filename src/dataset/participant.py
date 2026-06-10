"""Participant registry: anonymous participant identity management.

Design principles
-----------------
Participants are identified by a short opaque ID (participant_001, a9f2b3…)
that is never derived from personal information.  The only optional personal
fields are:
  - age_range : coarse bracket ("18-24", "25-34") not exact age
  - notes     : free-text; caller's responsibility not to include PII

Why coarse age ranges
----------------------
Many behavioral patterns (blink rate, attention span, eye fatigue onset) have
meaningful age-related trends.  A coarse range lets us stratify analysis
without storing a birth date.

Calibration linkage
-------------------
Each participant has an optional path to a UserProfile (user_profile.json)
from Day 12.  When the dataset builder ingests a session for this participant,
it automatically forwards the calibration path to the session entry so that
the comparison layer can compute personalised z-scores.

Modality extensibility
-----------------------
The `extra_metadata` dict is a forward-compatibility hook for future
biosignal modalities (EEG cap type, ECG sampling rate).  It is stored in the
JSON and passed through transparently.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Participant:
    """Anonymous participant descriptor.

    Attributes
    ----------
    participant_id          : Unique opaque identifier, e.g. ``"participant_001"``.
    created_at              : ISO-8601 UTC when the entry was first created.
    age_range               : Coarse bracket, e.g. ``"25-34"``, or empty string.
    calibration_profile_path: Path to ``user_profile.json`` from Day 12, or ``""``.
    notes                   : Free-text annotation (device, environment, etc.).
    session_ids             : Session IDs ingested for this participant.
    extra_metadata          : Forward-compatibility hook for future modalities.
    """
    participant_id:           str
    created_at:               str
    age_range:                str = ""
    calibration_profile_path: str = ""
    notes:                    str = ""
    session_ids:              List[str]       = field(default_factory=list)
    extra_metadata:           Dict[str, Any]  = field(default_factory=dict)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        participant_id:           str = "",
        age_range:                str = "",
        calibration_profile_path: str = "",
        notes:                    str = "",
    ) -> "Participant":
        """Create a new participant with a generated ID if none provided."""
        pid = participant_id.strip() or f"participant_{secrets.token_hex(4)}"
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            participant_id=           pid,
            created_at=               now,
            age_range=                age_range,
            calibration_profile_path= calibration_profile_path,
            notes=                    notes,
        )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "participant_id":           self.participant_id,
            "created_at":               self.created_at,
            "age_range":                self.age_range,
            "calibration_profile_path": self.calibration_profile_path,
            "notes":                    self.notes,
            "session_ids":              list(self.session_ids),
            "extra_metadata":           dict(self.extra_metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Participant":
        return cls(
            participant_id=           d["participant_id"],
            created_at=               d.get("created_at", ""),
            age_range=                d.get("age_range", ""),
            calibration_profile_path= d.get("calibration_profile_path", ""),
            notes=                    d.get("notes", ""),
            session_ids=              list(d.get("session_ids", [])),
            extra_metadata=           dict(d.get("extra_metadata", {})),
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Participant":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def try_load(cls, path: Optional[str]) -> Optional["Participant"]:
        if not path:
            return None
        try:
            return cls.load(path)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None
