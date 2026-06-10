"""Session catalog: per-session metadata and aggregate statistics.

SessionEntry is the central record for one ingested session.  It stores:
  - File presence flags (what artefacts exist)
  - Aggregate statistics (duration, frame count, window count)
  - Engagement distribution (computed once during ingestion)
  - Label summary (from labels.json if available)
  - Quality issues found during ingestion

The decision to cache aggregates in the entry (rather than recomputing from
raw files every time) is deliberate: reading a 9000-row samples.csv on every
stats query would be prohibitively slow once a dataset has 50+ sessions.
Trade-off: the cached stats can become stale if raw files are modified after
ingestion. The quality checker flags this if it can detect it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SessionEntry:
    """Metadata record for one ingested session.

    File presence
    -------------
    has_session_log  : session_log.json found
    has_samples      : samples.csv found (needed for engagement fractions)
    has_embeddings   : behavioral_windows.csv found (needed for ML)
    has_labels       : labels.json found (human-survey labels from Day 12)
    has_calibration  : user_profile.json associated

    Aggregates
    ----------
    n_frames         : rows in samples.csv (0 if unavailable)
    n_windows        : rows in behavioral_windows.csv (0 if unavailable)
    duration_s       : session length in seconds (from session_log)
    experiment_name  : from session_log.experiment_name

    engagement_fractions : {state: fraction} from samples.csv scan
    label_summary        : {task: "HIGH" | "LOW" | None} from labels.json

    Paths
    -----
    All paths stored as strings.  Empty string = not found / not set.

    Extensibility
    -------------
    extra_metadata : forward-compat hook for EEG / biosignal fields.
    """
    session_id:       str
    participant_id:   str
    ingested_at:      str          # ISO-8601 when this entry was created

    # File presence flags
    has_session_log:  bool = False
    has_samples:      bool = False
    has_embeddings:   bool = False
    has_labels:       bool = False
    has_calibration:  bool = False

    # Aggregate stats
    n_frames:         int   = 0
    n_windows:        int   = 0
    duration_s:       float = 0.0
    experiment_name:  str   = ""
    recorded_at:      str   = ""   # ISO-8601 from session_log start_ts

    # Signal aggregates (from samples.csv)
    engagement_fractions: Dict[str, float] = field(default_factory=dict)
    mean_engagement_score: float           = 0.0
    on_screen_fraction:    float           = 0.0   # fraction of frames is_on_screen=True
    mean_blink_rate:       float           = 0.0   # mean blink_rate_per_min over session

    # Label summary (from labels.json)
    label_summary: Dict[str, Optional[str]] = field(default_factory=dict)

    # File paths
    session_dir:       str = ""
    samples_path:      str = ""
    embeddings_path:   str = ""
    labels_path:       str = ""
    calibration_path:  str = ""

    # Quality issues detected during ingestion
    quality_issues: List[str] = field(default_factory=list)

    # Forward-compat: future modalities (EEG, ECG, etc.)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def is_complete(self) -> bool:
        """True when all expected artefacts are present."""
        return (
            self.has_session_log
            and self.has_samples
            and self.has_embeddings
        )

    @property
    def focused_fraction(self) -> float:
        return self.engagement_fractions.get("focused", 0.0)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "participant_id":   self.participant_id,
            "ingested_at":      self.ingested_at,
            "has_session_log":  self.has_session_log,
            "has_samples":      self.has_samples,
            "has_embeddings":   self.has_embeddings,
            "has_labels":       self.has_labels,
            "has_calibration":  self.has_calibration,
            "n_frames":         self.n_frames,
            "n_windows":        self.n_windows,
            "duration_s":       self.duration_s,
            "experiment_name":  self.experiment_name,
            "recorded_at":      self.recorded_at,
            "mean_engagement_score": self.mean_engagement_score,
            "on_screen_fraction":    self.on_screen_fraction,
            "mean_blink_rate":       self.mean_blink_rate,
            "engagement_fractions": dict(self.engagement_fractions),
            "label_summary":    {k: v for k, v in self.label_summary.items()},
            "session_dir":      self.session_dir,
            "samples_path":     self.samples_path,
            "embeddings_path":  self.embeddings_path,
            "labels_path":      self.labels_path,
            "calibration_path": self.calibration_path,
            "quality_issues":   list(self.quality_issues),
            "extra_metadata":   dict(self.extra_metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionEntry":
        return cls(
            session_id=       d["session_id"],
            participant_id=   d.get("participant_id", ""),
            ingested_at=      d.get("ingested_at", ""),
            has_session_log=  bool(d.get("has_session_log",  False)),
            has_samples=      bool(d.get("has_samples",      False)),
            has_embeddings=   bool(d.get("has_embeddings",   False)),
            has_labels=       bool(d.get("has_labels",       False)),
            has_calibration=  bool(d.get("has_calibration",  False)),
            n_frames=         int(d.get("n_frames",   0)),
            n_windows=        int(d.get("n_windows",  0)),
            duration_s=       float(d.get("duration_s", 0.0)),
            experiment_name=  d.get("experiment_name", ""),
            recorded_at=      d.get("recorded_at", ""),
            mean_engagement_score= float(d.get("mean_engagement_score", 0.0)),
            on_screen_fraction=    float(d.get("on_screen_fraction",    0.0)),
            mean_blink_rate=       float(d.get("mean_blink_rate",       0.0)),
            engagement_fractions= dict(d.get("engagement_fractions", {})),
            label_summary=    {k: v for k, v in d.get("label_summary", {}).items()},
            session_dir=      d.get("session_dir",      ""),
            samples_path=     d.get("samples_path",     ""),
            embeddings_path=  d.get("embeddings_path",  ""),
            labels_path=      d.get("labels_path",      ""),
            calibration_path= d.get("calibration_path", ""),
            quality_issues=   list(d.get("quality_issues", [])),
            extra_metadata=   dict(d.get("extra_metadata", {})),
        )


class SessionCatalog:
    """Ordered collection of SessionEntry records, keyed by session_id."""

    def __init__(self) -> None:
        self._entries: Dict[str, SessionEntry] = {}

    def add(self, entry: SessionEntry) -> None:
        self._entries[entry.session_id] = entry

    def get(self, session_id: str) -> Optional[SessionEntry]:
        return self._entries.get(session_id)

    def remove(self, session_id: str) -> bool:
        if session_id in self._entries:
            del self._entries[session_id]
            return True
        return False

    def all_entries(self) -> List[SessionEntry]:
        return list(self._entries.values())

    def for_participant(self, participant_id: str) -> List[SessionEntry]:
        return [e for e in self._entries.values() if e.participant_id == participant_id]

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> Dict[str, dict]:
        return {sid: e.to_dict() for sid, e in self._entries.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, dict]) -> "SessionCatalog":
        cat = cls()
        for sid, entry_dict in d.items():
            cat.add(SessionEntry.from_dict(entry_dict))
        return cat
