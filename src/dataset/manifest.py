"""Dataset manifest: the single source of truth for the whole dataset.

The manifest (dataset_manifest.json) is the root index file.  It records:
  - The dataset schema version (for future migrations)
  - When the dataset was created and last updated
  - The dataset root directory
  - All registered participants
  - All ingested session entries

Design choices
--------------
Flat JSON rather than a relational DB
    A behavioral experiment dataset with 100 sessions and 20 participants
    fits comfortably in a human-readable JSON file under 1 MB.  SQLite would
    add a library dependency, make diffs unreadable in git, and complicate
    portability.  JSON is plain text, diffable, and trivially editable in a
    text editor — important for a research dataset where provenance matters.

Single file vs distributed metadata
    Alternative: store per-session metadata next to the session files.
    Problem: if you move or rename the session directory, the central index
    becomes inconsistent.  The manifest records all paths as stored strings —
    when a path no longer exists, the quality checker flags it rather than
    silently dropping data.

Versioning
    The ``manifest_version`` field allows future migrations (add new fields,
    rename others) while preserving backward compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.dataset.catalog import SessionCatalog, SessionEntry
from src.dataset.participant import Participant

MANIFEST_VERSION = "1.0"


@dataclass
class DatasetManifest:
    """Root metadata container for the full multi-session dataset.

    All participant and session data lives here.  Save/load are the only
    methods that touch disk — mutations happen through DatasetRegistry.
    """
    manifest_version: str
    created_at:       str
    last_updated:     str
    dataset_root:     str
    participants:     Dict[str, Participant] = field(default_factory=dict)
    catalog:          SessionCatalog        = field(default_factory=SessionCatalog)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, dataset_root: str) -> "DatasetManifest":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            manifest_version= MANIFEST_VERSION,
            created_at=       now,
            last_updated=     now,
            dataset_root=     str(Path(dataset_root).resolve()),
        )

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def touch(self) -> None:
        """Update last_updated timestamp."""
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def add_participant(self, p: Participant) -> None:
        self.participants[p.participant_id] = p
        self.touch()

    def add_session(self, entry: SessionEntry) -> None:
        self.catalog.add(entry)
        self.touch()

    def get_participant(self, participant_id: str) -> Optional[Participant]:
        return self.participants.get(participant_id)

    def get_session(self, session_id: str) -> Optional[SessionEntry]:
        return self.catalog.get(session_id)

    def all_sessions(self) -> List[SessionEntry]:
        return self.catalog.all_entries()

    def sessions_for(self, participant_id: str) -> List[SessionEntry]:
        return self.catalog.for_participant(participant_id)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "manifest_version": self.manifest_version,
            "created_at":       self.created_at,
            "last_updated":     self.last_updated,
            "dataset_root":     self.dataset_root,
            "participants":     {pid: p.to_dict() for pid, p in self.participants.items()},
            "sessions":         self.catalog.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetManifest":
        participants = {
            pid: Participant.from_dict(pd)
            for pid, pd in d.get("participants", {}).items()
        }
        catalog = SessionCatalog.from_dict(d.get("sessions", {}))
        return cls(
            manifest_version= d.get("manifest_version", MANIFEST_VERSION),
            created_at=       d.get("created_at", ""),
            last_updated=     d.get("last_updated", ""),
            dataset_root=     d.get("dataset_root", ""),
            participants=     participants,
            catalog=          catalog,
        )

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "DatasetManifest":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def try_load(cls, path: str) -> Optional["DatasetManifest"]:
        try:
            return cls.load(path)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None
