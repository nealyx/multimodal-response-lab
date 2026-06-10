"""DatasetRegistry: the main interface for managing the multi-session dataset.

The registry wraps a DatasetManifest and provides high-level operations:
  - register_participant   : add a new participant record
  - ingest_session         : discover session artefacts and add to the catalog
  - rebuild                : re-scan all known session directories

All mutating operations save the manifest to disk immediately so that the
on-disk state stays consistent with in-memory state.  There is no explicit
transaction model — for a dataset of hundreds of sessions, this is fast enough
that an atomic save per operation is acceptable.

The registry owns the manifest path; callers never interact with the JSON file
directly.  This is intentional: the manifest format can change without
breaking callers if they only use the registry API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from src.dataset.catalog import SessionEntry
from src.dataset.ingestor import SessionIngestor
from src.dataset.manifest import DatasetManifest
from src.dataset.participant import Participant

log = logging.getLogger(__name__)

DEFAULT_MANIFEST_NAME = "dataset_manifest.json"
_PARTICIPANTS_DIR     = "participants"


class DatasetRegistry:
    """High-level dataset management interface.

    Parameters
    ----------
    dataset_root : str
        Root directory for the dataset.  The manifest is stored here as
        ``dataset_manifest.json``.  The directory is created if it does not
        exist.
    """

    def __init__(self, dataset_root: str) -> None:
        self._root     = Path(dataset_root)
        self._manifest_path = self._root / DEFAULT_MANIFEST_NAME
        self._manifest = self._load_or_create()

    # ── Participant management ─────────────────────────────────────────────────

    def register_participant(
        self,
        participant_id:           str = "",
        age_range:                str = "",
        calibration_profile_path: str = "",
        notes:                    str = "",
    ) -> Participant:
        """Register a new participant and return the Participant object.

        If *participant_id* is empty a random hex ID is generated.
        If the participant already exists, the existing record is returned
        without modification.
        """
        p = Participant.create(
            participant_id=           participant_id,
            age_range=                age_range,
            calibration_profile_path= calibration_profile_path,
            notes=                    notes,
        )

        if p.participant_id in self._manifest.participants:
            log.info("Participant '%s' already registered; returning existing.", p.participant_id)
            return self._manifest.participants[p.participant_id]

        self._manifest.add_participant(p)
        self._write_participant_dir(p)
        self._save()
        log.info("Registered participant '%s'", p.participant_id)
        return p

    def get_participant(self, participant_id: str) -> Optional[Participant]:
        return self._manifest.get_participant(participant_id)

    def list_participants(self) -> List[Participant]:
        return list(self._manifest.participants.values())

    # ── Session ingestion ──────────────────────────────────────────────────────

    def ingest_session(
        self,
        session_dir:      str,
        participant_id:   str,
        embeddings_path:  Optional[str] = None,
        labels_path:      Optional[str] = None,
        calibration_path: Optional[str] = None,
    ) -> SessionEntry:
        """Ingest one session directory and add it to the catalog.

        If the participant has a calibration profile and *calibration_path*
        is not explicitly given, the participant's profile is used.

        Re-ingesting an already-catalogued session_id overwrites the entry
        (useful when artefacts were added after first ingestion).
        """
        # Inherit calibration from participant if not explicitly provided
        if calibration_path is None:
            p = self._manifest.get_participant(participant_id)
            if p and p.calibration_profile_path:
                calibration_path = p.calibration_profile_path

        if participant_id not in self._manifest.participants:
            log.warning(
                "Participant '%s' not registered; ingesting session anyway.",
                participant_id,
            )

        entry = SessionIngestor.ingest(
            session_dir=      session_dir,
            participant_id=   participant_id,
            embeddings_path=  embeddings_path,
            labels_path=      labels_path,
            calibration_path= calibration_path,
        )

        self._manifest.add_session(entry)

        # Link session_id back to participant
        p = self._manifest.get_participant(participant_id)
        if p and entry.session_id not in p.session_ids:
            p.session_ids.append(entry.session_id)

        self._write_session_dir(entry)
        self._save()

        n_issues = len(entry.quality_issues)
        log.info(
            "Ingested session '%s' for participant '%s'  "
            "(%d frames, %d windows, %d quality issue(s))",
            entry.session_id, participant_id,
            entry.n_frames, entry.n_windows, n_issues,
        )
        return entry

    def rebuild(self) -> int:
        """Re-ingest all catalogued sessions from their stored paths.

        Useful after adding artefacts (labels, embeddings) for existing sessions.
        Returns the number of sessions re-ingested.
        """
        entries = self._manifest.all_sessions()
        count = 0
        for e in entries:
            if not e.session_dir:
                continue
            self.ingest_session(
                session_dir=      e.session_dir,
                participant_id=   e.participant_id,
                embeddings_path=  e.embeddings_path or None,
                labels_path=      e.labels_path     or None,
                calibration_path= e.calibration_path or None,
            )
            count += 1
        return count

    # ── Query ──────────────────────────────────────────────────────────────────

    def all_sessions(self) -> List[SessionEntry]:
        return self._manifest.all_sessions()

    def sessions_for(self, participant_id: str) -> List[SessionEntry]:
        return self._manifest.sessions_for(participant_id)

    def get_session(self, session_id: str) -> Optional[SessionEntry]:
        return self._manifest.get_session(session_id)

    @property
    def manifest(self) -> DatasetManifest:
        return self._manifest

    @property
    def manifest_path(self) -> str:
        return str(self._manifest_path)

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_or_create(self) -> DatasetManifest:
        if self._manifest_path.exists():
            m = DatasetManifest.try_load(str(self._manifest_path))
            if m is not None:
                log.info("Loaded manifest from %s  (%d sessions, %d participants)",
                         self._manifest_path,
                         len(m.all_sessions()),
                         len(m.participants))
                return m
            log.warning("Manifest at %s could not be loaded; creating fresh.", self._manifest_path)

        self._root.mkdir(parents=True, exist_ok=True)
        m = DatasetManifest.create(str(self._root))
        m.save(str(self._manifest_path))
        log.info("Created new dataset manifest at %s", self._manifest_path)
        return m

    def _participant_dir(self, participant_id: str) -> Path:
        return self._root / _PARTICIPANTS_DIR / participant_id

    def _write_participant_dir(self, p: Participant) -> None:
        """Create participants/<pid>/ hierarchy and write profile.json."""
        p_dir = self._participant_dir(p.participant_id)
        (p_dir / "calibration").mkdir(parents=True, exist_ok=True)
        (p_dir / "sessions").mkdir(parents=True, exist_ok=True)
        profile_path = p_dir / "profile.json"
        with open(profile_path, "w", encoding="utf-8") as fh:
            json.dump(p.to_dict(), fh, indent=2)
        log.debug("Wrote participant profile to %s", profile_path)

    def _write_session_dir(self, entry: SessionEntry) -> None:
        """Write participants/<pid>/sessions/<sid>/session_entry.json."""
        if not entry.participant_id:
            return
        s_dir = (
            self._participant_dir(entry.participant_id)
            / "sessions"
            / entry.session_id
        )
        s_dir.mkdir(parents=True, exist_ok=True)
        entry_path = s_dir / "session_entry.json"
        with open(entry_path, "w", encoding="utf-8") as fh:
            json.dump(entry.to_dict(), fh, indent=2)
        log.debug("Wrote session entry to %s", entry_path)

    def _save(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._manifest.save(str(self._manifest_path))
