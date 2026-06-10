"""Session ingestor: discover and read session artefacts into a SessionEntry.

What the ingestor does
-----------------------
Given a session directory (and optional path overrides), the ingestor:
  1. Reads session_log.json for session ID, timestamps, and experiment name.
  2. Counts rows in samples.csv and computes engagement-state fractions.
  3. Counts rows in behavioral_windows.csv (from embed_session.py).
  4. Reads labels.json for the human label summary (from run_calibration.py).
  5. Links the calibration profile from the participant or a specified path.
  6. Records all file presence flags and populates quality_issues.

Standard paths searched automatically
---------------------------------------
  session_log  : <session_dir>/session_log.json
  samples      : <session_dir>/samples.csv
  labels       : <session_dir>/labels.json
  embeddings   : outputs/embeddings/<session_id>/behavioral_windows.csv
  analytics    : outputs/reports/<session_id>/session_metrics.json

All paths can be overridden by the caller.  When an override is given, that
path is recorded in the entry even if it does not exist (it may be added later).

Why count rows instead of loading full dataframes
-------------------------------------------------
The ingestor is called once per session at ingestion time.  Reading a
9000-row CSV to get aggregate stats (engagement fractions) is acceptable then.
The cached result is stored in the SessionEntry so subsequent stats queries
need not read any CSVs.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from src.dataset.catalog import SessionEntry

log = logging.getLogger(__name__)


class SessionIngestor:
    """Read session artefacts and build a SessionEntry.

    Usage
    -----
    entry = SessionIngestor.ingest(
        session_dir=    "outputs/sessions/20241201_120000/",
        participant_id= "participant_001",
        embeddings_path="outputs/embeddings/20241201_120000/behavioral_windows.csv",
        labels_path=    "outputs/sessions/20241201_120000/labels.json",
        calibration_path="outputs/calibration/user_profile.json",
    )
    """

    @classmethod
    def ingest(
        cls,
        session_dir:       str,
        participant_id:    str,
        embeddings_path:   Optional[str] = None,
        labels_path:       Optional[str] = None,
        calibration_path:  Optional[str] = None,
    ) -> SessionEntry:
        """Discover and read session artefacts; return a populated SessionEntry."""
        now = datetime.now(timezone.utc).isoformat()
        session_path = Path(session_dir)
        quality_issues: list = []

        # ── Session log ───────────────────────────────────────────────────────
        session_id, experiment_name, duration_s, recorded_at, has_log = \
            cls._read_session_log(session_path, quality_issues)

        # ── Samples.csv ───────────────────────────────────────────────────────
        samples_path_str = str(session_path / "samples.csv")
        has_samples, n_frames, eng_fracs, mean_eng = cls._read_samples(
            samples_path_str, quality_issues
        )

        # ── Embeddings ────────────────────────────────────────────────────────
        if embeddings_path is None:
            embeddings_path = cls._default_embeddings_path(session_id)
        has_embeddings, n_windows = cls._count_embeddings(embeddings_path, quality_issues)

        # ── Labels ────────────────────────────────────────────────────────────
        if labels_path is None:
            candidate = str(session_path / "labels.json")
            if Path(candidate).exists():
                labels_path = candidate

        has_labels, label_summary = cls._read_labels(labels_path, quality_issues)

        # ── Calibration ───────────────────────────────────────────────────────
        has_calibration = False
        if calibration_path and Path(calibration_path).exists():
            has_calibration = True
        elif calibration_path:
            quality_issues.append(
                f"calibration_profile not found: {calibration_path}"
            )

        return SessionEntry(
            session_id=       session_id,
            participant_id=   participant_id,
            ingested_at=      now,
            has_session_log=  has_log,
            has_samples=      has_samples,
            has_embeddings=   has_embeddings,
            has_labels=       has_labels,
            has_calibration=  has_calibration,
            n_frames=         n_frames,
            n_windows=        n_windows,
            duration_s=       duration_s,
            experiment_name=  experiment_name,
            recorded_at=      recorded_at,
            engagement_fractions= eng_fracs,
            mean_engagement_score= mean_eng,
            label_summary=    label_summary,
            session_dir=      str(session_path.resolve()),
            samples_path=     samples_path_str,
            embeddings_path=  str(embeddings_path) if embeddings_path else "",
            labels_path=      str(labels_path)     if labels_path     else "",
            calibration_path= str(calibration_path) if calibration_path else "",
            quality_issues=   quality_issues,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_embeddings_path(session_id: str) -> str:
        return f"outputs/embeddings/{session_id}/behavioral_windows.csv"

    @staticmethod
    def _read_session_log(
        session_path: Path,
        issues: list,
    ):
        """Return (session_id, experiment_name, duration_s, recorded_at, has_log)."""
        json_path = session_path / "session_log.json"
        if not json_path.exists():
            issues.append(f"MISSING session_log.json in {session_path}")
            folder = session_path.name
            return folder, "", 0.0, "", False

        try:
            with open(json_path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(f"ERROR reading session_log.json: {exc}")
            return session_path.name, "", 0.0, "", False

        session_id      = doc.get("session_id", session_path.name)
        experiment_name = doc.get("experiment_name", "")
        start_ts        = float(doc.get("start_ts", 0.0))
        end_ts_raw      = doc.get("end_ts")
        end_ts          = float(end_ts_raw) if end_ts_raw is not None else None
        duration_s      = float(end_ts - start_ts) if end_ts else 0.0

        # Derive wall-clock ISO from folder name if possible
        recorded_at = ""
        try:
            # Folder names like 20241201_120000 → approximate wall-clock
            recorded_at = datetime.strptime(
                session_path.name, "%Y%m%d_%H%M%S"
            ).isoformat()
        except ValueError:
            pass

        if duration_s < 10.0 and duration_s > 0.0:
            issues.append(f"WARNING very short session: {duration_s:.1f} s")

        return session_id, experiment_name, duration_s, recorded_at, True

    @staticmethod
    def _read_samples(samples_path: str, issues: list):
        """Return (has_samples, n_frames, eng_fracs, mean_eng)."""
        p = Path(samples_path)
        if not p.exists():
            issues.append(f"WARNING samples.csv not found: {samples_path}")
            return False, 0, {}, 0.0

        state_counts: Dict[str, int] = {}
        score_sum = 0.0
        n = 0

        try:
            with open(p, encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    n += 1
                    state = row.get("engagement_state", "unreliable")
                    state_counts[state] = state_counts.get(state, 0) + 1
                    try:
                        score_sum += float(row.get("smoothed_score", "0.0"))
                    except ValueError:
                        pass
        except OSError as exc:
            issues.append(f"ERROR reading samples.csv: {exc}")
            return False, 0, {}, 0.0

        if n == 0:
            issues.append("WARNING samples.csv is empty")
            return True, 0, {}, 0.0

        eng_fracs = {state: cnt / n for state, cnt in state_counts.items()}
        mean_eng  = score_sum / n
        return True, n, eng_fracs, round(mean_eng, 4)

    @staticmethod
    def _count_embeddings(embeddings_path: str, issues: list):
        """Return (has_embeddings, n_windows)."""
        if not embeddings_path:
            issues.append("WARNING no embeddings path specified or found")
            return False, 0

        p = Path(embeddings_path)
        if not p.exists():
            issues.append(f"WARNING embeddings not found: {embeddings_path}")
            return False, 0

        n = 0
        try:
            with open(p, encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for _ in reader:
                    n += 1
        except OSError as exc:
            issues.append(f"ERROR reading embeddings CSV: {exc}")
            return False, 0

        if n == 0:
            issues.append("WARNING embeddings CSV is empty")

        return True, n

    @staticmethod
    def _read_labels(labels_path: Optional[str], issues: list):
        """Return (has_labels, label_summary_dict)."""
        if not labels_path:
            return False, {}

        p = Path(labels_path)
        if not p.exists():
            issues.append(f"WARNING labels.json not found: {labels_path}")
            return False, {}

        try:
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
            label_summary = {k: v for k, v in doc.get("labels", {}).items()}
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(f"ERROR reading labels.json: {exc}")
            return False, {}

        return True, label_summary
