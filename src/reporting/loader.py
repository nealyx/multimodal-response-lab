"""Load a SessionLog from files produced by SessionExporter (Day 7).

Handles the two-file layout:
    <session_dir>/session_log.json  — stimulus/response events + metadata
    <session_dir>/samples.csv       — per-frame BehavioralSample time series

The samples are stored separately because a 30-FPS, 5-minute session produces
~9000 rows which would bloat the JSON and slow down event-level inspection.

Loading is intentionally tolerant:
  - Extra JSON keys are ignored.
  - Missing CSV columns receive safe defaults.
  - Rows with parse errors are skipped with a warning.
  - Sessions with no samples (very short, user quit early) load cleanly.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

from src.stimulus.events import (
    BehavioralSample,
    ResponseEvent,
    SessionLog,
    StimulusEvent,
    TrialSummary,
)

log = logging.getLogger(__name__)


def load_session_dir(session_dir: str) -> SessionLog:
    """Load a complete session from a directory containing JSON + CSV files."""
    p = Path(session_dir)
    json_path = p / "session_log.json"
    csv_path  = p / "samples.csv"

    if not json_path.exists():
        raise FileNotFoundError(f"session_log.json not found in {session_dir}")

    session_log = load_session_json(str(json_path))

    if csv_path.exists():
        session_log.samples = _load_samples(str(csv_path))
        log.info("Loaded %d samples from %s", len(session_log.samples), csv_path)
    else:
        log.warning("samples.csv not found in %s — sample-based metrics unavailable", p)

    return session_log


def load_session_json(json_path: str) -> SessionLog:
    """Load a SessionLog from a session_log.json file (no samples)."""
    with open(json_path, encoding="utf-8") as fh:
        doc = json.load(fh)

    session_log = SessionLog(
        session_id=      doc.get("session_id",      "unknown"),
        experiment_name= doc.get("experiment_name", "unknown"),
        start_ts=        float(doc.get("start_ts",  0.0)),
        end_ts=          _opt_float(doc.get("end_ts")),
        config=          doc.get("config", {}),
    )

    for raw in doc.get("stimulus_events", []):
        try:
            session_log.stimulus_events.append(_stim_from_dict(raw))
        except Exception as exc:
            log.warning("Skipping malformed stimulus event: %s", exc)

    for raw in doc.get("response_events", []):
        try:
            session_log.response_events.append(_resp_from_dict(raw))
        except Exception as exc:
            log.warning("Skipping malformed response event: %s", exc)

    for raw in doc.get("trial_summaries", []):
        try:
            session_log.trial_summaries.append(_trial_from_dict(raw))
        except Exception as exc:
            log.warning("Skipping malformed trial summary: %s", exc)

    return session_log


# ── Reconstruction helpers ────────────────────────────────────────────────────

def _stim_from_dict(d: dict) -> StimulusEvent:
    return StimulusEvent(
        stimulus_id=  str(d["stimulus_id"]),
        trial_index=  int(d["trial_index"]),
        stimulus_type=str(d["stimulus_type"]),
        onset_ts=     float(d["onset_ts"]),
        offset_ts=    _opt_float(d.get("offset_ts")),
        duration_ms=  float(d["duration_ms"]),
        metadata=     d.get("metadata") or {},
    )


def _resp_from_dict(d: dict) -> ResponseEvent:
    return ResponseEvent(
        stimulus_id=  str(d["stimulus_id"]),
        trial_index=  int(d["trial_index"]),
        response_type=str(d["response_type"]),
        response_ts=  float(d["response_ts"]),
        latency_ms=   float(d["latency_ms"]),
        hit=          bool(d.get("hit", True)),
        metadata=     d.get("metadata") or {},
    )


def _trial_from_dict(d: dict) -> TrialSummary:
    return TrialSummary(
        stimulus_id=         str(d["stimulus_id"]),
        trial_index=         int(d["trial_index"]),
        stimulus_type=       str(d["stimulus_type"]),
        onset_ts=            float(d["onset_ts"]),
        duration_ms=         float(d["duration_ms"]),
        reaction_latency_ms= _opt_float(d.get("reaction_latency_ms")),
        response_type=       d.get("response_type"),
        hit=                 bool(d.get("hit", False)),
        baseline_score=      float(d.get("baseline_score", 0.5)),
        during_score=        float(d.get("during_score",   0.0)),
        recovery_s=          _opt_float(d.get("recovery_s")),
    )


def _load_samples(csv_path: str) -> List[BehavioralSample]:
    samples: List[BehavioralSample] = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            try:
                samples.append(_sample_from_row(row))
            except Exception as exc:
                log.warning("Skipping row %d in samples.csv: %s", i + 2, exc)
    return samples


def _sample_from_row(row: dict) -> BehavioralSample:
    return BehavioralSample(
        frame_index=        int(_get(row, "frame_index", "0")),
        timestamp=          float(_get(row, "timestamp", "0.0")),
        active_stimulus_id= _nonempty(row.get("active_stimulus_id", "")),
        engagement_state=   _get(row, "engagement_state", "unreliable"),
        smoothed_score=     float(_get(row, "smoothed_score", "0.5")),
        confidence=         float(_get(row, "confidence",     "0.0")),
        focused_fraction=   float(_get(row, "focused_fraction", "0.0")),
        gaze_zone=          _get(row, "gaze_zone",  "unknown"),
        is_on_screen=       _bool(row.get("is_on_screen", "False")),
        gaze_h=             float(_get(row, "gaze_h", "0.5")),
        gaze_v=             float(_get(row, "gaze_v", "0.5")),
        head_zone=          _get(row, "head_zone",  "unknown"),
        head_yaw=           float(_get(row, "head_yaw",   "0.0")),
        head_pitch=         float(_get(row, "head_pitch", "0.0")),
        blink_state=        _get(row, "blink_state", "unknown"),
        mean_ear=           float(_get(row, "mean_ear",    "0.0")),
        blink_rate=         float(_get(row, "blink_rate",  "0.0")),
        is_fatigued=        _bool(row.get("is_fatigued", "False")),
    )


def _opt_float(v) -> Optional[float]:
    if v is None or v == "" or v == "None":
        return None
    return float(v)


def _nonempty(s: str) -> Optional[str]:
    return s if s and s.strip() else None


def _bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def _get(row: dict, key: str, default: str) -> str:
    v = row.get(key, default)
    return v if v and v.strip() else default
