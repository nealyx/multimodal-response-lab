"""Session data export to JSON and CSV.

File layout
-----------
  <output_dir>/<session_id>/
    session_log.json   — full event log + metadata + trial summaries
    samples.csv        — one row per webcam frame (BehavioralSample)
    events.csv         — stimulus events and response events, merged by time

Why two formats?
----------------
  JSON  — preserves nested structure (metadata dicts, full config) and is
          easy to reload programmatically for further analysis.
  CSV   — flat tables are directly usable in pandas, R, Excel, and standard
          statistical tools without a JSON parser.

dataclasses.asdict()
--------------------
Used for both formats.  It recursively converts nested dataclasses and
replaces Enum values with their .value strings (Python dataclass behaviour).
Non-serialisable types (tuples inside metadata dicts) are handled by the
_json_default fallback in to_json and converted to lists for CSV rows.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.stimulus.events import SessionLog, TrialSummary


class SessionExporter:
    """Serialises a SessionLog to disk."""

    @staticmethod
    def to_json(
        log:       SessionLog,
        path:      str,
        summaries: Optional[List[TrialSummary]] = None,
        aggregate: Optional[Dict[str, Any]] = None,
        overall:   Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write full session log + optional analytics to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        doc = {
            "session_id":      log.session_id,
            "experiment_name": log.experiment_name,
            "start_ts":        log.start_ts,
            "end_ts":          log.end_ts,
            "duration_s":      log.duration_s,
            "config":          log.config,
            "stimulus_events": [asdict(e) for e in log.stimulus_events],
            "response_events": [asdict(r) for r in log.response_events],
            "trial_summaries": [asdict(t) for t in summaries] if summaries else [],
            "aggregate_stats": aggregate or {},
            "overall_stats":   overall   or {},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, default=_json_default)

    @staticmethod
    def to_csv(
        log:          SessionLog,
        samples_path: str,
        events_path:  str,
    ) -> None:
        """Write behavioral samples and events to separate CSV files."""
        Path(samples_path).parent.mkdir(parents=True, exist_ok=True)
        Path(events_path).parent.mkdir(parents=True, exist_ok=True)

        # ── samples.csv ───────────────────────────────────────────────────
        if log.samples:
            rows   = [asdict(s) for s in log.samples]
            fields = list(rows[0].keys())
            with open(samples_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(_flatten_rows(rows))

        # ── events.csv ────────────────────────────────────────────────────
        ev_rows: List[Dict[str, Any]] = []
        for e in log.stimulus_events:
            row              = asdict(e)
            row["event_kind"] = "stimulus"
            ev_rows.append(row)
        for r in log.response_events:
            row              = asdict(r)
            row["event_kind"] = "response"
            ev_rows.append(row)

        if ev_rows:
            ev_rows.sort(key=lambda r: r.get("onset_ts") or r.get("response_ts") or 0.0)
            # Union of all keys across both event types
            all_fields: list = []
            seen: set = set()
            for row in ev_rows:
                for k in row:
                    if k not in seen:
                        all_fields.append(k)
                        seen.add(k)

            with open(events_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=all_fields, extrasaction="ignore",
                    restval="",
                )
                writer.writeheader()
                writer.writerows(_flatten_rows(ev_rows))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (tuple, set, frozenset)):
        return list(obj)
    try:
        return str(obj)
    except Exception:
        return None


def _flatten_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert nested dicts/lists inside rows to JSON strings for CSV compat."""
    out = []
    for row in rows:
        flat = {}
        for k, v in row.items():
            if isinstance(v, (dict, list, tuple)):
                flat[k] = json.dumps(v, default=_json_default)
            else:
                flat[k] = v
        out.append(flat)
    return out
