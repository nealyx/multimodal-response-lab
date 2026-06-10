"""Dataset quality checker: validate session entries for completeness.

Quality checks are divided into three severity levels:

ERROR   — A problem that will cause downstream failures:
           session_log.json missing, samples.csv corrupt, embeddings CSV empty
           when n_windows > 0 was expected.

WARNING — A problem that degrades usefulness but does not block analysis:
           no embeddings (can't run ML), no labels (can't train supervised),
           session very short (< 60 s), face detection rate below threshold.

INFO    — Informational gaps that may be intentional:
           no calibration profile (personalisation disabled),
           no human labels (heuristic labels only).

All checks operate on the cached SessionEntry fields — no file I/O occurs.
The one exception is the path-existence check, which tests whether recorded
file paths still exist on disk (detects sessions that were moved after ingestion).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.dataset.catalog import SessionEntry

_MIN_DURATION_S   = 60.0    # sessions shorter than this get a WARNING
_MIN_FACE_DETECT  = 0.5     # below 50% face detection → WARNING
                             # (not stored in SessionEntry currently, kept for future)


@dataclass
class QualityIssue:
    """One quality problem found in a session entry.

    Attributes
    ----------
    session_id : the session this issue belongs to
    issue_type : short machine-readable code (e.g. "missing_session_log")
    severity   : "error" | "warning" | "info"
    detail     : human-readable description
    """
    session_id: str
    issue_type: str
    severity:   str    # "error" | "warning" | "info"
    detail:     str

    def __str__(self) -> str:
        return f"[{self.severity.upper():7}] {self.session_id}: {self.detail}"


class DatasetChecker:
    """Run quality checks across all session entries in the catalog."""

    @classmethod
    def check_all(cls, entries: List[SessionEntry]) -> List[QualityIssue]:
        """Return all quality issues found across *entries*."""
        issues: List[QualityIssue] = []
        for entry in entries:
            issues.extend(cls.check_session(entry))
        return issues

    @classmethod
    def check_session(cls, entry: SessionEntry) -> List[QualityIssue]:
        """Return quality issues for a single SessionEntry."""
        issues: List[QualityIssue] = []
        sid = entry.session_id

        def add(issue_type: str, severity: str, detail: str) -> None:
            issues.append(QualityIssue(sid, issue_type, severity, detail))

        # ── Errors: blocking ──────────────────────────────────────────────────
        if not entry.has_session_log:
            add("missing_session_log", "error",
                "session_log.json is missing — session identity unknown")

        if entry.has_samples and entry.n_frames == 0:
            add("empty_samples", "error",
                "samples.csv exists but contains 0 rows")

        if entry.has_embeddings and entry.n_windows == 0:
            add("empty_embeddings", "error",
                "behavioral_windows.csv exists but contains 0 rows")

        if not entry.participant_id:
            add("no_participant", "error",
                "session has no participant_id — cannot group by person")

        # ── Warnings: degrade usefulness ─────────────────────────────────────
        if not entry.has_samples:
            add("missing_samples", "warning",
                "samples.csv not found — frame-level metrics unavailable")

        if not entry.has_embeddings:
            add("missing_embeddings", "warning",
                "behavioral_windows.csv not found — ML training unavailable")

        if entry.duration_s > 0 and entry.duration_s < _MIN_DURATION_S:
            add("short_session", "warning",
                f"session is only {entry.duration_s:.1f} s — less than "
                f"{_MIN_DURATION_S:.0f} s minimum recommended")

        # ── Path existence checks: warn if recorded path no longer exists ─────
        for attr, label in [
            ("session_dir",      "session directory"),
            ("samples_path",     "samples.csv"),
            ("embeddings_path",  "behavioral_windows.csv"),
            ("labels_path",      "labels.json"),
            ("calibration_path", "user_profile.json"),
        ]:
            p = getattr(entry, attr, "")
            if p and not Path(p).exists():
                add(f"path_not_found_{attr}", "warning",
                    f"{label} recorded in manifest no longer exists: {p}")

        # ── Ingestion-time issues carried forward ────────────────────────────
        for msg in entry.quality_issues:
            if msg.startswith("ERROR"):
                add("ingestion_error", "error", msg)
            elif msg.startswith("WARNING"):
                add("ingestion_warning", "warning", msg)

        # ── Info: optional but useful ─────────────────────────────────────────
        if not entry.has_calibration:
            add("no_calibration", "info",
                "no calibration profile — personalised thresholds unavailable")

        if not entry.has_labels:
            add("no_labels", "info",
                "no human-survey labels — only heuristic labels available for ML")

        return issues

    @classmethod
    def summary(cls, issues: List[QualityIssue]) -> dict:
        """Return a dict with error/warning/info counts."""
        counts = {"error": 0, "warning": 0, "info": 0}
        for issue in issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return counts
