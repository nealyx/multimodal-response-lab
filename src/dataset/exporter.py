"""Export dataset-level summaries for downstream consumption.

Two export formats are produced:
  dataset_summary.json  — full structured summary with stats, per-session
                          entries, and quality check results. Machine-readable.
  dataset_summary.csv   — flat table, one row per session with key fields.
                          Easy to import into a spreadsheet or R/Python for
                          cohort-level analysis.

Why both formats
----------------
JSON preserves structure (nested per-participant stats, label distributions)
but is awkward to query with spreadsheet tools.  CSV loses structure but is
universally importable and directly usable in pandas/R without parsing.
Providing both means a researcher can immediately start analysis without
writing a parser.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from src.dataset.catalog import SessionEntry
from src.dataset.checker import QualityIssue
from src.dataset.manifest import DatasetManifest
from src.dataset.stats import DatasetStats


class DatasetExporter:
    """Write dataset summaries to disk."""

    @classmethod
    def export_all(
        cls,
        manifest: DatasetManifest,
        stats:    DatasetStats,
        issues:   List[QualityIssue],
        out_dir:  str,
    ) -> Dict[str, str]:
        """Export JSON and CSV summaries to *out_dir*.

        Returns a dict mapping artifact name → file path.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = cls.export_json(manifest, stats, issues, str(out / "dataset_summary.json"))
        csv_path  = cls.export_csv(manifest.all_sessions(), str(out / "dataset_summary.csv"))

        return {"dataset_summary_json": json_path, "dataset_summary_csv": csv_path}

    @classmethod
    def export_json(
        cls,
        manifest: DatasetManifest,
        stats:    DatasetStats,
        issues:   List[QualityIssue],
        path:     str,
    ) -> str:
        """Write dataset_summary.json to *path*."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        issue_counts = {}
        for iss in issues:
            issue_counts[iss.severity] = issue_counts.get(iss.severity, 0) + 1

        doc = {
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "manifest_version": manifest.manifest_version,
            "dataset_root":  manifest.dataset_root,
            "summary": {
                "n_participants":            stats.n_participants,
                "n_sessions":                stats.n_sessions,
                "total_hours":               stats.total_hours,
                "total_frames":              stats.total_frames,
                "total_windows":             stats.total_windows,
                "sessions_with_calibration": stats.sessions_with_calibration,
                "sessions_with_labels":      stats.sessions_with_labels,
                "sessions_with_embeddings":  stats.sessions_with_embeddings,
                "mean_engagement_score":     stats.mean_engagement_score,
            },
            "engagement_distribution": stats.engagement_distribution,
            "label_distribution":      stats.label_distribution,
            "quality_issue_counts":    issue_counts,
            "quality_issues": [
                {
                    "session_id": i.session_id,
                    "issue_type": i.issue_type,
                    "severity":   i.severity,
                    "detail":     i.detail,
                }
                for i in issues if i.severity in ("error", "warning")
            ],
            "per_participant": {
                pid: {
                    "n_sessions":            ps.n_sessions,
                    "total_hours":           ps.total_hours,
                    "total_windows":         ps.total_windows,
                    "has_calibration":       ps.has_calibration,
                    "sessions_with_labels":  ps.sessions_with_labels,
                    "mean_engagement_score": ps.mean_engagement_score,
                }
                for pid, ps in stats.per_participant.items()
            },
            "sessions": [e.to_dict() for e in manifest.all_sessions()],
        }

        with open(p, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)

        return str(p)

    @classmethod
    def export_csv(
        cls,
        entries: List[SessionEntry],
        path:    str,
    ) -> str:
        """Write dataset_summary.csv to *path* (one row per session)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "session_id", "participant_id", "recorded_at", "duration_s",
            "experiment_name", "n_frames", "n_windows",
            "has_session_log", "has_samples", "has_embeddings",
            "has_labels", "has_calibration",
            "mean_engagement_score",
            "focused_frac", "distracted_frac", "fatigued_frac", "unreliable_frac",
            "label_engagement", "label_fatigue", "label_distraction",
            "n_quality_issues",
        ]

        with open(p, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for e in entries:
                fracs = e.engagement_fractions
                labels = e.label_summary
                writer.writerow({
                    "session_id":       e.session_id,
                    "participant_id":   e.participant_id,
                    "recorded_at":      e.recorded_at,
                    "duration_s":       round(e.duration_s, 1),
                    "experiment_name":  e.experiment_name,
                    "n_frames":         e.n_frames,
                    "n_windows":        e.n_windows,
                    "has_session_log":  e.has_session_log,
                    "has_samples":      e.has_samples,
                    "has_embeddings":   e.has_embeddings,
                    "has_labels":       e.has_labels,
                    "has_calibration":  e.has_calibration,
                    "mean_engagement_score": round(e.mean_engagement_score, 4),
                    "focused_frac":     round(fracs.get("focused",     0.0), 4),
                    "distracted_frac":  round(fracs.get("distracted",  0.0), 4),
                    "fatigued_frac":    round(fracs.get("fatigued",    0.0), 4),
                    "unreliable_frac":  round(fracs.get("unreliable",  0.0), 4),
                    "label_engagement": labels.get("engagement",  ""),
                    "label_fatigue":    labels.get("fatigue",     ""),
                    "label_distraction":labels.get("distraction", ""),
                    "n_quality_issues": len(e.quality_issues),
                })

        return str(p)
