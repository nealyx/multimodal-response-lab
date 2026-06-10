"""Map survey responses to per-session labels for supervised training.

Label structure
----------------
A SessionLabel ties a session directory (where behavioral-window CSVs live)
to the labels derived from a SurveyResponse.  The supervised trainer can then
load these labels alongside the feature windows to build a dataset.

Mapping rules (same policy as Day 11 heuristic thresholds, but for human ratings)
------------------------------------------------------------------------------------
  engagement  : HIGH if engagement_q >= 4, LOW if <= 2, else None (excluded)
  fatigue     : HIGH if fatigue_q     >= 4, LOW if <= 2, else None
  distraction : HIGH if distraction_q >= 4, LOW if <= 2, else None

The middle band (score == 3) is excluded rather than forced because the Likert
midpoint is meaningfully ambiguous — respondents use it both as "average" and
as "can't decide."  Excluding it keeps the classes cleaner at the cost of
slightly fewer training samples.

Session-level label → window-level label
-----------------------------------------
The survey produces one label per session.  In supervised training we need one
label per behavioral window (typically 500 ms–2 s).  The current approach
assigns the session-level label to every window, which is a simplification.
Within-session dynamics (early focus, mid-session fatigue, recovery) are lost.
This is noted in the data quality metadata so analysts are not misled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.calibration.survey import SurveyResponse


_LABEL_TASKS = ("engagement", "fatigue", "distraction")


@dataclass
class SessionLabel:
    """Labels for one session derived from a SurveyResponse.

    ``labels`` maps task name → class string ("HIGH" / "LOW" / None).
    None means the session was excluded for that task (ambiguous middle).

    ``session_dir`` and ``survey_path`` are stored for traceability.
    """
    session_dir:  str
    survey_path:  str
    labels:       Dict[str, Optional[str]]   # task → "HIGH" | "LOW" | None
    recall_conf:  int                         # survey recall confidence (1–5)
    notes:        str = ""

    # ── Convenience accessors ─────────────────────────────────────────────────

    def label_for(self, task: str) -> Optional[str]:
        """Return "HIGH", "LOW", or None for *task*."""
        return self.labels.get(task)

    @property
    def is_high_confidence(self) -> bool:
        return self.recall_conf >= 4

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "session_dir":  self.session_dir,
            "survey_path":  self.survey_path,
            "labels":       self.labels,
            "recall_conf":  self.recall_conf,
            "notes":        self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionLabel":
        return cls(
            session_dir= d["session_dir"],
            survey_path= d.get("survey_path", ""),
            labels=      {k: v for k, v in d["labels"].items()},
            recall_conf= int(d.get("recall_conf", 3)),
            notes=       d.get("notes", ""),
        )

    def save(self, path: str) -> None:
        """Write labels to *path* as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SessionLabel":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def try_load(cls, path: Optional[str]) -> Optional["SessionLabel"]:
        if not path:
            return None
        try:
            return cls.load(path)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None


class SessionLabeler:
    """Derive a SessionLabel from a SurveyResponse.

    Usage
    ------
    label = SessionLabeler.from_survey(survey, session_dir=..., survey_path=...)
    label.save(out_path)
    """

    @staticmethod
    def from_survey(
        survey:       SurveyResponse,
        session_dir:  str = "",
        survey_path:  str = "",
        notes:        str = "",
    ) -> SessionLabel:
        """Map survey ratings to class labels for all supported tasks."""
        labels: Dict[str, Optional[str]] = {}

        # engagement
        labels["engagement"] = survey.engagement_level  # HIGH / LOW / None

        # fatigue
        if survey.fatigue_q >= 4:
            labels["fatigue"] = "HIGH"
        elif survey.fatigue_q <= 2:
            labels["fatigue"] = "LOW"
        else:
            labels["fatigue"] = None

        # distraction
        if survey.distraction_q >= 4:
            labels["distraction"] = "HIGH"
        elif survey.distraction_q <= 2:
            labels["distraction"] = "LOW"
        else:
            labels["distraction"] = None

        return SessionLabel(
            session_dir= session_dir or survey.session_dir,
            survey_path= survey_path,
            labels=      labels,
            recall_conf= survey.recall_conf_q,
            notes=       notes,
        )

    @staticmethod
    def load_labels_for_sessions(
        label_paths: List[str],
    ) -> List[SessionLabel]:
        """Load multiple SessionLabel files, skipping any that fail to load."""
        results = []
        for p in label_paths:
            sl = SessionLabel.try_load(p)
            if sl is not None:
                results.append(sl)
        return results
