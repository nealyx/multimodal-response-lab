"""Post-session self-report survey.

Survey design notes
--------------------
Five Likert-scale questions (1–5) covering the main dimensions the system
tries to infer from signals.  Keeping it short maximises completion rate;
longer surveys suffer from fatigue effects and retrospective-reporting bias.

Questions are ordered from most salient (focus) to least (meta: confidence
in recall) so that the most important labels are captured even if the user
quits early.

Scale anchor: 1 = "Not at all / Very low", 5 = "Completely / Very high"

Label derivation (SessionLabeler uses these)
---------------------------------------------
  engagement_level : "HIGH" if engagement_q >= 4, "LOW" if <= 2, else None
  fatigue_present  : True if fatigue_q >= 4
  high_distraction : True if distraction_q >= 4
  high_difficulty  : True if difficulty_q >= 4

These deliberately binary thresholds match the two-class structure used in
the supervised-ML evaluation.  The middle region (3) is excluded rather than
forcing an arbitrary assignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Survey questions ──────────────────────────────────────────────────────────

_QUESTIONS = [
    ("engagement_q",   "How engaged or focused were you during this session?"),
    ("fatigue_q",      "How fatigued or tired did you feel during this session?"),
    ("distraction_q",  "How often were you distracted from the task?"),
    ("difficulty_q",   "How difficult or cognitively demanding was the task?"),
    ("recall_conf_q",  "How confident are you in your answers above?"),
]


@dataclass
class SurveyResponse:
    """Self-reported ratings from a post-session questionnaire.

    All five fields are Likert scores in [1, 5] where 1 = very low / not at all
    and 5 = very high / completely.

    ``session_dir`` stores the path to the session that was rated so that the
    labeler can associate the response with the correct feature windows.
    """
    engagement_q:  int   # 1=not engaged … 5=fully engaged
    fatigue_q:     int   # 1=not fatigued … 5=very fatigued
    distraction_q: int   # 1=not distracted … 5=very distracted
    difficulty_q:  int   # 1=not difficult … 5=very difficult
    recall_conf_q: int   # 1=not confident … 5=very confident in recall
    session_dir:   str = ""   # path to the rated session directory

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def engagement_level(self) -> Optional[str]:
        """'HIGH', 'LOW', or None (ambiguous middle)."""
        if self.engagement_q >= 4:
            return "HIGH"
        if self.engagement_q <= 2:
            return "LOW"
        return None

    @property
    def fatigue_present(self) -> bool:
        return self.fatigue_q >= 4

    @property
    def high_distraction(self) -> bool:
        return self.distraction_q >= 4

    @property
    def high_difficulty(self) -> bool:
        return self.difficulty_q >= 4

    @property
    def is_high_confidence(self) -> bool:
        """Recall confidence ≥ 4 — responses are more likely to be reliable."""
        return self.recall_conf_q >= 4

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "engagement_q":  self.engagement_q,
            "fatigue_q":     self.fatigue_q,
            "distraction_q": self.distraction_q,
            "difficulty_q":  self.difficulty_q,
            "recall_conf_q": self.recall_conf_q,
            "session_dir":   self.session_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SurveyResponse":
        return cls(
            engagement_q=  int(d["engagement_q"]),
            fatigue_q=     int(d["fatigue_q"]),
            distraction_q= int(d["distraction_q"]),
            difficulty_q=  int(d["difficulty_q"]),
            recall_conf_q= int(d["recall_conf_q"]),
            session_dir=   d.get("session_dir", ""),
        )

    def save(self, path: str) -> None:
        """Write survey response to *path* as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "SurveyResponse":
        """Load a survey response from *path*."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def try_load(cls, path: Optional[str]) -> Optional["SurveyResponse"]:
        """Load if *path* exists, else return None."""
        if not path:
            return None
        try:
            return cls.load(path)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None


# ── Interactive CLI runner ────────────────────────────────────────────────────

def run_survey_cli(session_dir: str = "") -> SurveyResponse:
    """Run the interactive post-session survey in the terminal.

    Prompts the user for each question and validates input.  Returns a
    completed SurveyResponse.  Does not save to disk — caller is responsible
    for calling `response.save(path)`.
    """
    print()
    print("=" * 60)
    print("  Post-Session Survey")
    print("  Rate each item from 1 (very low) to 5 (very high).")
    print("=" * 60)
    print()

    answers: dict = {}
    for key, question in _QUESTIONS:
        while True:
            try:
                raw = input(f"  {question}\n  [1–5]: ").strip()
                value = int(raw)
                if 1 <= value <= 5:
                    answers[key] = value
                    break
                print("  Please enter a number between 1 and 5.")
            except (ValueError, EOFError):
                print("  Please enter a number between 1 and 5.")

    print()
    print("  Thank you. Survey complete.")
    print()

    return SurveyResponse(**answers, session_dir=session_dir)
