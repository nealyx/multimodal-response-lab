"""Dataset-level statistics computed from the session catalog.

These statistics answer the key dataset characterisation questions:
  - How many participants and sessions do we have?
  - How much data is there in total?
  - What is the label distribution (needed to detect class imbalance)?
  - What is the engagement state distribution across all sessions?
  - Which sessions have calibration vs labels?

All computation is done from SessionEntry fields (cached aggregates) rather
than re-reading raw files.  This means stats computation is O(n_sessions)
not O(n_frames), and runs in milliseconds even for large catalogs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.dataset.catalog import SessionEntry
from src.dataset.participant import Participant


@dataclass
class PerParticipantStats:
    """Summary statistics for one participant."""
    participant_id:        str
    n_sessions:            int
    total_hours:           float
    total_frames:          int
    total_windows:         int
    has_calibration:       bool
    sessions_with_labels:  int
    mean_engagement_score: float


@dataclass
class DatasetStats:
    """Aggregate statistics for the full multi-session dataset.

    Attributes
    ----------
    n_participants           : distinct participant IDs in the catalog
    n_sessions               : total ingested sessions
    total_hours              : sum of session durations in hours
    total_frames             : sum of n_frames across all sessions
    total_windows            : sum of n_windows across all sessions
    sessions_with_calibration: sessions that have has_calibration=True
    sessions_with_labels     : sessions that have has_labels=True
    sessions_with_embeddings : sessions that have has_embeddings=True
    label_distribution       : task → {"HIGH": N, "LOW": N, "None": N}
    engagement_distribution  : state → weighted fraction (across all frames)
    mean_engagement_score    : dataset-wide weighted mean engagement score
    per_participant          : participant_id → PerParticipantStats
    """
    n_participants:            int
    n_sessions:                int
    total_hours:               float
    total_frames:              int
    total_windows:             int
    sessions_with_calibration: int
    sessions_with_labels:      int
    sessions_with_embeddings:  int
    label_distribution:        Dict[str, Dict[str, int]] = field(default_factory=dict)
    engagement_distribution:   Dict[str, float]          = field(default_factory=dict)
    mean_engagement_score:     float                     = 0.0
    mean_on_screen_fraction:   float                     = 0.0
    mean_blink_rate:           float                     = 0.0
    # Missing-data counts (sessions with zero for that artefact)
    sessions_missing_samples:    int                     = 0
    sessions_missing_embeddings: int                     = 0
    sessions_missing_labels:     int                     = 0
    sessions_missing_calibration:int                     = 0
    per_participant:           Dict[str, PerParticipantStats] = field(default_factory=dict)


class DatasetStatsComputer:
    """Compute DatasetStats from a list of SessionEntry records."""

    @classmethod
    def compute(
        cls,
        entries:      List[SessionEntry],
        participants: Dict[str, Participant],
    ) -> DatasetStats:
        if not entries:
            return DatasetStats(
                n_participants=0, n_sessions=0,
                total_hours=0.0, total_frames=0, total_windows=0,
                sessions_with_calibration=0, sessions_with_labels=0,
                sessions_with_embeddings=0,
            )

        total_frames    = sum(e.n_frames   for e in entries)
        total_windows   = sum(e.n_windows  for e in entries)
        total_duration_s = sum(e.duration_s for e in entries)

        n_calibration = sum(1 for e in entries if e.has_calibration)
        n_labels      = sum(1 for e in entries if e.has_labels)
        n_embeddings  = sum(1 for e in entries if e.has_embeddings)

        n_miss_samples    = sum(1 for e in entries if not e.has_samples)
        n_miss_embeddings = sum(1 for e in entries if not e.has_embeddings)
        n_miss_labels     = sum(1 for e in entries if not e.has_labels)
        n_miss_calib      = sum(1 for e in entries if not e.has_calibration)

        # ── Engagement distribution (frame-weighted) ───────────────────────
        eng_totals: Dict[str, float] = {}
        score_weighted = 0.0
        frames_for_eng = 0

        for e in entries:
            if e.n_frames > 0 and e.engagement_fractions:
                for state, frac in e.engagement_fractions.items():
                    eng_totals[state] = eng_totals.get(state, 0.0) + frac * e.n_frames
                score_weighted += e.mean_engagement_score * e.n_frames
                frames_for_eng += e.n_frames

        on_screen_weighted = sum(
            e.on_screen_fraction * e.n_frames for e in entries if e.n_frames > 0
        )
        blink_weighted = sum(
            e.mean_blink_rate * e.n_frames for e in entries if e.n_frames > 0
        )

        eng_dist: Dict[str, float] = {}
        if frames_for_eng > 0:
            eng_dist = {s: v / frames_for_eng for s, v in eng_totals.items()}
            mean_score      = score_weighted    / frames_for_eng
            mean_on_screen  = on_screen_weighted / frames_for_eng
            mean_blink_rate = blink_weighted    / frames_for_eng
        else:
            mean_score = mean_on_screen = mean_blink_rate = 0.0

        # ── Label distribution ─────────────────────────────────────────────
        label_dist: Dict[str, Dict[str, int]] = {}
        for e in entries:
            if not e.has_labels:
                continue
            for task, lbl in e.label_summary.items():
                if task not in label_dist:
                    label_dist[task] = {"HIGH": 0, "LOW": 0, "ambiguous": 0}
                key = lbl if lbl in ("HIGH", "LOW") else "ambiguous"
                label_dist[task][key] += 1

        # ── Per-participant stats ──────────────────────────────────────────
        by_participant: Dict[str, List[SessionEntry]] = {}
        for e in entries:
            by_participant.setdefault(e.participant_id, []).append(e)

        per_p: Dict[str, PerParticipantStats] = {}
        for pid, sess in by_participant.items():
            pf = sum(s.n_frames for s in sess)
            ps = sum(s.mean_engagement_score * s.n_frames for s in sess)
            per_p[pid] = PerParticipantStats(
                participant_id=        pid,
                n_sessions=            len(sess),
                total_hours=           round(sum(s.duration_s for s in sess) / 3600, 4),
                total_frames=          pf,
                total_windows=         sum(s.n_windows for s in sess),
                has_calibration=       any(s.has_calibration for s in sess),
                sessions_with_labels=  sum(1 for s in sess if s.has_labels),
                mean_engagement_score= round(ps / pf, 4) if pf > 0 else 0.0,
            )

        n_participants = len({e.participant_id for e in entries})

        return DatasetStats(
            n_participants=             n_participants,
            n_sessions=                 len(entries),
            total_hours=                round(total_duration_s / 3600, 4),
            total_frames=               total_frames,
            total_windows=              total_windows,
            sessions_with_calibration=  n_calibration,
            sessions_with_labels=       n_labels,
            sessions_with_embeddings=   n_embeddings,
            label_distribution=         label_dist,
            engagement_distribution=    {k: round(v, 4) for k, v in eng_dist.items()},
            mean_engagement_score=      round(mean_score, 4),
            mean_on_screen_fraction=    round(mean_on_screen, 4),
            mean_blink_rate=            round(mean_blink_rate, 2),
            sessions_missing_samples=   n_miss_samples,
            sessions_missing_embeddings=n_miss_embeddings,
            sessions_missing_labels=    n_miss_labels,
            sessions_missing_calibration=n_miss_calib,
            per_participant=            per_p,
        )
