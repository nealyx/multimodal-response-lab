"""Heuristic label creation for supervised behavioral ML.

Why heuristic labels, not ground truth
----------------------------------------
A supervised model requires labeled training examples ("this window = high
engagement").  True ground truth — a human annotator watching video and marking
cognitive state second-by-second — is expensive, time-consuming, and requires an
IRB-approved study design.  What we have instead are *heuristic labels*: rules
derived from the signal values themselves (e.g., "if focused_frac > 0.7, label as
HIGH_ATTENTION").

Heuristic labels are circular by construction: they partially reflect the same
signals used as features.  A model trained on them learns to reproduce the
heuristic, not to predict a ground-truth cognitive state.  This is useful as a
prototype — it validates the pipeline architecture, exposes feature interactions,
and guides which signals are most discriminative — but it cannot be presented as
evidence of accurate cognitive inference.

How to move toward real ground truth:
  1. Collect sessions where a separate objective measure exists (task accuracy,
     comprehension quiz score, reaction time on validated vigilance task).
  2. Align that external measure to session timestamps.
  3. Replace heuristic labels with the aligned external measure.
  4. Re-train with cross-session validation (not per-session split).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.embedding.features import BehavioralWindow, FEATURE_NAMES

log = logging.getLogger(__name__)


# ── Label configuration ───────────────────────────────────────────────────────

@dataclass
class LabelConfig:
    """Defines a binary or multi-class labeling rule over BehavioralWindows.

    Parameters
    ----------
    task : str
        Logical name for the prediction task, e.g. ``"high_vs_low_attention"``.
    label_type : str
        One of ``"binary"`` or ``"multiclass"``.
    classes : list[str]
        Ordered class names (determines integer encoding: index in list).
    rules : dict[str, Any]
        Task-specific threshold dict consumed by the labeling function.
    min_reliable_frac : float
        Windows with unreliable_frac above this threshold are excluded from
        training (too many frames where MediaPipe lost the face).
    """
    task:               str
    label_type:         str                    # "binary" | "multiclass"
    classes:            List[str]
    rules:              Dict[str, Any]         = field(default_factory=dict)
    min_reliable_frac:  float                  = 0.5

    @property
    def n_classes(self) -> int:
        return len(self.classes)


# ── Built-in label configs ────────────────────────────────────────────────────

def attention_level_config(high_threshold: float = 0.65,
                            low_threshold:  float = 0.40) -> LabelConfig:
    """Binary: high vs low attention based on mean engagement score.

    Heuristic: windows where the mean engagement score (feature 0) exceeds
    ``high_threshold`` → HIGH; below ``low_threshold`` → LOW; middle band is
    excluded (ambiguous).
    """
    return LabelConfig(
        task="high_vs_low_attention",
        label_type="binary",
        classes=["low_attention", "high_attention"],
        rules={
            "high_threshold": high_threshold,
            "low_threshold":  low_threshold,
            "feature":        "engagement_mean",
        },
    )


def focused_vs_distracted_config() -> LabelConfig:
    """Binary: focused vs distracted dominant state.

    Heuristic: uses dominant_state field from the BehavioralWindow.
    Windows whose dominant_state is 'drifting' or 'unreliable' are excluded.
    """
    return LabelConfig(
        task="focused_vs_distracted",
        label_type="binary",
        classes=["distracted", "focused"],
        rules={"method": "dominant_state"},
    )


def fatigue_config(fatigue_threshold: float = 0.30) -> LabelConfig:
    """Binary: fatigued vs alert based on fatigue_frac and ear_mean.

    Heuristic: windows where fatigue_frac (feature 16) exceeds threshold → FATIGUED.
    """
    return LabelConfig(
        task="fatigue_detection",
        label_type="binary",
        classes=["alert", "fatigued"],
        rules={
            "feature":    "fatigue_frac",
            "threshold":  fatigue_threshold,
        },
    )


def engagement_state_config() -> LabelConfig:
    """Multiclass: 4-class engagement state from dominant_state.

    Classes: focused, drifting, distracted, fatigued.
    Windows with dominant_state == 'unreliable' are excluded.
    """
    return LabelConfig(
        task="engagement_state",
        label_type="multiclass",
        classes=["focused", "drifting", "distracted", "fatigued"],
        rules={"method": "dominant_state"},
    )


# Map config task name → factory function (used by CLI)
BUILTIN_TASKS: Dict[str, Any] = {
    "high_vs_low_attention":  attention_level_config,
    "focused_vs_distracted":  focused_vs_distracted_config,
    "fatigue_detection":      fatigue_config,
    "engagement_state":       engagement_state_config,
}


# ── Label assignment ──────────────────────────────────────────────────────────

def assign_labels(
    windows:    List[BehavioralWindow],
    config:     LabelConfig,
) -> tuple:
    """Assign integer labels to windows according to *config*.

    Returns
    -------
    kept_windows : List[BehavioralWindow]
        Windows that received a valid label (ambiguous/excluded windows dropped).
    labels : np.ndarray
        Integer label array, shape (len(kept_windows),).
    label_names : List[str]
        Human-readable name for each integer label (= config.classes).
    excluded_count : int
        Number of windows that were excluded (ambiguous or unreliable).
    """
    feat_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    kept_windows: List[BehavioralWindow] = []
    label_list:   List[int]             = []
    excluded                            = 0

    for w in windows:
        # Quality gate: too many unreliable frames
        unreliable_frac = float(w.features[feat_idx["unreliable_frac"]])
        if unreliable_frac > (1.0 - config.min_reliable_frac):
            excluded += 1
            continue

        label = _label_one(w, config, feat_idx)
        if label is None:
            excluded += 1
            continue

        kept_windows.append(w)
        label_list.append(label)

    return kept_windows, np.array(label_list, dtype=np.int64), config.classes, excluded


def load_human_label_windows(
    windows:     List[BehavioralWindow],
    label_paths: List[str],
    task:        str,
) -> Tuple[List[BehavioralWindow], np.ndarray, List[str], int]:
    """Assign human-survey labels to windows using SessionLabel files.

    Labels are session-level: every window from session *X* receives the same
    label derived from the survey rating for session *X*.  Ambiguous labels
    (None) exclude all windows from that session.

    Parameters
    ----------
    windows     : all BehavioralWindows from one or more sessions
    label_paths : paths to ``labels.json`` files produced by ``run_calibration.py``
    task        : one of "engagement", "fatigue", "distraction"

    Returns
    -------
    kept_windows, labels, label_names, excluded_count
        Same contract as ``assign_labels`` so the trainer pipeline can use either
        source identically.
    """
    from src.calibration.labeler import SessionLabel

    # Load all label files
    session_labels: Dict[str, Optional[str]] = {}
    for p in label_paths:
        sl = SessionLabel.try_load(p)
        if sl is None:
            log.warning("Could not load label file: %s", p)
            continue
        class_label = sl.label_for(task)
        # Use the session_dir as the key so we can match windows by session_id
        session_labels[sl.session_dir] = class_label
        log.info("Loaded human label for '%s': %s → %s", sl.session_dir, task, class_label)

    if not session_labels:
        log.warning("No human label files loaded for task '%s'", task)
        return [], np.array([], dtype=np.int64), ["low", "high"], 0

    label_names = ["low", "high"]

    kept: List[BehavioralWindow] = []
    label_list: List[int] = []
    excluded = 0

    for w in windows:
        # Match window to a session label by session_id substring match
        matched_label: Optional[str] = None
        for sess_dir, cls in session_labels.items():
            if w.session_id and (
                w.session_id in sess_dir or sess_dir.endswith(w.session_id)
            ):
                matched_label = cls
                break

        if matched_label is None:
            excluded += 1
            continue
        if matched_label == "HIGH":
            kept.append(w)
            label_list.append(1)
        elif matched_label == "LOW":
            kept.append(w)
            label_list.append(0)
        else:
            # Ambiguous middle (None)
            excluded += 1

    return kept, np.array(label_list, dtype=np.int64), label_names, excluded


def _label_one(
    w:        BehavioralWindow,
    config:   LabelConfig,
    feat_idx: Dict[str, int],
) -> Optional[int]:
    """Return integer label for one window, or None if ambiguous/excluded."""
    rules = config.rules

    if config.task == "high_vs_low_attention":
        feat  = rules["feature"]
        val   = float(w.features[feat_idx[feat]])
        if val >= rules["high_threshold"]:
            return 1  # high_attention
        if val <= rules["low_threshold"]:
            return 0  # low_attention
        return None   # middle band — excluded

    if config.task == "focused_vs_distracted":
        state = w.dominant_state
        if state == "focused":
            return 1
        if state in ("distracted", "fatigued"):
            return 0
        return None  # drifting or unreliable — excluded

    if config.task == "fatigue_detection":
        feat = rules["feature"]
        val  = float(w.features[feat_idx[feat]])
        return 1 if val >= rules["threshold"] else 0

    if config.task == "engagement_state":
        state = w.dominant_state
        if state in config.classes:
            return config.classes.index(state)
        return None  # unreliable — excluded

    # Custom task: look for feature + threshold in rules
    if "feature" in rules and "threshold" in rules:
        feat = rules["feature"]
        val  = float(w.features[feat_idx[feat]])
        return 1 if val >= rules["threshold"] else 0

    return None
