"""Load behavioral window data from Day 10 CSV outputs and build ML datasets.

Why train/test split can be misleading on small personal datasets
------------------------------------------------------------------
A typical session produces ~60–300 windows (2 s window, 0.5 s stride over a
5–20 minute session, @30 fps).  A standard 80/20 split on 100 windows gives 20
test samples.  Accuracy computed over 20 samples has a 95% confidence interval
of roughly ±10 percentage points for a random classifier and wider for a
real one.  The split is also time-correlated: consecutive windows share frames,
so test samples may be "too close" to training samples in feature space.

This means reported metrics on a single-session split are illustrative, not
reliable.  To get trustworthy metrics you need either:
  - Multiple independent sessions from the same person (within-person CV)
  - Sessions from multiple people (between-person CV)
  - A temporal hold-out (train on first 80% of session, test on last 20%)

The pipeline supports all three via the split_strategy parameter.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.embedding.features import BehavioralWindow, FEATURE_NAMES, N_FEATURES
from src.supervised.labels import LabelConfig, assign_labels


@dataclass
class BehavioralDataset:
    """Feature matrix + label vector ready for sklearn estimators.

    Attributes
    ----------
    X : np.ndarray, shape (n_samples, N_FEATURES)
    y : np.ndarray, shape (n_samples,)
    feature_names : List[str]
    label_names : List[str]
    task : str
    n_excluded : int     — windows dropped during label assignment
    session_ids : List[str]
    window_ids : List[str]
    """
    X:             np.ndarray
    y:             np.ndarray
    feature_names: List[str]
    label_names:   List[str]
    task:          str
    n_excluded:    int
    session_ids:   List[str]
    window_ids:    List[str]

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for i, name in enumerate(self.label_names):
            counts[name] = int(np.sum(self.y == i))
        return counts


def load_windows_csv(path: str) -> List[BehavioralWindow]:
    """Load behavioral_windows.csv written by embed_session.py.

    Tolerant: missing columns get safe defaults so older CSV files still load.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"behavioral_windows.csv not found: {path}")

    windows: List[BehavioralWindow] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            feat_vec = np.array(
                [float(row.get(name, 0.0)) for name in FEATURE_NAMES],
                dtype=np.float32,
            )
            w = BehavioralWindow(
                window_id=           row.get("window_id", ""),
                session_id=          row.get("session_id", ""),
                start_ts=            float(row.get("start_ts", 0.0)),
                end_ts=              float(row.get("end_ts", 0.0)),
                n_frames=            int(row.get("n_frames", 0)),
                dominant_state=      row.get("dominant_state", "unreliable"),
                active_stimulus_ids= _parse_stim_ids(row.get("active_stimulus_ids", "")),
                features=            feat_vec,
                metadata={
                    "cluster_label": row.get("cluster_label", ""),
                    "is_anomaly":    row.get("is_anomaly", ""),
                },
            )
            windows.append(w)
    return windows


def load_windows_multi(csv_paths: List[str]) -> List[BehavioralWindow]:
    """Load and concatenate multiple behavioral_windows.csv files."""
    all_windows: List[BehavioralWindow] = []
    for path in csv_paths:
        all_windows.extend(load_windows_csv(path))
    return all_windows


def build_dataset(
    windows: List[BehavioralWindow],
    config:  LabelConfig,
) -> BehavioralDataset:
    """Assign labels and build a BehavioralDataset from windows."""
    kept, labels, label_names, n_excluded = assign_labels(windows, config)

    if not kept:
        X = np.empty((0, N_FEATURES), dtype=np.float32)
        return BehavioralDataset(
            X=X, y=np.array([], dtype=np.int64),
            feature_names=list(FEATURE_NAMES),
            label_names=list(label_names),
            task=config.task,
            n_excluded=n_excluded,
            session_ids=[],
            window_ids=[],
        )

    X = np.stack([w.features for w in kept], axis=0).astype(np.float32)
    return BehavioralDataset(
        X=X,
        y=labels,
        feature_names=list(FEATURE_NAMES),
        label_names=list(label_names),
        task=config.task,
        n_excluded=n_excluded,
        session_ids=[w.session_id for w in kept],
        window_ids= [w.window_id  for w in kept],
    )


def train_test_split_dataset(
    dataset:        BehavioralDataset,
    test_size:      float = 0.20,
    strategy:       str   = "random",
    random_state:   int   = 42,
) -> Tuple[BehavioralDataset, BehavioralDataset]:
    """Split a BehavioralDataset into train/test subsets.

    Parameters
    ----------
    strategy : str
        ``"random"``   — stratified random split (preserves class balance).
        ``"temporal"`` — first (1-test_size) fraction in time → train,
                         last test_size fraction → test.  Avoids leaking
                         temporally adjacent windows across train/test boundary.
    """
    n = dataset.n_samples
    if n < 4:
        # Not enough data for any split — return full dataset as both
        return dataset, dataset

    if strategy == "temporal":
        split_idx = max(1, int(n * (1.0 - test_size)))
        train_idx = np.arange(split_idx)
        test_idx  = np.arange(split_idx, n)
    else:
        # Stratified random split via sklearn
        from sklearn.model_selection import train_test_split
        idx = np.arange(n)
        # Fall back to non-stratified if only 1 class after label assignment
        unique_classes = np.unique(dataset.y)
        stratify = dataset.y if len(unique_classes) > 1 else None
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    def _subset(indices):
        return BehavioralDataset(
            X=dataset.X[indices],
            y=dataset.y[indices],
            feature_names=dataset.feature_names,
            label_names=dataset.label_names,
            task=dataset.task,
            n_excluded=0,
            session_ids=[dataset.session_ids[i] for i in indices],
            window_ids= [dataset.window_ids[i]  for i in indices],
        )

    return _subset(train_idx), _subset(test_idx)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_stim_ids(s: str) -> List[str]:
    if not s or s.strip() == "":
        return []
    return [x for x in s.split("|") if x]
