"""Session-level and stimulus-level embedding computation.

A session embedding is a single fixed-length vector that represents the
behavioral character of an entire session.  It is computed as the mean
of all per-window feature vectors — a simple but interpretable aggregation
that preserves the feature semantics (each dimension still represents the
same signal channel as in the per-window vectors).

Per-stimulus embeddings isolate windows that overlapped with a specific
stimulus trial.  Comparing per-stimulus embeddings across trial types or
across sessions gives a feature-space answer to: "did this stimulus type
produce a consistently different behavioral state?"

Connection to Egra-style representation learning
-------------------------------------------------
An Egra-style system would learn task-specific representations from labeled
behavioral data (e.g., "high comprehension" vs. "low comprehension" video
segments).  The session and stimulus embeddings produced here are the
*input* to that representation learning step — they convert variable-length
behavioral time series into fixed vectors that a classifier, regression model,
or contrastive learner can consume.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.embedding.features import BehavioralWindow, N_FEATURES


class SessionEncoder:
    """Aggregate per-window vectors into session and stimulus embeddings."""

    def encode_session(self, windows: List[BehavioralWindow]) -> np.ndarray:
        """Return the mean feature vector across all windows.

        Shape: (N_FEATURES,), dtype float32.
        Returns a neutral 0.5-filled vector for empty input.
        """
        if not windows:
            return np.full(N_FEATURES, 0.5, dtype=np.float32)
        stack = np.stack([w.features for w in windows])   # (n_windows, N_FEATURES)
        return stack.mean(axis=0).astype(np.float32)

    def encode_stimulus_windows(
        self,
        windows:     List[BehavioralWindow],
        stimulus_id: str,
    ) -> Optional[np.ndarray]:
        """Mean embedding of windows that overlapped *stimulus_id*.

        Returns None if no overlapping windows exist.
        """
        relevant = [w for w in windows if stimulus_id in w.active_stimulus_ids]
        if not relevant:
            return None
        stack = np.stack([w.features for w in relevant])
        return stack.mean(axis=0).astype(np.float32)

    def encode_all_stimuli(
        self,
        windows:         List[BehavioralWindow],
        stimulus_events: list,   # List[StimulusEvent]
    ) -> Dict[str, np.ndarray]:
        """Return {stimulus_id: embedding} for every stimulus in *stimulus_events*."""
        result: Dict[str, np.ndarray] = {}
        for ev in stimulus_events:
            vec = self.encode_stimulus_windows(windows, ev.stimulus_id)
            if vec is not None:
                result[ev.stimulus_id] = vec
        return result
