"""Anomaly detection over behavioral window sequences.

What anomaly detection can and cannot mean here
-----------------------------------------------
CAN detect:
  - Windows where the engagement score drops sharply below the session baseline
    (e.g., a sudden attentional lapse or fatigue episode)
  - Windows with an unusual combination of signal values (e.g., high blink rate
    AND low EAR AND distracted state simultaneously)
  - Behavioral outliers relative to the session distribution

CANNOT determine:
  - Whether an anomaly represents something clinically or cognitively significant
  - The *cause* of the anomaly (task difficulty, distraction, fatigue, artifact)
  - Whether it's a negative event — a "focused" outlier (unusually high engagement)
    is also flagged as anomalous

Two detection methods
---------------------
z_score:
  Flags windows where the engagement_mean deviates more than *threshold* standard
  deviations from the session mean.  Simple, interpretable, requires no training.
  Works well for unimodal session distributions.

isolation_forest:
  Flags windows that are isolated in the full 18-dimensional feature space.
  More sensitive to multi-channel anomalies (e.g., a window with normal engagement
  score but abnormal gaze and head pose simultaneously).  Requires sklearn.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.embedding.features import BehavioralWindow


class AnomalyDetector:
    """Flag behavioral windows that deviate significantly from session norms.

    Parameters
    ----------
    method : str
        ``"z_score"`` (default) or ``"isolation_forest"``.
    threshold : float
        Z-score method: |z| > threshold → anomaly (default 2.5).
    contamination : float
        IsolationForest: expected proportion of anomalies (default 0.10).
    """

    def __init__(
        self,
        method:        str   = "z_score",
        threshold:     float = 2.5,
        contamination: float = 0.10,
    ) -> None:
        if method not in ("z_score", "isolation_forest"):
            raise ValueError(f"Unknown anomaly method: {method!r}")
        self._method        = method
        self._threshold     = threshold
        self._contamination = contamination

    def fit_detect(self, windows: List[BehavioralWindow]) -> np.ndarray:
        """Return a boolean array of shape (n_windows,).

        True = anomaly.  Requires at least 3 windows to produce meaningful
        results; returns all-False for shorter inputs.
        """
        n = len(windows)
        if n < 3:
            return np.zeros(n, dtype=bool)

        if self._method == "isolation_forest":
            return self._isolation_forest(windows)
        return self._z_score(windows)

    # ── Detection methods ─────────────────────────────────────────────────────

    def _z_score(self, windows: List[BehavioralWindow]) -> np.ndarray:
        """Flag windows whose engagement_mean is an outlier (|z| > threshold)."""
        scores = np.array([w.features[0] for w in windows], dtype=float)
        std    = scores.std()
        if std < 1e-9:
            return np.zeros(len(windows), dtype=bool)
        z = np.abs((scores - scores.mean()) / std)
        return z > self._threshold

    def _isolation_forest(self, windows: List[BehavioralWindow]) -> np.ndarray:
        """Flag windows that are isolated in the full feature space."""
        from sklearn.ensemble import IsolationForest
        X = np.stack([w.features for w in windows])
        clf = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=100,
        )
        preds = clf.fit_predict(X)
        return preds == -1   # -1 = anomaly in sklearn convention
