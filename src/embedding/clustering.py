"""Unsupervised clustering of behavioral windows.

Why unsupervised clustering before supervised ML
-------------------------------------------------
Supervised behavioral ML requires labeled ground truth (e.g., "this window
is a high-comprehension state, this one is low-comprehension").  Collecting
such labels is expensive and requires a validated protocol.

Unsupervised clustering serves as a prior step:
  1. It reveals naturally occurring behavioral states without requiring labels.
  2. Clusters can be *inspected* to understand what behavioral combinations
     tend to co-occur — e.g., a cluster might correspond to "high gaze
     stability + high engagement + normal blink rate" without anyone having
     labeled it "focused reading".
  3. Cluster assignments become weakly supervised labels for the next ML
     iteration — if three clear clusters emerge, they can be given tentative
     names ("focused", "wandering", "fatigued") and used to train a classifier.
  4. Cross-session cluster comparison: if a user consistently falls into the
     "wandering" cluster during a particular task type, that's a meaningful
     behavioral pattern even without a supervised label.

Cluster interpretation notes
------------------------------
K-means clusters in 18-dimensional feature space are not directly interpretable
from their labels (0, 1, 2, ...).  Inspect cluster_centers to understand what
behavioral combination defines each cluster.  The dominant_state distribution
within each cluster also helps with interpretation.

Silhouette score
-----------------
Ranges from -1 (poor separation) to +1 (well-separated clusters).
Scores > 0.4 suggest meaningful structure.  Scores near 0 suggest the data
does not naturally cluster at the chosen k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.embedding.features import BehavioralWindow, N_FEATURES


@dataclass
class ClusterResult:
    """Output of BehavioralClusterer.fit()."""
    n_clusters:       int
    labels:           np.ndarray      # shape (n_windows,), int
    cluster_centers:  np.ndarray      # shape (n_clusters, N_FEATURES)
    inertia:          float
    silhouette_score: Optional[float] # None when < 2 clusters or < 2 windows


class BehavioralClusterer:
    """K-Means clustering of behavioral window feature vectors.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (default 3).  Automatically reduced if there
        are fewer windows than clusters.
    random_state : int
        For reproducible results (default 42).
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        self.n_clusters   = n_clusters
        self.random_state = random_state

    def fit(self, windows: List[BehavioralWindow]) -> ClusterResult:
        """Cluster *windows* and return a ClusterResult.

        Degrades gracefully:
          - Fewer windows than n_clusters → reduces k to n_windows.
          - Fewer than 2 windows → returns a trivial single-cluster result.
        """
        n = len(windows)
        if n < 2:
            vec = windows[0].features if windows else np.zeros(N_FEATURES, dtype=np.float32)
            return ClusterResult(
                n_clusters=1,
                labels=np.zeros(n, dtype=int),
                cluster_centers=vec.reshape(1, -1),
                inertia=0.0,
                silhouette_score=None,
            )

        k = min(self.n_clusters, n)
        X = np.stack([w.features for w in windows])

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(X)

        sil: Optional[float] = None
        if k > 1 and len(set(labels.tolist())) > 1:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(X, labels))

        return ClusterResult(
            n_clusters=k,
            labels=labels,
            cluster_centers=km.cluster_centers_.astype(np.float32),
            inertia=float(km.inertia_),
            silhouette_score=sil,
        )
