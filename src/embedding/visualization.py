"""2-D embedding visualization using PCA (and optionally UMAP).

PCA is preferred over raw feature dimensions for visualization because:
  - 18 features cannot be visualized directly.
  - PCA finds the two directions of maximum variance, giving the best
    2-D linear projection of the behavioral space.
  - PC1 and PC2 explained-variance percentages quantify how much of
    the structure is captured by the plot.

UMAP (if installed) preserves non-linear neighbourhood structure and
often produces clearer cluster separation, at the cost of interpretability
(UMAP axes have no direct feature meaning).

The plot has two panels:
  Left  — windows colored by dominant behavioral state
  Right — windows colored by K-Means cluster, with anomalies marked ×
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.embedding.clustering import ClusterResult
from src.embedding.features import BehavioralWindow

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import umap as _umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

_STATE_COLORS = {
    "focused":    "#27AE60",
    "drifting":   "#F1C40F",
    "distracted": "#E67E22",
    "fatigued":   "#E74C3C",
    "unreliable": "#95A5A6",
}
_CLUSTER_PALETTE = ["#3498DB", "#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C",
                    "#E67E22", "#2ECC71", "#E91E63", "#00BCD4"]


def plot_embeddings(
    windows:        List[BehavioralWindow],
    cluster_result: ClusterResult,
    anomaly_flags:  np.ndarray,
    session_id:     str = "session",
    use_umap:       bool = False,
) -> Optional[Any]:
    """Generate a 2-panel PCA (or UMAP) scatter plot.

    Returns a matplotlib Figure, or None if matplotlib is unavailable or
    if there are fewer than 3 windows.
    """
    if not HAS_MPL:
        log.warning("matplotlib not available — skipping embedding plot")
        return None
    if len(windows) < 3:
        log.info("Too few windows (%d) to generate embedding plot", len(windows))
        return None

    X = np.stack([w.features for w in windows])

    # ── Dimensionality reduction ──────────────────────────────────────────
    if use_umap and HAS_UMAP:
        coords, method, var_info = _reduce_umap(X)
    else:
        coords, method, var_info = _reduce_pca(X)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = _plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(
        f"Behavioral Embeddings — {session_id}  ({method}, {len(windows)} windows)",
        color="#C9D1D9", fontsize=12, y=1.01,
    )

    _panel_states(axes[0], coords, windows, var_info, method)
    _panel_clusters(axes[1], coords, cluster_result, anomaly_flags, windows, var_info, method)

    _plt.tight_layout(pad=1.5)
    return fig


# ── Panel renderers ───────────────────────────────────────────────────────────

def _panel_states(ax, coords, windows, var_info, method):
    ax.set_facecolor("#161B22")
    states = [w.dominant_state for w in windows]
    for state, color in _STATE_COLORS.items():
        mask = [i for i, s in enumerate(states) if s == state]
        if mask:
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, label=state, s=60, alpha=0.85,
                edgecolors="#0D1117", linewidths=0.5,
            )
    _style_ax(ax, f"Colored by dominant state", var_info, method)


def _panel_clusters(ax, coords, cluster_result, anomaly_flags, windows, var_info, method):
    ax.set_facecolor("#161B22")
    labels = cluster_result.labels
    stim_mask = [i for i, w in enumerate(windows) if w.active_stimulus_ids]

    for k in range(cluster_result.n_clusters):
        mask = np.where(labels == k)[0]
        col  = _CLUSTER_PALETTE[k % len(_CLUSTER_PALETTE)]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=col, label=f"Cluster {k}", s=55, alpha=0.80,
            edgecolors="#0D1117", linewidths=0.5,
        )

    # Overlay anomalies
    a_mask = np.where(anomaly_flags)[0]
    if len(a_mask):
        ax.scatter(
            coords[a_mask, 0], coords[a_mask, 1],
            marker="x", c="#E74C3C", s=120, linewidths=2.0,
            label=f"Anomaly ({len(a_mask)})", zorder=5,
        )

    # Mark stimulus windows with a ring
    if stim_mask:
        ax.scatter(
            coords[stim_mask, 0], coords[stim_mask, 1],
            s=120, facecolors="none", edgecolors="#ECF0F1",
            linewidths=1.2, label="Stimulus window", zorder=4,
        )

    sil_str = ""
    if cluster_result.silhouette_score is not None:
        sil_str = f"  sil={cluster_result.silhouette_score:.2f}"
    _style_ax(ax, f"K-Means (k={cluster_result.n_clusters}){sil_str}", var_info, method)


def _style_ax(ax, subtitle, var_info, method):
    x_label = var_info.get("x_label", f"{method} 1")
    y_label = var_info.get("y_label", f"{method} 2")
    ax.set_xlabel(x_label, color="#8B949E", fontsize=9)
    ax.set_ylabel(y_label, color="#8B949E", fontsize=9)
    ax.set_title(subtitle, color="#C9D1D9", fontsize=10, pad=8)
    ax.tick_params(colors="#484F58", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262D")
    legend = ax.legend(
        fontsize=8, loc="best",
        facecolor="#161B22", edgecolor="#21262D", labelcolor="#C9D1D9",
        markerscale=0.9,
    )


# ── Reduction helpers ─────────────────────────────────────────────────────────

def _reduce_pca(X: np.ndarray):
    from sklearn.decomposition import PCA
    n_components = min(2, X.shape[0], X.shape[1])
    pca    = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    if n_components < 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    evr = pca.explained_variance_ratio_
    var_info = {
        "x_label": f"PC1 ({evr[0]*100:.1f}% var)" if len(evr) > 0 else "PC1",
        "y_label": f"PC2 ({evr[1]*100:.1f}% var)" if len(evr) > 1 else "PC2",
    }
    return coords, "PCA", var_info


def _reduce_umap(X: np.ndarray):
    n_neighbors = min(15, len(X) - 1)
    reducer = _umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    coords  = reducer.fit_transform(X)
    var_info = {"x_label": "UMAP 1", "y_label": "UMAP 2"}
    return coords, "UMAP", var_info
