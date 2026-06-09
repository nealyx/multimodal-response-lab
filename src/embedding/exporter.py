"""Export behavioral embeddings, cluster results, and session vectors to disk."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.embedding.clustering import ClusterResult
from src.embedding.features import BehavioralWindow, FEATURE_NAMES, N_FEATURES


class EmbeddingExporter:
    """Write embedding artifacts to CSV and JSON files."""

    @staticmethod
    def windows_to_csv(
        windows:       List[BehavioralWindow],
        anomaly_flags: np.ndarray,
        cluster_labels: np.ndarray,
        path:          str,
    ) -> None:
        """Write per-window features + cluster/anomaly metadata to CSV.

        Columns: window_id, session_id, start_ts, end_ts, n_frames,
                 dominant_state, active_stimulus_ids, cluster_label, is_anomaly,
                 <feature_0>, ..., <feature_17>
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fields = (
            ["window_id", "session_id", "start_ts", "end_ts",
             "n_frames", "dominant_state", "active_stimulus_ids",
             "cluster_label", "is_anomaly"]
            + FEATURE_NAMES
        )
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for i, (w, is_anom, label) in enumerate(
                zip(windows, anomaly_flags, cluster_labels)
            ):
                row: Dict[str, Any] = {
                    "window_id":           w.window_id,
                    "session_id":          w.session_id,
                    "start_ts":            round(w.start_ts, 4),
                    "end_ts":              round(w.end_ts,   4),
                    "n_frames":            w.n_frames,
                    "dominant_state":      w.dominant_state,
                    "active_stimulus_ids": "|".join(w.active_stimulus_ids),
                    "cluster_label":       int(label),
                    "is_anomaly":          bool(is_anom),
                }
                for j, name in enumerate(FEATURE_NAMES):
                    row[name] = round(float(w.features[j]), 5)
                writer.writerow(row)

    @staticmethod
    def session_embedding_to_json(
        session_id:          str,
        experiment_name:     str,
        session_vector:      np.ndarray,
        cluster_result:      ClusterResult,
        per_stimulus:        Dict[str, np.ndarray],
        n_windows:           int,
        n_frames:            int,
        window_s:            float,
        stride_s:            float,
        anomaly_count:       int,
        generated_at:        str,
        path:                str,
    ) -> None:
        """Write session-level embedding and cluster summary to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        doc = {
            "session_id":        session_id,
            "experiment_name":   experiment_name,
            "generated_at":      generated_at,
            "window_size_s":     window_s,
            "stride_s":          stride_s,
            "n_windows":         n_windows,
            "n_frames":          n_frames,
            "n_features":        N_FEATURES,
            "feature_names":     FEATURE_NAMES,
            "session_vector":    [round(float(v), 5) for v in session_vector],
            "per_stimulus_embeddings": {
                k: [round(float(v), 5) for v in vec]
                for k, vec in per_stimulus.items()
            },
            "anomaly_count": anomaly_count,
            "cluster_info": {
                "n_clusters":       cluster_result.n_clusters,
                "inertia":          round(cluster_result.inertia, 4),
                "silhouette_score": (
                    round(cluster_result.silhouette_score, 4)
                    if cluster_result.silhouette_score is not None else None
                ),
                "cluster_centers":  [
                    [round(float(v), 5) for v in row]
                    for row in cluster_result.cluster_centers
                ],
                "window_counts": [
                    int(np.sum(cluster_result.labels == k))
                    for k in range(cluster_result.n_clusters)
                ],
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)

    @staticmethod
    def clusters_to_csv(
        cluster_result: ClusterResult,
        path:           str,
    ) -> None:
        """Write cluster centres and window counts to CSV.

        Rows: one per cluster.  Columns: cluster_id, n_windows,
              silhouette_score, <feature_0>, ..., <feature_17>
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fields = ["cluster_id", "n_windows", "silhouette_score"] + FEATURE_NAMES
        sil = (
            round(cluster_result.silhouette_score, 4)
            if cluster_result.silhouette_score is not None else ""
        )
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for k in range(cluster_result.n_clusters):
                count = int(np.sum(cluster_result.labels == k))
                row: Dict[str, Any] = {
                    "cluster_id":       k,
                    "n_windows":        count,
                    "silhouette_score": sil if k == 0 else "",  # write once
                }
                for j, name in enumerate(FEATURE_NAMES):
                    row[name] = round(float(cluster_result.cluster_centers[k, j]), 5)
                writer.writerow(row)
