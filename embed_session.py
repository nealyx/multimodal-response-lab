#!/usr/bin/env python3
"""Behavioral embedding and unsupervised pattern discovery — Day 10.

Loads a recorded session, extracts sliding-window feature vectors from the
behavioral time series, clusters them with K-Means, detects anomalous windows,
and writes a portable embedding artifact alongside a PCA scatter plot.

What a behavioral embedding is
-------------------------------
A 18-dimensional feature vector computed from a short time window (~2 s) of
behavioral signals: engagement score distribution, gaze on-screen fraction and
stability, head-pose orientation and stability, and blink/EAR/fatigue metrics.
All 18 features are normalized to [0, 1] so they can be directly compared and
clustered.

Why this is useful before supervised ML
-----------------------------------------
Supervised learning requires labeled data ("this window is high-engagement"),
which is expensive to collect.  Unsupervised clustering reveals naturally
occurring behavioral states first.  The clusters can then be *inspected*,
tentatively labeled, and used as weak supervision for the next iteration.

Usage
-----
    python embed_session.py outputs/sessions/<session_id>/
    python embed_session.py outputs/sessions/<session_id>/ --window-s 2.0 --stride-s 0.5
    python embed_session.py outputs/sessions/<session_id>/ --n-clusters 4 --anomaly isolation_forest

Output layout
-------------
    outputs/embeddings/<session_id>/
        behavioral_windows.csv    — per-window features + cluster + anomaly flag
        session_embedding.json    — session vector, per-stimulus vectors, cluster summary
        clusters.csv              — cluster centre feature vectors
        embedding_plot.png        — PCA scatter (state-colored + cluster-colored)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.embedding.anomaly import AnomalyDetector
from src.embedding.clustering import BehavioralClusterer
from src.embedding.encoder import SessionEncoder
from src.embedding.exporter import EmbeddingExporter
from src.embedding.features import FEATURE_NAMES, FeatureExtractor
from src.embedding.visualization import plot_embeddings
from src.reporting.loader import load_session_dir, load_session_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _resolve_session(path_str: str):
    p = Path(path_str)
    if p.is_dir():
        return load_session_dir(str(p))
    if p.is_file() and p.suffix == ".json":
        return load_session_json(str(p))
    raise ValueError(f"Expected a session directory or session_log.json, got: {path_str}")


def _print_summary(windows, cluster_result, anomaly_flags, session_id):
    print(f"\n── Embedding  [{session_id}] ───────────────────────────────────────")
    print(f"  Windows extracted     : {len(windows)}")

    # State distribution
    from collections import Counter
    state_counts = Counter(w.dominant_state for w in windows)
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / max(1, len(windows)) * 20)
        print(f"  {state:<12} {count:3d}  {bar}")

    print(f"\n  Clusters (k={cluster_result.n_clusters})")
    for k in range(cluster_result.n_clusters):
        count = int(np.sum(cluster_result.labels == k))
        # Describe cluster by top-3 highest-valued features
        center = cluster_result.cluster_centers[k]
        top3   = sorted(range(len(center)), key=lambda i: -center[i])[:3]
        desc   = " + ".join(FEATURE_NAMES[i].replace("_frac","").replace("_mean","") for i in top3)
        print(f"    Cluster {k}: {count:3d} windows  [{desc}]")

    sil = cluster_result.silhouette_score
    if sil is not None:
        print(f"  Silhouette score      : {sil:.3f}  "
              f"({'good separation' if sil > 0.4 else 'weak structure'})")

    n_anom = int(np.sum(anomaly_flags))
    print(f"  Anomalous windows     : {n_anom} / {len(windows)}")
    print("─────────────────────────────────────────────────────────────────────")


def run(args: argparse.Namespace) -> None:
    # ── Load session ──────────────────────────────────────────────────────
    log.info("Loading session from %s", args.input)
    session_log = _resolve_session(args.input)
    log.info(
        "Session %s  |  %d samples  |  %d trials",
        session_log.session_id,
        len(session_log.samples),
        len(session_log.stimulus_events),
    )

    if not session_log.samples:
        print("No behavioral samples found in this session.  "
              "Run run_stimulus_experiment.py first.")
        return

    # ── Extract windows ───────────────────────────────────────────────────
    extractor = FeatureExtractor(
        window_s=args.window_s,
        stride_s=args.stride_s,
        min_frames=5,
    )
    windows = extractor.extract(
        session_log.samples,
        session_id=session_log.session_id,
        start_ts=session_log.start_ts,
    )
    log.info("Extracted %d behavioral windows  (w=%.1fs, stride=%.1fs)",
             len(windows), args.window_s, args.stride_s)

    if not windows:
        print("Too few samples for the requested window size.  "
              f"Try --window-s smaller or record a longer session.")
        return

    # ── Session and stimulus embeddings ───────────────────────────────────
    encoder        = SessionEncoder()
    session_vector = encoder.encode_session(windows)
    per_stimulus   = encoder.encode_all_stimuli(windows, session_log.stimulus_events)
    log.info("Computed session embedding  (shape %s)", session_vector.shape)

    # ── Anomaly detection ─────────────────────────────────────────────────
    detector      = AnomalyDetector(method=args.anomaly, threshold=2.5)
    anomaly_flags = detector.fit_detect(windows)
    log.info("Anomaly detection (%s): %d anomalous windows",
             args.anomaly, int(np.sum(anomaly_flags)))

    # ── Clustering ────────────────────────────────────────────────────────
    clusterer      = BehavioralClusterer(n_clusters=args.n_clusters)
    cluster_result = clusterer.fit(windows)
    log.info(
        "K-Means (k=%d): inertia=%.3f  silhouette=%s",
        cluster_result.n_clusters,
        cluster_result.inertia,
        f"{cluster_result.silhouette_score:.3f}" if cluster_result.silhouette_score else "—",
    )

    _print_summary(windows, cluster_result, anomaly_flags, session_log.session_id)

    # ── Output directory ──────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / session_log.session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # behavioral_windows.csv
    EmbeddingExporter.windows_to_csv(
        windows, anomaly_flags, cluster_result.labels,
        str(out_dir / "behavioral_windows.csv"),
    )
    log.info("Windows CSV → %s", out_dir / "behavioral_windows.csv")

    # session_embedding.json
    EmbeddingExporter.session_embedding_to_json(
        session_id=      session_log.session_id,
        experiment_name= session_log.experiment_name,
        session_vector=  session_vector,
        cluster_result=  cluster_result,
        per_stimulus=    per_stimulus,
        n_windows=       len(windows),
        n_frames=        len(session_log.samples),
        window_s=        args.window_s,
        stride_s=        args.stride_s,
        anomaly_count=   int(np.sum(anomaly_flags)),
        generated_at=    datetime.now(timezone.utc).isoformat(),
        path=            str(out_dir / "session_embedding.json"),
    )
    log.info("Session embedding JSON → %s", out_dir / "session_embedding.json")

    # clusters.csv
    EmbeddingExporter.clusters_to_csv(
        cluster_result,
        str(out_dir / "clusters.csv"),
    )
    log.info("Clusters CSV → %s", out_dir / "clusters.csv")

    # embedding_plot.png
    if not args.no_plot:
        fig = plot_embeddings(
            windows, cluster_result, anomaly_flags,
            session_id=session_log.session_id,
            use_umap=args.umap,
        )
        if fig is not None:
            plot_path = out_dir / "embedding_plot.png"
            fig.savefig(str(plot_path), dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
            log.info("Embedding plot → %s", plot_path)

    print(f"\n  Output directory: {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral embedding and unsupervised clustering — Day 10"
    )
    parser.add_argument(
        "input",
        help="Session directory (with session_log.json + samples.csv) "
             "or path to session_log.json",
    )
    parser.add_argument(
        "--output-dir", default="outputs/embeddings",
        help="Root output directory (default: outputs/embeddings/)",
    )
    parser.add_argument(
        "--window-s", type=float, default=2.0,
        help="Sliding window duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--stride-s", type=float, default=0.5,
        help="Window stride in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=3,
        help="Number of K-Means clusters (default: 3)",
    )
    parser.add_argument(
        "--anomaly", choices=["z_score", "isolation_forest"],
        default="z_score",
        help="Anomaly detection method (default: z_score)",
    )
    parser.add_argument(
        "--umap", action="store_true",
        help="Use UMAP instead of PCA (requires umap-learn)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip embedding plot generation",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
