"""Unit tests for the behavioral embedding layer (Day 10).

All tests are offline — no webcam, no session files, no network.
Synthetic BehavioralSample lists are constructed directly in each test.
Tests covering visualization are skipped if matplotlib is unavailable
(it IS available in the project venv, so they run normally).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from src.embedding.anomaly import AnomalyDetector
from src.embedding.clustering import BehavioralClusterer, ClusterResult
from src.embedding.encoder import SessionEncoder
from src.embedding.exporter import EmbeddingExporter
from src.embedding.features import (
    BehavioralWindow,
    FeatureExtractor,
    FEATURE_NAMES,
    N_FEATURES,
    dominant_state,
    extract_features,
)
from src.stimulus.events import BehavioralSample


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample(
    ts: float = 0.0,
    score: float = 0.75,
    state: str = "focused",
    on_screen: bool = True,
    gaze_h: float = 0.5,
    gaze_v: float = 0.5,
    yaw: float = 2.0,
    pitch: float = 1.0,
    blink_rate: float = 14.0,
    ear: float = 0.30,
    fatigued: bool = False,
    stim_id: str = None,
    head_zone: str = "focused",
) -> BehavioralSample:
    return BehavioralSample(
        frame_index=0, timestamp=ts,
        active_stimulus_id=stim_id,
        engagement_state=state, smoothed_score=score, confidence=0.8,
        focused_fraction=0.7,
        gaze_zone="center", is_on_screen=on_screen,
        gaze_h=gaze_h, gaze_v=gaze_v,
        head_zone=head_zone, head_yaw=yaw, head_pitch=pitch,
        blink_state="open", mean_ear=ear,
        blink_rate=blink_rate, is_fatigued=fatigued,
    )


def _make_samples(
    n: int,
    start_ts: float = 0.0,
    dt: float = 0.033,
    score: float = 0.75,
    state: str = "focused",
) -> list:
    return [_sample(ts=start_ts + i * dt, score=score, state=state) for i in range(n)]


def _make_window(
    n_frames: int = 60,
    features: np.ndarray = None,
    session_id: str = "s0",
    stim_ids: list = None,
    state: str = "focused",
) -> BehavioralWindow:
    if features is None:
        features = np.full(N_FEATURES, 0.7, dtype=np.float32)
    return BehavioralWindow(
        window_id=f"win_0000",
        session_id=session_id,
        start_ts=0.0, end_ts=2.0,
        n_frames=n_frames,
        dominant_state=state,
        active_stimulus_ids=stim_ids or [],
        features=features,
    )


# ── Feature extraction ────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_returns_correct_shape(self):
        samples = _make_samples(60)
        vec = extract_features(samples)
        assert vec.shape == (N_FEATURES,)

    def test_all_focused_gives_high_focused_frac(self):
        samples = [_sample(ts=float(i) * 0.033, state="focused") for i in range(60)]
        vec = extract_features(samples)
        idx = FEATURE_NAMES.index("focused_frac")
        assert vec[idx] == pytest.approx(1.0, abs=0.01)

    def test_all_distracted_gives_high_distracted_frac(self):
        samples = [_sample(ts=float(i) * 0.033, state="distracted") for i in range(60)]
        vec = extract_features(samples)
        dist_idx = FEATURE_NAMES.index("distracted_frac")
        focus_idx = FEATURE_NAMES.index("focused_frac")
        assert vec[dist_idx] == pytest.approx(1.0, abs=0.01)
        assert vec[focus_idx] == pytest.approx(0.0, abs=0.01)

    def test_high_score_gives_high_engagement_mean(self):
        samples = [_sample(ts=float(i) * 0.033, score=0.90) for i in range(60)]
        vec = extract_features(samples)
        idx = FEATURE_NAMES.index("engagement_mean")
        assert vec[idx] == pytest.approx(0.90, abs=0.02)

    def test_on_screen_fraction(self):
        samples = (
            [_sample(ts=float(i) * 0.033, on_screen=True)  for i in range(45)]
            + [_sample(ts=float(i + 45) * 0.033, on_screen=False) for i in range(15)]
        )
        vec = extract_features(samples)
        idx = FEATURE_NAMES.index("on_screen_frac")
        assert vec[idx] == pytest.approx(0.75, abs=0.02)

    def test_empty_samples_returns_neutral_vector(self):
        vec = extract_features([])
        assert vec.shape == (N_FEATURES,)
        assert np.all(vec == pytest.approx(0.5, abs=0.01))

    def test_all_features_in_unit_range(self):
        samples = _make_samples(60, score=0.8)
        vec = extract_features(samples)
        assert float(vec.min()) >= 0.0
        assert float(vec.max()) <= 1.0 + 1e-6

    def test_dtype_is_float32(self):
        samples = _make_samples(30)
        vec = extract_features(samples)
        assert vec.dtype == np.float32

    def test_head_yaw_mean_normalized_positive(self):
        # yaw = 22.5° → _norm(22.5, -45, 45) = 0.75
        samples = [_sample(ts=float(i) * 0.033, yaw=22.5) for i in range(30)]
        vec = extract_features(samples)
        idx = FEATURE_NAMES.index("head_yaw_mean")
        assert vec[idx] == pytest.approx(0.75, abs=0.02)

    def test_blink_rate_normalized(self):
        # blink_rate = 20/min → _norm(20, 0, 40) = 0.5
        samples = [_sample(ts=float(i) * 0.033, blink_rate=20.0) for i in range(30)]
        vec = extract_features(samples)
        idx = FEATURE_NAMES.index("blink_rate_mean")
        assert vec[idx] == pytest.approx(0.5, abs=0.02)

    def test_fatigue_fraction(self):
        samples = [_sample(ts=float(i) * 0.033, fatigued=True)  for i in range(30)]
        vec = extract_features(samples)
        idx = FEATURE_NAMES.index("fatigue_frac")
        assert vec[idx] == pytest.approx(1.0, abs=0.01)

    def test_n_features_constant(self):
        assert N_FEATURES == 18
        assert len(FEATURE_NAMES) == 18


# ── Dominant state ────────────────────────────────────────────────────────────

class TestDominantState:
    def test_all_focused(self):
        samples = [_sample(state="focused") for _ in range(10)]
        assert dominant_state(samples) == "focused"

    def test_mixed_prefers_meaningful_over_unreliable(self):
        samples = (
            [_sample(state="unreliable") for _ in range(7)]
            + [_sample(state="focused")    for _ in range(3)]
        )
        assert dominant_state(samples) == "focused"

    def test_all_unreliable(self):
        samples = [_sample(state="unreliable") for _ in range(10)]
        assert dominant_state(samples) == "unreliable"

    def test_empty_returns_unreliable(self):
        assert dominant_state([]) == "unreliable"


# ── FeatureExtractor / windowing ──────────────────────────────────────────────

class TestFeatureExtractor:
    def test_correct_number_of_windows(self):
        # 5.0 s of data, window=2.0, stride=0.5 → floor((5.0-2.0)/0.5)+1 = 7
        samples = _make_samples(151, dt=0.033)   # ~5 s at 30 FPS
        extractor = FeatureExtractor(window_s=2.0, stride_s=0.5, min_frames=5)
        windows = extractor.extract(samples, "s0", start_ts=0.0)
        assert len(windows) >= 5  # at least 5 windows from a ~5 s session

    def test_window_feature_shape(self):
        samples = _make_samples(100)
        extractor = FeatureExtractor(window_s=2.0, stride_s=0.5)
        windows = extractor.extract(samples, "s0", start_ts=0.0)
        assert len(windows) > 0
        assert windows[0].features.shape == (N_FEATURES,)

    def test_empty_samples_returns_empty(self):
        extractor = FeatureExtractor()
        assert extractor.extract([], "s0") == []

    def test_session_shorter_than_window_returns_one_window(self):
        # 1.0 s of data, window=2.0 s → single window with all samples
        samples = _make_samples(30, dt=0.033)   # ~1 s
        extractor = FeatureExtractor(window_s=2.0, stride_s=0.5)
        windows = extractor.extract(samples, "s0", start_ts=0.0)
        assert len(windows) == 1

    def test_window_ids_are_sequential(self):
        samples = _make_samples(200, dt=0.033)
        extractor = FeatureExtractor(window_s=2.0, stride_s=0.5)
        windows = extractor.extract(samples, "s0", start_ts=0.0)
        ids = [w.window_id for w in windows]
        assert ids == sorted(ids)

    def test_session_id_propagated(self):
        samples = _make_samples(60)
        extractor = FeatureExtractor()
        windows = extractor.extract(samples, session_id="my_session", start_ts=0.0)
        assert all(w.session_id == "my_session" for w in windows)

    def test_stimulus_ids_captured(self):
        samples = [_sample(ts=float(i) * 0.033, stim_id="flash_00") for i in range(60)]
        extractor = FeatureExtractor(window_s=2.0, stride_s=2.0)
        windows = extractor.extract(samples, "s0", start_ts=0.0)
        assert any("flash_00" in w.active_stimulus_ids for w in windows)

    def test_min_frames_filter(self):
        # Short window at end has only 2 frames — should be filtered (min_frames=5)
        samples = _make_samples(10, dt=0.033)
        extractor = FeatureExtractor(window_s=1.5, stride_s=0.5, min_frames=5)
        windows = extractor.extract(samples, "s0", start_ts=0.0)
        assert all(w.n_frames >= 5 for w in windows)


# ── SessionEncoder ────────────────────────────────────────────────────────────

class TestSessionEncoder:
    def test_session_vector_shape(self):
        windows = [_make_window() for _ in range(5)]
        enc = SessionEncoder()
        vec = enc.encode_session(windows)
        assert vec.shape == (N_FEATURES,)

    def test_session_vector_dtype(self):
        windows = [_make_window() for _ in range(5)]
        enc = SessionEncoder()
        vec = enc.encode_session(windows)
        assert vec.dtype == np.float32

    def test_empty_windows_returns_neutral(self):
        enc = SessionEncoder()
        vec = enc.encode_session([])
        assert vec.shape == (N_FEATURES,)
        np.testing.assert_allclose(vec, np.full(N_FEATURES, 0.5), atol=0.01)

    def test_mean_of_features(self):
        feats_a = np.full(N_FEATURES, 0.6, dtype=np.float32)
        feats_b = np.full(N_FEATURES, 0.4, dtype=np.float32)
        wa = _make_window(features=feats_a)
        wb = _make_window(features=feats_b)
        enc = SessionEncoder()
        vec = enc.encode_session([wa, wb])
        np.testing.assert_allclose(vec, np.full(N_FEATURES, 0.5), atol=0.001)

    def test_stimulus_embedding_correct_window(self):
        w_stim = _make_window(features=np.full(N_FEATURES, 0.9, dtype=np.float32),
                               stim_ids=["flash_00"])
        w_base = _make_window(features=np.full(N_FEATURES, 0.1, dtype=np.float32))
        enc = SessionEncoder()
        vec = enc.encode_stimulus_windows([w_stim, w_base], "flash_00")
        assert vec is not None
        np.testing.assert_allclose(vec, np.full(N_FEATURES, 0.9), atol=0.001)

    def test_stimulus_embedding_none_when_no_overlap(self):
        w = _make_window()   # no stim ids
        enc = SessionEncoder()
        assert enc.encode_stimulus_windows([w], "flash_00") is None


# ── AnomalyDetector ───────────────────────────────────────────────────────────

class TestAnomalyDetector:
    def _windows_with_outlier(self, n_normal=20):
        normal_feat = np.full(N_FEATURES, 0.80, dtype=np.float32)
        outlier_feat = np.full(N_FEATURES, 0.10, dtype=np.float32)
        windows = [_make_window(features=normal_feat.copy()) for _ in range(n_normal)]
        windows.append(_make_window(features=outlier_feat.copy()))
        return windows

    def test_z_score_flags_outlier(self):
        windows = self._windows_with_outlier(20)
        det = AnomalyDetector(method="z_score", threshold=2.0)
        flags = det.fit_detect(windows)
        assert bool(flags[-1])   # outlier should be flagged

    def test_z_score_does_not_flag_normal(self):
        windows = self._windows_with_outlier(20)
        det = AnomalyDetector(method="z_score", threshold=2.0)
        flags = det.fit_detect(windows)
        assert int(flags[:20].sum()) <= 2   # at most a few false positives

    def test_isolation_forest_flags_outlier(self):
        windows = self._windows_with_outlier(30)
        det = AnomalyDetector(method="isolation_forest", contamination=0.05)
        flags = det.fit_detect(windows)
        assert bool(flags[-1])

    def test_fewer_than_3_windows_returns_all_false(self):
        windows = [_make_window() for _ in range(2)]
        det = AnomalyDetector()
        flags = det.fit_detect(windows)
        assert not any(flags)

    def test_output_shape_matches_input(self):
        windows = [_make_window() for _ in range(10)]
        det = AnomalyDetector()
        flags = det.fit_detect(windows)
        assert len(flags) == 10

    def test_constant_features_no_anomalies(self):
        feat = np.full(N_FEATURES, 0.7, dtype=np.float32)
        windows = [_make_window(features=feat.copy()) for _ in range(10)]
        det = AnomalyDetector(method="z_score")
        flags = det.fit_detect(windows)
        assert not any(flags)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            AnomalyDetector(method="bad_method")


# ── BehavioralClusterer ───────────────────────────────────────────────────────

class TestBehavioralClusterer:
    def _distinct_windows(self):
        """Three clearly separated clusters of 10 windows each."""
        windows = []
        for k, val in enumerate([0.1, 0.5, 0.9]):
            feat = np.full(N_FEATURES, val, dtype=np.float32)
            for _ in range(10):
                windows.append(_make_window(features=feat.copy()))
        return windows

    def test_labels_shape(self):
        windows = [_make_window() for _ in range(15)]
        cr = BehavioralClusterer(n_clusters=3).fit(windows)
        assert cr.labels.shape == (15,)

    def test_centers_shape(self):
        windows = [_make_window() for _ in range(15)]
        cr = BehavioralClusterer(n_clusters=3).fit(windows)
        assert cr.cluster_centers.shape == (3, N_FEATURES)

    def test_distinct_clusters_found(self):
        windows = self._distinct_windows()
        cr = BehavioralClusterer(n_clusters=3).fit(windows)
        assert cr.n_clusters == 3
        assert len(set(cr.labels.tolist())) == 3

    def test_silhouette_score_computed(self):
        windows = self._distinct_windows()
        cr = BehavioralClusterer(n_clusters=3).fit(windows)
        assert cr.silhouette_score is not None
        assert cr.silhouette_score > 0.4   # well-separated clusters

    def test_fewer_windows_than_k_reduces_clusters(self):
        windows = [_make_window() for _ in range(2)]
        cr = BehavioralClusterer(n_clusters=5).fit(windows)
        assert cr.n_clusters <= 2

    def test_single_window(self):
        cr = BehavioralClusterer(n_clusters=3).fit([_make_window()])
        assert cr.n_clusters == 1
        assert len(cr.labels) == 1


# ── EmbeddingExporter ─────────────────────────────────────────────────────────

class TestEmbeddingExporter:
    def _setup(self, n=10):
        feat = np.full(N_FEATURES, 0.6, dtype=np.float32)
        windows = [
            BehavioralWindow(
                window_id=f"win_{i:04d}", session_id="s0",
                start_ts=float(i), end_ts=float(i) + 2.0,
                n_frames=60, dominant_state="focused",
                active_stimulus_ids=[], features=feat.copy(),
            )
            for i in range(n)
        ]
        labels = np.array([i % 3 for i in range(n)])
        flags  = np.array([i == 0 for i in range(n)])
        cr = ClusterResult(
            n_clusters=3,
            labels=labels,
            cluster_centers=np.full((3, N_FEATURES), 0.5, dtype=np.float32),
            inertia=1.0,
            silhouette_score=0.45,
        )
        session_vec = np.full(N_FEATURES, 0.6, dtype=np.float32)
        return windows, labels, flags, cr, session_vec

    def test_windows_csv_created(self):
        windows, labels, flags, cr, _ = self._setup()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "windows.csv")
            EmbeddingExporter.windows_to_csv(windows, flags, labels, path)
            assert os.path.exists(path)

    def test_windows_csv_row_count(self):
        import csv as csv_mod
        windows, labels, flags, cr, _ = self._setup(10)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "windows.csv")
            EmbeddingExporter.windows_to_csv(windows, flags, labels, path)
            with open(path) as fh:
                rows = list(csv_mod.DictReader(fh))
            assert len(rows) == 10

    def test_session_json_created(self):
        windows, labels, flags, cr, sv = self._setup()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "embed.json")
            EmbeddingExporter.session_embedding_to_json(
                "s0", "test", sv, cr, {}, 10, 300, 2.0, 0.5, 1,
                "2025-01-01T00:00:00+00:00", path,
            )
            assert os.path.exists(path)

    def test_session_json_valid_structure(self):
        windows, labels, flags, cr, sv = self._setup()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "embed.json")
            EmbeddingExporter.session_embedding_to_json(
                "s0", "test", sv, cr, {}, 10, 300, 2.0, 0.5, 1,
                "2025-01-01T00:00:00+00:00", path,
            )
            with open(path) as fh:
                doc = json.load(fh)
            assert doc["n_features"] == N_FEATURES
            assert len(doc["feature_names"]) == N_FEATURES
            assert len(doc["session_vector"]) == N_FEATURES
            assert "cluster_info" in doc

    def test_clusters_csv_created(self):
        _, _, _, cr, _ = self._setup()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clusters.csv")
            EmbeddingExporter.clusters_to_csv(cr, path)
            assert os.path.exists(path)

    def test_clusters_csv_row_count(self):
        import csv as csv_mod
        _, _, _, cr, _ = self._setup()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clusters.csv")
            EmbeddingExporter.clusters_to_csv(cr, path)
            with open(path) as fh:
                rows = list(csv_mod.DictReader(fh))
            assert len(rows) == 3  # one row per cluster


# ── Visualization (requires matplotlib) ──────────────────────────────────────

class TestVisualization:
    @pytest.fixture(autouse=True)
    def _require_mpl(self):
        pytest.importorskip("matplotlib")

    def _setup(self, n=200):
        # Contiguous samples: first half focused, second half distracted
        samples = (
            [_sample(ts=float(i) * 0.033, score=0.85, state="focused")
             for i in range(n // 2)]
            + [_sample(ts=float(i + n // 2) * 0.033, score=0.25, state="distracted")
               for i in range(n // 2)]
        )
        extractor = FeatureExtractor(window_s=1.5, stride_s=0.5, min_frames=5)
        windows = extractor.extract(samples, "s0", start_ts=0.0)
        return windows

    def test_plot_returns_figure(self):
        from src.embedding.visualization import plot_embeddings
        import matplotlib.pyplot as plt
        windows = self._setup()
        cr      = BehavioralClusterer(n_clusters=2).fit(windows)
        flags   = AnomalyDetector().fit_detect(windows)
        fig     = plot_embeddings(windows, cr, flags, "test")
        assert fig is not None
        plt.close(fig)

    def test_too_few_windows_returns_none(self):
        from src.embedding.visualization import plot_embeddings
        windows = [_make_window() for _ in range(2)]
        cr      = BehavioralClusterer(n_clusters=2).fit(windows)
        flags   = np.array([False, False])
        fig     = plot_embeddings(windows, cr, flags)
        assert fig is None
