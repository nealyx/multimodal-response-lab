"""Tests for Day 11: supervised behavioral prediction layer.

Coverage:
  TestLabelCreation        (12 tests) — assign_labels, each task, edge cases
  TestDatasetBuilder       ( 8 tests) — build_dataset, class counts, empty input
  TestCSVLoader            ( 5 tests) — load_windows_csv from temp file
  TestTrainTestSplit       ( 6 tests) — random + temporal strategies
  TestTrainer              ( 8 tests) — TrainResult shape, configs, small-n warns
  TestEvaluator            ( 7 tests) — EvalResult fields, feature importances
  TestExporter             ( 6 tests) — file creation, metrics.json structure
  Total                    (52 tests)
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.embedding.features import BehavioralWindow, FEATURE_NAMES, N_FEATURES
from src.supervised.dataset import (
    BehavioralDataset,
    build_dataset,
    load_windows_csv,
    train_test_split_dataset,
)
from src.supervised.evaluator import EvalResult, evaluate
from src.supervised.exporter import ModelExporter
from src.supervised.labels import (
    BUILTIN_TASKS,
    LabelConfig,
    assign_labels,
    attention_level_config,
    engagement_state_config,
    fatigue_config,
    focused_vs_distracted_config,
)
from src.supervised.trainer import BehaviorModelTrainer, TrainResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_window(
    dominant_state: str = "focused",
    engagement_mean: float = 0.8,
    focused_frac: float = 0.9,
    fatigue_frac: float = 0.05,
    unreliable_frac: float = 0.0,
    window_id: str = "win_0000",
    session_id: str = "sess_test",
) -> BehavioralWindow:
    feat = np.zeros(N_FEATURES, dtype=np.float32)
    idx  = {name: i for i, name in enumerate(FEATURE_NAMES)}
    feat[idx["engagement_mean"]]  = engagement_mean
    feat[idx["focused_frac"]]     = focused_frac
    feat[idx["fatigue_frac"]]     = fatigue_frac
    feat[idx["unreliable_frac"]]  = unreliable_frac
    feat[idx["distracted_frac"]]  = 1.0 - focused_frac - fatigue_frac - unreliable_frac
    feat[idx["on_screen_frac"]]   = 0.9
    feat[idx["head_focused_frac"]]= 0.8
    feat[idx["ear_mean"]]         = 0.3
    return BehavioralWindow(
        window_id=window_id,
        session_id=session_id,
        start_ts=0.0,
        end_ts=2.0,
        n_frames=60,
        dominant_state=dominant_state,
        active_stimulus_ids=[],
        features=feat,
    )


def _make_windows(n: int, states=None, eng_values=None) -> List[BehavioralWindow]:
    if states is None:
        states = ["focused"] * (n // 2) + ["distracted"] * (n - n // 2)
    if eng_values is None:
        eng_values = [0.85 if s == "focused" else 0.25 for s in states]
    return [
        _make_window(
            dominant_state=states[i],
            engagement_mean=eng_values[i],
            focused_frac=0.9 if states[i] == "focused" else 0.05,
            window_id=f"win_{i:04d}",
        )
        for i in range(n)
    ]


def _write_windows_csv(windows: List[BehavioralWindow], path: str) -> None:
    fields = (
        ["window_id", "session_id", "start_ts", "end_ts",
         "n_frames", "dominant_state", "active_stimulus_ids",
         "cluster_label", "is_anomaly"]
        + FEATURE_NAMES
    )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for w in windows:
            row = {
                "window_id": w.window_id,
                "session_id": w.session_id,
                "start_ts": w.start_ts,
                "end_ts": w.end_ts,
                "n_frames": w.n_frames,
                "dominant_state": w.dominant_state,
                "active_stimulus_ids": "",
                "cluster_label": 0,
                "is_anomaly": False,
            }
            for i, name in enumerate(FEATURE_NAMES):
                row[name] = float(w.features[i])
            writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# TestLabelCreation
# ═══════════════════════════════════════════════════════════════════════════════

class TestLabelCreation:
    def test_attention_level_high_label(self):
        w = _make_window(engagement_mean=0.8)
        cfg = attention_level_config(high_threshold=0.65, low_threshold=0.40)
        kept, labels, names, excluded = assign_labels([w], cfg)
        assert len(kept) == 1
        assert labels[0] == 1  # high_attention

    def test_attention_level_low_label(self):
        w = _make_window(engagement_mean=0.2)
        cfg = attention_level_config(high_threshold=0.65, low_threshold=0.40)
        kept, labels, names, excluded = assign_labels([w], cfg)
        assert len(kept) == 1
        assert labels[0] == 0  # low_attention

    def test_attention_level_middle_excluded(self):
        w = _make_window(engagement_mean=0.5)
        cfg = attention_level_config(high_threshold=0.65, low_threshold=0.40)
        kept, labels, names, excluded = assign_labels([w], cfg)
        assert len(kept) == 0
        assert excluded == 1

    def test_focused_vs_distracted_focused(self):
        w = _make_window(dominant_state="focused")
        cfg = focused_vs_distracted_config()
        kept, labels, names, excluded = assign_labels([w], cfg)
        assert labels[0] == 1  # focused

    def test_focused_vs_distracted_distracted(self):
        w = _make_window(dominant_state="distracted")
        cfg = focused_vs_distracted_config()
        kept, labels, names, excluded = assign_labels([w], cfg)
        assert labels[0] == 0  # distracted

    def test_focused_vs_distracted_drifting_excluded(self):
        w = _make_window(dominant_state="drifting")
        cfg = focused_vs_distracted_config()
        kept, labels, names, excluded = assign_labels([w], cfg)
        assert len(kept) == 0

    def test_fatigue_above_threshold(self):
        w = _make_window(fatigue_frac=0.5)
        cfg = fatigue_config(fatigue_threshold=0.30)
        kept, labels, _, _ = assign_labels([w], cfg)
        assert labels[0] == 1  # fatigued

    def test_fatigue_below_threshold(self):
        w = _make_window(fatigue_frac=0.1)
        cfg = fatigue_config(fatigue_threshold=0.30)
        kept, labels, _, _ = assign_labels([w], cfg)
        assert labels[0] == 0  # alert

    def test_engagement_state_multiclass(self):
        windows = [
            _make_window(dominant_state="focused",    window_id="w0"),
            _make_window(dominant_state="distracted", window_id="w1"),
            _make_window(dominant_state="fatigued",   window_id="w2"),
            _make_window(dominant_state="drifting",   window_id="w3"),
        ]
        cfg = engagement_state_config()
        kept, labels, names, excluded = assign_labels(windows, cfg)
        assert len(kept) == 4
        assert labels[0] == names.index("focused")
        assert labels[1] == names.index("distracted")
        assert labels[2] == names.index("fatigued")
        assert labels[3] == names.index("drifting")

    def test_unreliable_window_excluded_by_quality_gate(self):
        w = _make_window(unreliable_frac=0.6)
        cfg = attention_level_config()
        kept, labels, _, excluded = assign_labels([w], cfg)
        assert excluded == 1
        assert len(kept) == 0

    def test_all_builtin_tasks_exist(self):
        assert set(BUILTIN_TASKS.keys()) == {
            "high_vs_low_attention",
            "focused_vs_distracted",
            "fatigue_detection",
            "engagement_state",
        }

    def test_label_names_match_config_classes(self):
        cfg = attention_level_config()
        w = _make_window(engagement_mean=0.9)
        _, _, names, _ = assign_labels([w], cfg)
        assert names == cfg.classes


# ═══════════════════════════════════════════════════════════════════════════════
# TestDatasetBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetBuilder:
    def test_dataset_shape(self):
        windows = _make_windows(20)
        cfg     = focused_vs_distracted_config()
        ds      = build_dataset(windows, cfg)
        assert ds.X.shape == (ds.n_samples, N_FEATURES)
        assert ds.y.shape == (ds.n_samples,)

    def test_dataset_feature_names(self):
        windows = _make_windows(10)
        ds = build_dataset(windows, attention_level_config())
        assert ds.feature_names == list(FEATURE_NAMES)

    def test_class_counts_sum_to_n_samples(self):
        windows = _make_windows(20)
        ds = build_dataset(windows, focused_vs_distracted_config())
        assert sum(ds.class_counts.values()) == ds.n_samples

    def test_empty_windows_returns_empty_dataset(self):
        ds = build_dataset([], attention_level_config())
        assert ds.n_samples == 0
        assert ds.X.shape == (0, N_FEATURES)

    def test_task_name_preserved(self):
        ds = build_dataset(_make_windows(10), attention_level_config())
        assert ds.task == "high_vs_low_attention"

    def test_session_ids_and_window_ids_length(self):
        windows = _make_windows(10)
        ds = build_dataset(windows, focused_vs_distracted_config())
        assert len(ds.session_ids) == ds.n_samples
        assert len(ds.window_ids)  == ds.n_samples

    def test_n_excluded_tracked(self):
        # All middle-band windows should be excluded
        windows = [_make_window(engagement_mean=0.52, window_id=f"w{i}") for i in range(5)]
        ds = build_dataset(windows, attention_level_config())
        assert ds.n_excluded == 5

    def test_x_dtype_float32(self):
        windows = _make_windows(10)
        ds = build_dataset(windows, focused_vs_distracted_config())
        assert ds.X.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# TestCSVLoader
# ═══════════════════════════════════════════════════════════════════════════════

class TestCSVLoader:
    def test_load_basic(self):
        windows = _make_windows(5)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp = f.name
        try:
            _write_windows_csv(windows, tmp)
            loaded = load_windows_csv(tmp)
            assert len(loaded) == 5
        finally:
            os.unlink(tmp)

    def test_feature_values_preserved(self):
        w = _make_window(engagement_mean=0.77)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp = f.name
        try:
            _write_windows_csv([w], tmp)
            loaded = load_windows_csv(tmp)
            idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
            assert abs(float(loaded[0].features[idx["engagement_mean"]]) - 0.77) < 1e-4
        finally:
            os.unlink(tmp)

    def test_dominant_state_preserved(self):
        w = _make_window(dominant_state="fatigued")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp = f.name
        try:
            _write_windows_csv([w], tmp)
            loaded = load_windows_csv(tmp)
            assert loaded[0].dominant_state == "fatigued"
        finally:
            os.unlink(tmp)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_windows_csv("/nonexistent/path/behavioral_windows.csv")

    def test_feature_vector_shape(self):
        windows = _make_windows(3)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp = f.name
        try:
            _write_windows_csv(windows, tmp)
            loaded = load_windows_csv(tmp)
            assert loaded[0].features.shape == (N_FEATURES,)
        finally:
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════════════════
# TestTrainTestSplit
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainTestSplit:
    def _make_ds(self, n=50):
        windows = _make_windows(n)
        return build_dataset(windows, focused_vs_distracted_config())

    def test_random_split_sizes(self):
        ds = self._make_ds(50)
        train, test = train_test_split_dataset(ds, test_size=0.20, strategy="random")
        assert abs(test.n_samples - 10) <= 2  # 20% of 50 ≈ 10

    def test_temporal_split_order(self):
        ds = self._make_ds(50)
        train, test = train_test_split_dataset(ds, test_size=0.20, strategy="temporal")
        assert train.n_samples + test.n_samples == ds.n_samples

    def test_split_no_overlap(self):
        ds = self._make_ds(50)
        train, test = train_test_split_dataset(ds, test_size=0.20, strategy="random", random_state=42)
        train_ids = set(train.window_ids)
        test_ids  = set(test.window_ids)
        assert train_ids.isdisjoint(test_ids)

    def test_too_small_dataset_returns_full(self):
        ds = self._make_ds(3)
        train, test = train_test_split_dataset(ds, test_size=0.20)
        assert train.n_samples == ds.n_samples
        assert test.n_samples  == ds.n_samples

    def test_feature_matrix_preserved_in_split(self):
        ds = self._make_ds(20)
        train, test = train_test_split_dataset(ds, test_size=0.20, strategy="temporal")
        assert train.X.shape[1] == N_FEATURES
        assert test.X.shape[1]  == N_FEATURES

    def test_label_names_preserved_in_split(self):
        ds = self._make_ds(20)
        train, test = train_test_split_dataset(ds, test_size=0.20)
        assert train.label_names == ds.label_names
        assert test.label_names  == ds.label_names


# ═══════════════════════════════════════════════════════════════════════════════
# TestTrainer
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainer:
    def _make_dataset(self, n=60):
        windows = _make_windows(n)
        return build_dataset(windows, focused_vs_distracted_config())

    def test_train_all_returns_three_results(self):
        ds = self._make_dataset()
        trainer = BehaviorModelTrainer(random_state=42)
        results = trainer.train_all(ds)
        assert len(results) == 3

    def test_model_names_in_results(self):
        ds = self._make_dataset()
        trainer = BehaviorModelTrainer(random_state=42)
        results = trainer.train_all(ds)
        names = {r.model_name for r in results}
        assert "logistic_regression" in names
        assert "random_forest"       in names
        assert "gradient_boosting"   in names

    def test_single_model_selection(self):
        ds = self._make_dataset()
        trainer = BehaviorModelTrainer(model_names=["random_forest"], random_state=42)
        results = trainer.train_all(ds)
        assert len(results) == 1
        assert results[0].model_name == "random_forest"

    def test_train_score_between_0_and_1(self):
        ds = self._make_dataset()
        trainer = BehaviorModelTrainer(random_state=42)
        for result in trainer.train_all(ds):
            assert 0.0 <= result.train_score <= 1.0

    def test_config_stored_in_result(self):
        ds = self._make_dataset()
        trainer = BehaviorModelTrainer(random_state=99)
        result = trainer.train_all(ds)[0]
        assert result.config["random_state"] == 99
        assert result.config["task"] == "focused_vs_distracted"

    def test_small_n_warning_emitted(self):
        windows = _make_windows(8)
        ds = build_dataset(windows, focused_vs_distracted_config())
        trainer = BehaviorModelTrainer(random_state=42)
        results = trainer.train_all(ds)
        # At least one model should warn about small n
        all_warns = [w for r in results for w in r.warnings_list]
        assert any("training sample" in w.lower() or "test sample" in w.lower()
                   for w in all_warns)

    def test_model_has_predict(self):
        ds = self._make_dataset()
        trainer = BehaviorModelTrainer(model_names=["random_forest"], random_state=42)
        result = trainer.train_all(ds)[0]
        preds = result.model.predict(result.test_dataset.X)
        assert preds.shape == (result.test_dataset.n_samples,)

    def test_reproducibility(self):
        ds = self._make_dataset()
        r1 = BehaviorModelTrainer(model_names=["random_forest"], random_state=42).train_all(ds)[0]
        r2 = BehaviorModelTrainer(model_names=["random_forest"], random_state=42).train_all(ds)[0]
        assert r1.train_score == r2.train_score


# ═══════════════════════════════════════════════════════════════════════════════
# TestEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluator:
    def _trained_result(self, n=60):
        windows = _make_windows(n)
        ds = build_dataset(windows, focused_vs_distracted_config())
        trainer = BehaviorModelTrainer(model_names=["random_forest"], random_state=42)
        return trainer.train_all(ds)[0]

    def test_eval_result_fields(self):
        result = self._trained_result()
        er = evaluate(result)
        assert isinstance(er, EvalResult)
        assert er.model_name == "random_forest"
        assert er.task == "focused_vs_distracted"

    def test_confusion_matrix_shape(self):
        result = self._trained_result()
        er = evaluate(result)
        n = len(er.label_names)
        assert er.confusion_matrix.shape == (n, n)

    def test_per_class_metrics_count(self):
        result = self._trained_result()
        er = evaluate(result)
        assert len(er.per_class) == len(er.label_names)

    def test_feature_importances_shape(self):
        result = self._trained_result()
        er = evaluate(result)
        assert er.feature_importances is not None
        assert er.feature_importances.shape == (N_FEATURES,)

    def test_top_features_nonempty(self):
        result = self._trained_result()
        er = evaluate(result)
        assert len(er.top_features) > 0
        # Each entry is (name, importance)
        assert isinstance(er.top_features[0][0], str)
        assert isinstance(er.top_features[0][1], float)

    def test_logistic_regression_coef_importance(self):
        windows = _make_windows(60)
        ds = build_dataset(windows, focused_vs_distracted_config())
        trainer = BehaviorModelTrainer(model_names=["logistic_regression"], random_state=42)
        result = trainer.train_all(ds)[0]
        er = evaluate(result)
        assert er.feature_importances is not None

    def test_accuracies_in_range(self):
        result = self._trained_result()
        er = evaluate(result)
        for acc in [er.train_accuracy, er.test_accuracy]:
            if not np.isnan(acc):
                assert 0.0 <= acc <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestExporter
# ═══════════════════════════════════════════════════════════════════════════════

class TestExporter:
    def _train_and_eval(self, n=60):
        windows = _make_windows(n)
        ds = build_dataset(windows, focused_vs_distracted_config())
        trainer = BehaviorModelTrainer(model_names=["random_forest"], random_state=42)
        result = trainer.train_all(ds)[0]
        return result, evaluate(result)

    def test_save_all_creates_files(self):
        result, er = self._train_and_eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ModelExporter.save_all(result, er, tmpdir)
            assert "model"   in paths
            assert "metrics" in paths
            assert "feature_importance_csv" in paths
            assert Path(paths["model"]).exists()
            assert Path(paths["metrics"]).exists()

    def test_metrics_json_structure(self):
        result, er = self._train_and_eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = str(Path(tmpdir) / "metrics.json")
            ModelExporter.save_metrics(er, metrics_path)
            with open(metrics_path) as f:
                doc = json.load(f)
            assert "model_name"     in doc
            assert "task"           in doc
            assert "test_accuracy"  in doc
            assert "per_class"      in doc
            assert "confusion_matrix" in doc
            assert "data_quality_note" in doc

    def test_model_pkl_loadable(self):
        result, er = self._train_and_eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "model.pkl")
            ModelExporter.save_model(result, model_path)
            artifact = ModelExporter.load_model(model_path)
            assert "model"        in artifact
            assert "feature_names" in artifact
            assert "label_names"  in artifact

    def test_feature_importance_csv_row_count(self):
        result, er = self._train_and_eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            fi_path = str(Path(tmpdir) / "feature_importance.csv")
            ModelExporter.save_feature_importance_csv(er, fi_path)
            with open(fi_path) as f:
                rows = list(csv.reader(f))
            # header + N_FEATURES rows
            assert len(rows) == N_FEATURES + 1

    def test_confusion_matrix_png_created(self):
        result, er = self._train_and_eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = str(Path(tmpdir) / "confusion_matrix.png")
            ok = ModelExporter.save_confusion_matrix_png(er, cm_path)
            assert ok
            assert Path(cm_path).exists()

    def test_feature_importance_png_created(self):
        result, er = self._train_and_eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            fi_path = str(Path(tmpdir) / "feature_importance.png")
            ok = ModelExporter.save_feature_importance_png(er, fi_path)
            assert ok
            assert Path(fi_path).exists()
