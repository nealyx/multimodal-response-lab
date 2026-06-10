"""Persist trained models and evaluation artifacts to disk."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.supervised.evaluator import EvalResult, plot_confusion_matrix, plot_feature_importance
from src.supervised.trainer import TrainResult


class ModelExporter:
    """Write model.pkl, metrics.json, confusion_matrix.png, feature_importance.csv."""

    @staticmethod
    def save_model(result: TrainResult, path: str) -> None:
        """Pickle the fitted estimator + metadata wrapper."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model":       result.model,
            "model_name":  result.model_name,
            "config":      result.config,
            "feature_names": result.train_dataset.feature_names,
            "label_names":   result.train_dataset.label_names,
            "task":          result.train_dataset.task,
        }
        with open(path, "wb") as fh:
            pickle.dump(artifact, fh, protocol=4)

    @staticmethod
    def load_model(path: str) -> Dict[str, Any]:
        """Load a pickled model artifact. Returns the metadata dict."""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def save_metrics(eval_result: EvalResult, path: str) -> None:
        """Write all evaluation metrics to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        def _safe(v):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        doc = {
            "model_name":     eval_result.model_name,
            "task":           eval_result.task,
            "train_accuracy": _safe(eval_result.train_accuracy),
            "test_accuracy":  _safe(eval_result.test_accuracy),
            "macro_f1":       _safe(eval_result.macro_f1),
            "weighted_f1":    _safe(eval_result.weighted_f1),
            "label_names":    eval_result.label_names,
            "per_class": [
                {
                    "label":     cm.label,
                    "precision": _safe(cm.precision),
                    "recall":    _safe(cm.recall),
                    "f1":        _safe(cm.f1),
                    "support":   cm.support,
                }
                for cm in eval_result.per_class
            ],
            "confusion_matrix": eval_result.confusion_matrix.tolist(),
            "top_features": [
                {"feature": name, "importance": _safe(imp)}
                for name, imp in eval_result.top_features
            ],
            "config":         eval_result.config,
            "warnings":       eval_result.warnings_list,
            "data_quality_note": (
                "Labels are heuristic (derived from the same signals as features). "
                "Metrics should be treated as exploratory, not as validated inference."
            ),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)

    @staticmethod
    def save_feature_importance_csv(eval_result: EvalResult, path: str) -> None:
        """Write feature importances to CSV, sorted by descending importance."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        feat_names   = eval_result.feature_names
        importances  = eval_result.feature_importances

        if importances is None:
            importances = np.zeros(len(feat_names))

        rows = sorted(
            zip(feat_names, importances.tolist()),
            key=lambda x: -x[1],
        )
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["rank", "feature", "importance"])
            for rank, (name, imp) in enumerate(rows, 1):
                writer.writerow([rank, name, round(float(imp), 6)])

    @staticmethod
    def save_confusion_matrix_png(eval_result: EvalResult, path: str) -> bool:
        """Write confusion_matrix.png. Returns True on success."""
        fig = plot_confusion_matrix(eval_result)
        if fig is None:
            return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        return True

    @staticmethod
    def save_feature_importance_png(eval_result: EvalResult, path: str) -> bool:
        """Write feature_importance.png. Returns True on success."""
        fig = plot_feature_importance(eval_result)
        if fig is None:
            return False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        return True

    @classmethod
    def save_all(
        cls,
        result:      TrainResult,
        eval_result: EvalResult,
        out_dir:     str,
    ) -> Dict[str, str]:
        """Write all artifacts to *out_dir*. Returns dict of {artifact: path}."""
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        model_path = str(base / "model.pkl")
        cls.save_model(result, model_path)
        paths["model"] = model_path

        metrics_path = str(base / "metrics.json")
        cls.save_metrics(eval_result, metrics_path)
        paths["metrics"] = metrics_path

        fi_csv = str(base / "feature_importance.csv")
        cls.save_feature_importance_csv(eval_result, fi_csv)
        paths["feature_importance_csv"] = fi_csv

        cm_png = str(base / "confusion_matrix.png")
        if cls.save_confusion_matrix_png(eval_result, cm_png):
            paths["confusion_matrix_png"] = cm_png

        fi_png = str(base / "feature_importance.png")
        if cls.save_feature_importance_png(eval_result, fi_png):
            paths["feature_importance_png"] = fi_png

        return paths
