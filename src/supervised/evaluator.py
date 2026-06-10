"""Evaluation metrics, confusion matrix, and feature importance for trained models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.supervised.trainer import TrainResult


@dataclass
class ClassMetrics:
    """Per-class precision / recall / F1."""
    label:     str
    precision: float
    recall:    float
    f1:        float
    support:   int


@dataclass
class EvalResult:
    """Full evaluation report for one trained model.

    Attributes
    ----------
    model_name : str
    task : str
    train_accuracy, test_accuracy : float
    per_class : List[ClassMetrics]
    macro_f1, weighted_f1 : float
    confusion_matrix : np.ndarray, shape (n_classes, n_classes)
    label_names : List[str]
    feature_importances : Optional[np.ndarray]   shape (n_features,) or None
    feature_names : List[str]
    top_features : List[tuple]   [(feature_name, importance), ...]
    config : dict
    warnings_list : List[str]
    """
    model_name:          str
    task:                str
    train_accuracy:      float
    test_accuracy:       float
    per_class:           List[ClassMetrics]
    macro_f1:            float
    weighted_f1:         float
    confusion_matrix:    np.ndarray
    label_names:         List[str]
    feature_importances: Optional[np.ndarray]
    feature_names:       List[str]
    top_features:        List[tuple]
    config:              Dict[str, Any]   = field(default_factory=dict)
    warnings_list:       List[str]        = field(default_factory=list)


def evaluate(result: TrainResult) -> EvalResult:
    """Compute full evaluation metrics for a TrainResult."""
    from sklearn.metrics import (
        precision_recall_fscore_support,
        f1_score,
        confusion_matrix,
    )

    model      = result.model
    test_ds    = result.test_dataset
    label_names = test_ds.label_names
    n_classes   = len(label_names)

    warns = list(result.warnings_list)

    # Predictions on test set
    if test_ds.n_samples == 0 or np.isnan(result.test_score):
        y_pred = np.array([], dtype=np.int64)
    else:
        y_pred = model.predict(test_ds.X)

    y_true = test_ds.y

    # ── Per-class metrics ─────────────────────────────────────────────────
    present_classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist()) if len(y_true) > 0 else []

    per_class: List[ClassMetrics] = []
    if len(y_true) > 0 and len(present_classes) >= 2:
        labels_range = list(range(n_classes))
        prec, rec, f1s, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_range, zero_division=0,
        )
        for i, name in enumerate(label_names):
            per_class.append(ClassMetrics(
                label=name,
                precision=float(prec[i]),
                recall=float(rec[i]),
                f1=float(f1s[i]),
                support=int(supports[i]),
            ))
        macro_f1    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    else:
        macro_f1    = float("nan")
        weighted_f1 = float("nan")
        warns.append("Not enough test samples or classes to compute F1 metrics.")

    # ── Confusion matrix ──────────────────────────────────────────────────
    if len(y_true) > 0 and len(present_classes) >= 1:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    else:
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    # ── Feature importances ───────────────────────────────────────────────
    importances = _extract_importances(model, result.train_dataset.X)
    feat_names  = result.train_dataset.feature_names

    if importances is not None:
        top_k = min(10, len(importances))
        top_idx = np.argsort(importances)[::-1][:top_k]
        top_features = [(feat_names[i], float(importances[i])) for i in top_idx]
    else:
        top_features = []

    return EvalResult(
        model_name=          result.model_name,
        task=                result.train_dataset.task,
        train_accuracy=      result.train_score,
        test_accuracy=       result.test_score,
        per_class=           per_class,
        macro_f1=            macro_f1,
        weighted_f1=         weighted_f1,
        confusion_matrix=    cm,
        label_names=         label_names,
        feature_importances= importances,
        feature_names=       feat_names,
        top_features=        top_features,
        config=              result.config,
        warnings_list=       warns,
    )


def _extract_importances(model: Any, X_train: np.ndarray) -> Optional[np.ndarray]:
    """Return feature importances array or None if not available."""
    # Tree-based models expose feature_importances_
    if hasattr(model, "feature_importances_"):
        return np.array(model.feature_importances_, dtype=np.float64)

    # Logistic regression: use abs(coef_) as a proxy
    if hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        # Multi-class: mean abs across classes
        if coef.ndim == 2:
            return np.mean(np.abs(coef), axis=0)
        return np.abs(coef)

    return None


def plot_confusion_matrix(eval_result: EvalResult) -> Optional[Any]:
    """Return a matplotlib Figure of the confusion matrix, or None if unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    cm = eval_result.confusion_matrix
    labels = eval_result.label_names
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), max(3.5, n * 1.0)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Normalize for color intensity, show raw counts as text
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            color = "white" if cm_norm[i, j] < 0.5 else "#1a1a2e"
            ax.text(j, i, str(count), ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=9)
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.set_xlabel("Predicted", color="white", fontsize=10)
    ax.set_ylabel("True", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    acc = eval_result.test_accuracy
    acc_str = f"{acc:.1%}" if not np.isnan(acc) else "N/A"
    ax.set_title(
        f"{eval_result.model_name}  |  {eval_result.task}\n"
        f"Test accuracy: {acc_str}  |  Macro-F1: "
        f"{'N/A' if np.isnan(eval_result.macro_f1) else f'{eval_result.macro_f1:.3f}'}",
        color="white", fontsize=10, pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.tight_layout()
    return fig


def plot_feature_importance(eval_result: EvalResult, top_n: int = 15) -> Optional[Any]:
    """Return a matplotlib Figure of feature importances, or None."""
    if not eval_result.top_features:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    names  = [t[0] for t in eval_result.top_features[:top_n]]
    values = [t[1] for t in eval_result.top_features[:top_n]]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.45)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color="#4a90e2", edgecolor="none", height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, color="white", fontsize=9)
    ax.set_xlabel("Importance", color="white", fontsize=10)
    ax.set_title(
        f"Feature Importance — {eval_result.model_name}",
        color="white", fontsize=10, pad=8,
    )
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.invert_yaxis()

    fig.tight_layout()
    return fig
