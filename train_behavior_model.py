#!/usr/bin/env python3
"""Supervised behavioral prediction — Day 11.

Loads one or more behavioral_windows.csv files (from embed_session.py),
assigns heuristic labels, trains baseline classifiers, and writes model
artifacts alongside evaluation metrics.

Why heuristic labels are not ground truth
------------------------------------------
Labels are derived from the same signals used as features (e.g., "high
attention" is defined as engagement_mean > 0.65, and engagement_mean is
feature 0).  The model therefore learns to reproduce the labeling rule, not
to predict an independent measure of cognition.  This is useful for validating
the pipeline and identifying discriminative features, but cannot be presented
as evidence of accurate cognitive state inference.

Why metrics on a single session can mislead
--------------------------------------------
A 5-minute session at 30 fps with 2 s windows / 0.5 s stride produces ~170
windows.  An 80/20 split gives ~34 test samples.  Accuracy on 34 samples has
a 95% CI of roughly ±8 pp around a 70% baseline.  Use multiple sessions or
temporal hold-out for reliable estimates.

Usage
-----
    python train_behavior_model.py outputs/embeddings/<session_id>/behavioral_windows.csv
    python train_behavior_model.py embeddings/s1/behavioral_windows.csv embeddings/s2/behavioral_windows.csv
    python train_behavior_model.py <csv> --task focused_vs_distracted --models random_forest
    python train_behavior_model.py <csv> --task high_vs_low_attention --split temporal
    python train_behavior_model.py <csv> --list-tasks

Output layout
-------------
    outputs/models/<task>/<model_name>/
        model.pkl
        metrics.json
        confusion_matrix.png
        feature_importance.csv
        feature_importance.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.supervised.dataset import build_dataset, load_windows_multi
from src.supervised.evaluator import evaluate
from src.supervised.exporter import ModelExporter
from src.supervised.labels import BUILTIN_TASKS
from src.supervised.trainer import BehaviorModelTrainer, MODEL_FACTORIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _print_eval(eval_result) -> None:
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  Model : {eval_result.model_name}")
    print(f"  Task  : {eval_result.task}")
    print(f"  Train accuracy : {eval_result.train_accuracy:.1%}")
    print(f"  Test  accuracy : {eval_result.test_accuracy:.1%}"
          if not np.isnan(eval_result.test_accuracy)
          else "  Test  accuracy : N/A")

    if eval_result.per_class:
        print(f"\n  {'Class':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>4}")
        print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*4}")
        for cm in eval_result.per_class:
            print(f"  {cm.label:<22} {cm.precision:6.3f} {cm.recall:6.3f} "
                  f"{cm.f1:6.3f} {cm.support:4d}")
        mf1 = eval_result.macro_f1
        wf1 = eval_result.weighted_f1
        print(f"\n  Macro-F1   : {'N/A' if np.isnan(mf1) else f'{mf1:.3f}'}")
        print(f"  Weighted-F1: {'N/A' if np.isnan(wf1) else f'{wf1:.3f}'}")

    if eval_result.top_features:
        print(f"\n  Top features (importance):")
        for name, imp in eval_result.top_features[:8]:
            bar = "█" * int(imp * 30)
            print(f"    {name:<24} {imp:.4f}  {bar}")

    if eval_result.warnings_list:
        print(f"\n  ⚠  Warnings:")
        for w in eval_result.warnings_list:
            print(f"    - {w}")

    print(sep)


def run(args: argparse.Namespace) -> None:
    if args.list_tasks:
        print("\nAvailable label tasks:")
        for name in BUILTIN_TASKS:
            print(f"  {name}")
        return

    # ── Load windows ──────────────────────────────────────────────────────
    log.info("Loading %d CSV file(s)...", len(args.inputs))
    windows = load_windows_multi(args.inputs)
    log.info("Loaded %d behavioral windows total", len(windows))

    if not windows:
        print("No windows loaded. Run embed_session.py first.")
        sys.exit(1)

    # ── Build label config ────────────────────────────────────────────────
    task_name = args.task
    if task_name not in BUILTIN_TASKS:
        print(f"Unknown task '{task_name}'. Use --list-tasks to see options.")
        sys.exit(1)

    config = BUILTIN_TASKS[task_name]()
    log.info("Task: %s  |  Classes: %s", config.task, config.classes)

    # ── Build dataset ─────────────────────────────────────────────────────
    dataset = build_dataset(windows, config)
    log.info(
        "Dataset: %d samples  |  excluded: %d  |  class counts: %s",
        dataset.n_samples,
        dataset.n_excluded,
        dataset.class_counts,
    )

    if dataset.n_samples < 4:
        print(
            f"\nOnly {dataset.n_samples} labeled samples for task '{task_name}'.\n"
            "Record a longer session or choose a different task.\n"
            "Use --list-tasks to see available tasks."
        )
        sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────
    model_names = args.models if args.models else list(MODEL_FACTORIES.keys())
    trainer = BehaviorModelTrainer(
        model_names=    model_names,
        test_size=      args.test_size,
        split_strategy= args.split,
        random_state=   args.seed,
    )
    results = trainer.train_all(dataset)
    log.info("Trained %d model(s)", len(results))

    # ── Evaluate + export ─────────────────────────────────────────────────
    for result in results:
        eval_result = evaluate(result)
        _print_eval(eval_result)

        out_dir = Path(args.output_dir) / task_name / result.model_name
        saved   = ModelExporter.save_all(result, eval_result, str(out_dir))

        log.info("Artifacts written to %s/", out_dir)
        for artifact, path in saved.items():
            log.info("  %-28s %s", artifact + ":", path)

    print(f"\nOutput root: {Path(args.output_dir) / task_name}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised behavioral prediction — Day 11"
    )
    parser.add_argument(
        "inputs", nargs="*",
        help="One or more behavioral_windows.csv paths",
    )
    parser.add_argument(
        "--task",
        default="high_vs_low_attention",
        help="Label task (default: high_vs_low_attention). Use --list-tasks.",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List available label tasks and exit",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_FACTORIES.keys()),
        help="Which models to train (default: all three)",
    )
    parser.add_argument(
        "--output-dir", default="outputs/models",
        help="Root output directory (default: outputs/models/)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20,
        help="Fraction held out for testing (default: 0.20)",
    )
    parser.add_argument(
        "--split", choices=["random", "temporal"], default="random",
        help="Train/test split strategy (default: random)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
