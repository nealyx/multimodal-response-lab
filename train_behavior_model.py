#!/usr/bin/env python3
"""Supervised behavioral prediction — Day 11/12.

Loads one or more behavioral_windows.csv files (from embed_session.py),
assigns labels (heuristic or human-survey), trains baseline classifiers, and
writes model artifacts alongside evaluation metrics.

Label sources
-------------
  heuristic  (default) : Rules derived from the same signals used as features.
                         Circular — the model reproduces the formula, not an
                         independent cognitive state.  Good for pipeline
                         validation; not evidence of real inference.
  human                : Post-session survey ratings (engagement/fatigue/
                         distraction) collected via run_calibration.py.
                         Independent of the signals; less circular.
                         Requires --labels pointing to one or more labels.json
                         files and --task matching a survey task name
                         (engagement, fatigue, distraction).

Why metrics on a single session can mislead
--------------------------------------------
A 5-minute session at 30 fps with 2 s windows / 0.5 s stride produces ~170
windows.  An 80/20 split gives ~34 test samples.  Accuracy on 34 samples has
a 95% CI of roughly ±8 pp around a 70% baseline.  Use multiple sessions or
temporal hold-out for reliable estimates.

Usage
-----
    # Heuristic labels (default)
    python train_behavior_model.py outputs/embeddings/<session_id>/behavioral_windows.csv
    python train_behavior_model.py <csv> --task focused_vs_distracted --models random_forest
    python train_behavior_model.py <csv> --list-tasks

    # Human-survey labels
    python train_behavior_model.py <csv> --label-source human \\
        --labels outputs/sessions/<sid>/labels.json --task engagement

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
from src.supervised.labels import BUILTIN_TASKS, load_human_label_windows
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

    label_source = getattr(args, "label_source", "heuristic")
    task_name    = args.task

    if label_source == "human":
        # ── Human-survey labels ───────────────────────────────────────────
        human_tasks = ("engagement", "fatigue", "distraction")
        if task_name not in human_tasks:
            print(
                f"--label-source human requires --task in {human_tasks}, "
                f"got '{task_name}'."
            )
            sys.exit(1)
        label_files = getattr(args, "labels", None) or []
        if not label_files:
            print("--label-source human requires --labels <path(s) to labels.json>.")
            sys.exit(1)

        kept, labels_arr, label_names, excluded = load_human_label_windows(
            windows, label_files, task_name
        )
        log.info(
            "Human labels: %d kept, %d excluded  |  classes: %s",
            len(kept), excluded, label_names,
        )
        if len(kept) < 4:
            print(
                f"\nOnly {len(kept)} labeled windows for task '{task_name}' "
                "with human labels.\n"
                "Check that the session IDs in your labels.json match the CSV filenames."
            )
            sys.exit(1)

        # Build a pseudo-LabelConfig so the rest of the pipeline is unchanged
        from src.supervised.labels import LabelConfig
        config = LabelConfig(
            task=      task_name,
            label_type="binary",
            classes=   label_names,
        )
        dataset = build_dataset(kept, config)
        # Overwrite the labels with the human ones (build_dataset assigns heuristic)
        dataset = dataset.__class__(
            X=             dataset.X,
            y=             labels_arr,
            feature_names= dataset.feature_names,
            label_names=   label_names,
            task=          task_name,
            n_excluded=    excluded,
            session_ids=   dataset.session_ids,
            window_ids=    dataset.window_ids,
        )

    else:
        # ── Heuristic labels (default) ────────────────────────────────────
        if task_name not in BUILTIN_TASKS:
            print(f"Unknown task '{task_name}'. Use --list-tasks to see options.")
            sys.exit(1)
        config = BUILTIN_TASKS[task_name]()
        log.info("Task: %s  |  Classes: %s", config.task, config.classes)
        dataset = build_dataset(windows, config)
    log.info(
        "Dataset: %d samples  |  excluded: %d  |  class counts: %s",
        dataset.n_samples,
        dataset.n_excluded,
        dataset.class_counts,
    )

    if label_source == "heuristic" and dataset.n_samples < 4:
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
    parser.add_argument(
        "--label-source", choices=["heuristic", "human"], default="heuristic",
        help="Label source: 'heuristic' (default) or 'human' (survey ratings from "
             "run_calibration.py). Human labels require --labels and a survey task.",
    )
    parser.add_argument(
        "--labels", nargs="+", metavar="LABELS_JSON",
        help="Path(s) to labels.json files (required when --label-source human).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
