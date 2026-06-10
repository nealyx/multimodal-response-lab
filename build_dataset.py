#!/usr/bin/env python3
"""Multi-session behavioral dataset management — Day 13.

Subcommands
-----------
register    Register a new participant in the dataset.
ingest      Add one session directory to the dataset catalog.
rebuild     Re-ingest all sessions from their stored paths.
stats       Print dataset-level statistics.
check       Run quality checks and print a report.
export      Write dataset_summary.json + dataset_summary.csv.

Dataset root
------------
All commands default to ``outputs/dataset/`` as the root directory.
The manifest is stored there as ``dataset_manifest.json``.
Override with ``--dataset-root``.

Usage
-----
    # Register a participant
    python build_dataset.py register --participant participant_001

    # Register with optional metadata
    python build_dataset.py register \\
        --participant participant_001 \\
        --age-range 25-34 \\
        --calibration outputs/calibration/user_profile.json \\
        --notes "recorded in office, natural light"

    # Ingest a session
    python build_dataset.py ingest \\
        --session outputs/sessions/20241201_120000/ \\
        --participant participant_001

    # Ingest with explicit embeddings and labels paths
    python build_dataset.py ingest \\
        --session outputs/sessions/20241201_120000/ \\
        --participant participant_001 \\
        --embeddings outputs/embeddings/20241201_120000/behavioral_windows.csv \\
        --labels outputs/sessions/20241201_120000/labels.json

    # Rebuild catalog after adding artefacts
    python build_dataset.py rebuild

    # Dataset statistics
    python build_dataset.py stats

    # Quality check
    python build_dataset.py check

    # Export summaries
    python build_dataset.py export --output-dir outputs/dataset/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset.checker import DatasetChecker
from src.dataset.exporter import DatasetExporter
from src.dataset.registry import DatasetRegistry
from src.dataset.stats import DatasetStatsComputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = "outputs/dataset"


# ── Subcommand handlers ───────────────────────────────────────────────────────

def cmd_register(args: argparse.Namespace) -> None:
    registry = DatasetRegistry(args.dataset_root)
    p = registry.register_participant(
        participant_id=           args.participant or "",
        age_range=                args.age_range   or "",
        calibration_profile_path= args.calibration or "",
        notes=                    args.notes        or "",
    )
    print(f"\n  Participant registered:")
    print(f"    ID          : {p.participant_id}")
    print(f"    Age range   : {p.age_range or '(not set)'}")
    print(f"    Calibration : {p.calibration_profile_path or '(not set)'}")
    print(f"    Notes       : {p.notes or '(none)'}")
    print(f"\n  Manifest: {registry.manifest_path}\n")


def cmd_ingest(args: argparse.Namespace) -> None:
    if not args.session:
        print("--session is required for ingest")
        sys.exit(1)
    if not args.participant:
        print("--participant is required for ingest")
        sys.exit(1)

    registry = DatasetRegistry(args.dataset_root)
    entry = registry.ingest_session(
        session_dir=      args.session,
        participant_id=   args.participant,
        embeddings_path=  args.embeddings or None,
        labels_path=      args.labels     or None,
        calibration_path= args.calibration or None,
    )

    print(f"\n  Session ingested:")
    print(f"    Session ID   : {entry.session_id}")
    print(f"    Participant  : {entry.participant_id}")
    print(f"    Duration     : {entry.duration_s:.1f} s")
    print(f"    Frames       : {entry.n_frames}")
    print(f"    Windows      : {entry.n_windows}")
    print(f"    Has samples  : {entry.has_samples}")
    print(f"    Has embeddings:{entry.has_embeddings}")
    print(f"    Has labels   : {entry.has_labels}")
    print(f"    Has calibr.  : {entry.has_calibration}")

    if entry.quality_issues:
        print(f"\n  Quality issues ({len(entry.quality_issues)}):")
        for msg in entry.quality_issues:
            print(f"    - {msg}")

    print(f"\n  Manifest: {registry.manifest_path}\n")


def cmd_rebuild(args: argparse.Namespace) -> None:
    registry = DatasetRegistry(args.dataset_root)
    count = registry.rebuild()
    print(f"\n  Rebuilt {count} session(s) in {registry.manifest_path}\n")


def cmd_stats(args: argparse.Namespace) -> None:
    registry = DatasetRegistry(args.dataset_root)
    entries  = registry.all_sessions()
    participants = {pid: p for pid, p in registry.manifest.participants.items()}
    stats    = DatasetStatsComputer.compute(entries, participants)

    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Dataset Statistics  [{args.dataset_root}]")
    print(sep)
    print(f"  Participants     : {stats.n_participants}")
    print(f"  Sessions         : {stats.n_sessions}")
    print(f"  Total hours      : {stats.total_hours:.3f}")
    print(f"  Total frames     : {stats.total_frames:,}")
    print(f"  Total windows    : {stats.total_windows:,}")
    print(f"  With calibration : {stats.sessions_with_calibration}")
    print(f"  With labels      : {stats.sessions_with_labels}")
    print(f"  With embeddings  : {stats.sessions_with_embeddings}")
    print(f"  Mean eng. score  : {stats.mean_engagement_score:.3f}")

    if stats.engagement_distribution:
        print(f"\n  Engagement distribution (frame-weighted):")
        for state, frac in sorted(stats.engagement_distribution.items(),
                                   key=lambda x: -x[1]):
            bar = "█" * int(frac * 30)
            print(f"    {state:<15} {frac:6.1%}  {bar}")

    if stats.label_distribution:
        print(f"\n  Label distribution (by session):")
        for task, counts in sorted(stats.label_distribution.items()):
            print(f"    {task}: HIGH={counts.get('HIGH',0)}  "
                  f"LOW={counts.get('LOW',0)}  "
                  f"ambiguous={counts.get('ambiguous',0)}")

    if stats.per_participant:
        print(f"\n  Per-participant:")
        for pid, ps in sorted(stats.per_participant.items()):
            calib_flag = "calibrated" if ps.has_calibration else "no calibration"
            print(f"    {pid:<25}  sessions={ps.n_sessions}  "
                  f"hours={ps.total_hours:.2f}  "
                  f"windows={ps.total_windows}  "
                  f"[{calib_flag}]")

    print(sep + "\n")


def cmd_check(args: argparse.Namespace) -> None:
    registry = DatasetRegistry(args.dataset_root)
    entries  = registry.all_sessions()
    issues   = DatasetChecker.check_all(entries)
    summary  = DatasetChecker.summary(issues)

    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  Quality Check  ({len(entries)} sessions)")
    print(sep)
    print(f"  Errors   : {summary.get('error', 0)}")
    print(f"  Warnings : {summary.get('warning', 0)}")
    print(f"  Info     : {summary.get('info', 0)}")

    shown_severities = {"error", "warning"}
    if args.verbose:
        shown_severities.add("info")

    printed = [i for i in issues if i.severity in shown_severities]
    if printed:
        print()
        for issue in printed:
            print(f"  {issue}")

    if summary.get("error", 0) == 0 and summary.get("warning", 0) == 0:
        print("\n  All sessions passed quality checks.")
    print(sep + "\n")

    if summary.get("error", 0) > 0:
        sys.exit(1)


def cmd_export(args: argparse.Namespace) -> None:
    registry = DatasetRegistry(args.dataset_root)
    entries  = registry.all_sessions()
    participants = {pid: p for pid, p in registry.manifest.participants.items()}
    stats    = DatasetStatsComputer.compute(entries, participants)
    issues   = DatasetChecker.check_all(entries)

    out_dir = args.output_dir or args.dataset_root
    paths   = DatasetExporter.export_all(registry.manifest, stats, issues, out_dir)

    print(f"\n  Exported:")
    for name, path in paths.items():
        print(f"    {name:<28} {path}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-session behavioral dataset management — Day 13",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-root", default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root directory (default: {DEFAULT_DATASET_ROOT})",
    )

    sub = parser.add_subparsers(dest="command")

    # register
    p_reg = sub.add_parser("register", help="Register a participant")
    p_reg.add_argument("--participant", help="Participant ID (auto-generated if omitted)")
    p_reg.add_argument("--age-range",   help="Age bracket, e.g. 25-34")
    p_reg.add_argument("--calibration", help="Path to user_profile.json")
    p_reg.add_argument("--notes",       help="Free-text annotation")

    # ingest
    p_ing = sub.add_parser("ingest", help="Ingest a session directory")
    p_ing.add_argument("--session",     required=True, help="Session directory path")
    p_ing.add_argument("--participant", required=True, help="Participant ID")
    p_ing.add_argument("--embeddings",  help="Path to behavioral_windows.csv")
    p_ing.add_argument("--labels",      help="Path to labels.json")
    p_ing.add_argument("--calibration", help="Path to user_profile.json (overrides participant's)")

    # rebuild
    sub.add_parser("rebuild", help="Re-ingest all sessions from stored paths")

    # stats
    sub.add_parser("stats", help="Print dataset statistics")

    # check
    p_chk = sub.add_parser("check", help="Run quality checks")
    p_chk.add_argument("--verbose", action="store_true",
                        help="Also show INFO-level issues")

    # export
    p_exp = sub.add_parser("export", help="Export dataset_summary.json + .csv")
    p_exp.add_argument("--output-dir", help="Output directory (default: dataset root)")

    args = parser.parse_args()

    dispatch = {
        "register": cmd_register,
        "ingest":   cmd_ingest,
        "rebuild":  cmd_rebuild,
        "stats":    cmd_stats,
        "check":    cmd_check,
        "export":   cmd_export,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(0)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
