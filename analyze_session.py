#!/usr/bin/env python3
"""Offline session analytics and report generation — Day 8.

Loads a session produced by run_stimulus_experiment.py and computes
session-level behavioral metrics, generates charts, and writes an
HTML report.

Usage
-----
    # Analyze a session directory (contains session_log.json + samples.csv)
    python analyze_session.py outputs/sessions/20241201_120000/

    # Analyze just the JSON (no sample-based metrics)
    python analyze_session.py outputs/sessions/20241201_120000/session_log.json

    # Specify output directory
    python analyze_session.py <session> --output-dir outputs/reports/

    # Skip charts or HTML
    python analyze_session.py <session> --no-charts
    python analyze_session.py <session> --no-html

    # Open the HTML report in the default browser
    python analyze_session.py <session> --open

Output layout
-------------
    <output_dir>/<session_id>/
        session_metrics.json   — SessionMetrics as JSON
        timeline.csv           — sample-by-sample time series with relative timestamp
        charts/
            engagement_timeline.png
            state_distribution.png
            signal_channels.png
            stimulus_comparison.png   (only if stimuli were presented)
            reaction_latency.png      (only if key-press responses recorded)
        report.html            — self-contained HTML with embedded charts

Why keep analytics separate from live inference
-----------------------------------------------
The live pipeline (run_stimulus_experiment.py) must maintain <33 ms frame
processing to avoid dropped frames.  Generating matplotlib figures, writing
multi-MB HTML files, and computing distribution statistics would all disrupt
timing.  Separating them means:
  - The live script is simple: record and persist.
  - Reanalysis with different parameters is cheap: just re-run this script.
  - The reporting layer has no dependency on OpenCV or MediaPipe.

Per-stimulus analysis makes the system more research-like
---------------------------------------------------------
Aggregate session metrics (mean score, focused %) are useful for dashboards.
Per-stimulus analysis — comparing engagement score during a ColorFlash vs. a
ReactionPrompt, measuring how quickly the score recovers after each trial —
is what makes the system comparable to research paradigms like ERP studies or
event-related fMRI.  It allows causal questions: "did this specific stimulus
disrupt attention?" rather than "was this person generally engaged?"

Limitations
-----------
  - Display latency (~8–32 ms) is NOT corrected in reaction times.
  - Frame-rate granularity limits temporal precision to ~33 ms.
  - EMA smoothing in the engagement scorer adds 200–400 ms lag to state-change
    responses, so 'attention_change' latencies are systematically longer than
    'key_press' latencies.
  - Population-average thresholds: per-user calibration would improve accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import webbrowser
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.calibration.baseline import UserProfile
from src.calibration.comparison import BaselineComparator
from src.reporting.charts import generate_all, figure_to_png_bytes
from src.reporting.html_report import generate_html
from src.reporting.loader import load_session_dir, load_session_json
from src.reporting.reporter import SessionReporter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _resolve_input(path_str: str):
    """Return (session_log, load_fn) based on whether input is a dir or file."""
    p = Path(path_str)
    if p.is_dir():
        return load_session_dir(str(p))
    elif p.is_file() and p.suffix == ".json":
        return load_session_json(str(p))
    else:
        raise ValueError(f"Input must be a session directory or session_log.json, got: {path_str}")


def _write_metrics_json(metrics, out_dir: Path) -> Path:
    path = out_dir / "session_metrics.json"

    def _default(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return str(obj)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(metrics), fh, indent=2, default=_default)
    return path


def _write_timeline_csv(session_log, out_dir: Path) -> Path:
    """Write a timeline CSV with relative timestamps (seconds from session start)."""
    path = out_dir / "timeline.csv"
    if not session_log.samples:
        log.warning("No samples to write timeline from")
        return path

    start = session_log.start_ts
    fields = [
        "time_s", "frame_index", "active_stimulus_id",
        "engagement_state", "smoothed_score", "confidence",
        "is_on_screen", "gaze_zone", "gaze_h", "gaze_v",
        "head_zone", "head_yaw", "head_pitch",
        "blink_state", "blink_rate", "mean_ear", "is_fatigued",
    ]

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in session_log.samples:
            writer.writerow({
                "time_s":             round(s.timestamp - start, 4),
                "frame_index":        s.frame_index,
                "active_stimulus_id": s.active_stimulus_id or "",
                "engagement_state":   s.engagement_state,
                "smoothed_score":     s.smoothed_score,
                "confidence":         s.confidence,
                "is_on_screen":       s.is_on_screen,
                "gaze_zone":          s.gaze_zone,
                "gaze_h":             round(s.gaze_h, 4),
                "gaze_v":             round(s.gaze_v, 4),
                "head_zone":          s.head_zone,
                "head_yaw":           round(s.head_yaw, 2),
                "head_pitch":         round(s.head_pitch, 2),
                "blink_state":        s.blink_state,
                "blink_rate":         round(s.blink_rate, 2),
                "mean_ear":           round(s.mean_ear, 4),
                "is_fatigued":        s.is_fatigued,
            })
    return path


def _print_baseline_comparison(metrics, profile: UserProfile, session_dir: str) -> None:
    """Print a z-score comparison of session signals against the user's profile."""
    e = metrics.engagement
    b = metrics.blink

    comparison = BaselineComparator.compare(
        profile,
        session_dir=session_dir,
        blink_rate=       b.blink_rate_summary.mean if b.blink_rate_summary else None,
        engagement_score= e.score_summary.mean      if e.score_summary else None,
    )

    print(f"\n── Baseline Comparison  [profile: {profile.profile_id}] ──────────────────────")
    for line in comparison.summary_lines():
        print(line)
    print("────────────────────────────────────────────────────────────────────")


def _print_summary(metrics) -> None:
    e = metrics.engagement
    g = metrics.gaze
    b = metrics.blink

    dur = f"{metrics.duration_s:.1f} s" if metrics.duration_s else "unknown"
    print(f"\n── Session  [{metrics.session_id}] ─────────────────────────────────────")
    print(f"  Experiment      : {metrics.experiment_name}")
    print(f"  Duration        : {dur}  ({metrics.n_frames} frames)")
    print(f"  Trials / Resp.  : {metrics.n_trials} / {metrics.n_responses}")
    print(f"  Engagement score: {e.score_summary.mean:.3f} ± {e.score_summary.std:.3f}")
    print(f"  Focused         : {e.focused_fraction*100:.1f}%")
    print(f"  Drifting        : {e.drifting_fraction*100:.1f}%")
    print(f"  Distracted      : {e.distracted_fraction*100:.1f}%")
    print(f"  Fatigued        : {e.fatigued_fraction*100:.1f}%")
    print(f"  Unreliable      : {e.unreliable_fraction*100:.1f}%")
    print(f"  On-screen       : {g.on_screen_fraction*100:.1f}%")
    print(f"  Blink rate      : {b.blink_rate_summary.mean:.1f} ± {b.blink_rate_summary.std:.1f} /min")
    if metrics.stimulus_types:
        print("  Per-type:")
        for stype, st in sorted(metrics.stimulus_types.items()):
            lat = f"{st.mean_latency_ms:.0f} ms" if st.mean_latency_ms else "—"
            print(f"    {stype:<22} n={st.n_trials}  hit={st.hit_rate*100:.0f}%  RT={lat}  Δscore={st.score_delta:+.3f}")
    if metrics.reaction_latencies_ms:
        import statistics
        lats = metrics.reaction_latencies_ms
        print(f"  Key-press RT    : mean={statistics.mean(lats):.0f} ms  "
              f"median={statistics.median(lats):.0f} ms  n={len(lats)}")
    print("────────────────────────────────────────────────────────────────────")


def run(args: argparse.Namespace) -> None:
    # ── Load session ──────────────────────────────────────────────────────
    log.info("Loading session from %s", args.input)
    session_log = _resolve_input(args.input)
    log.info(
        "Loaded session %s  |  %d samples  |  %d trials  |  %d responses",
        session_log.session_id,
        len(session_log.samples),
        len(session_log.stimulus_events),
        len(session_log.response_events),
    )

    # ── Compute metrics ───────────────────────────────────────────────────
    reporter = SessionReporter()
    metrics  = reporter.compute(session_log)

    _print_summary(metrics)

    # ── Baseline comparison (optional) ────────────────────────────────────
    profile_path = getattr(args, "profile", None)
    profile = UserProfile.try_load(profile_path)
    if profile is not None:
        _print_baseline_comparison(metrics, profile, str(args.input))
    elif profile_path:
        log.warning("Could not load profile from %s — skipping baseline comparison", profile_path)

    # ── Output directory ──────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / session_log.session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    # ── Metrics JSON ──────────────────────────────────────────────────────
    p = _write_metrics_json(metrics, out_dir)
    log.info("Metrics JSON → %s", p)

    # ── Timeline CSV ──────────────────────────────────────────────────────
    p = _write_timeline_csv(session_log, out_dir)
    log.info("Timeline CSV → %s", p)

    # ── Charts ────────────────────────────────────────────────────────────
    chart_pngs: dict = {}
    if not args.no_charts:
        log.info("Generating charts …")
        figs = generate_all(session_log, metrics)
        for name, fig in figs.items():
            png  = figure_to_png_bytes(fig)
            path = charts_dir / f"{name}.png"
            path.write_bytes(png)
            chart_pngs[name] = png
            log.info("  chart → %s", path)
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass

    # ── HTML report ───────────────────────────────────────────────────────
    html_path = None
    if not args.no_html:
        trial_dicts = [
            {
                "trial_index":        t.trial_index,
                "stimulus_id":        t.stimulus_id,
                "stimulus_type":      t.stimulus_type,
                "reaction_latency_ms": t.reaction_latency_ms,
                "response_type":      t.response_type,
                "hit":                t.hit,
                "baseline_score":     t.baseline_score,
                "during_score":       t.during_score,
                "recovery_s":         t.recovery_s,
            }
            for t in session_log.trial_summaries
        ]
        html = generate_html(metrics, chart_pngs, trial_dicts)
        html_path = out_dir / "report.html"
        html_path.write_text(html, encoding="utf-8")
        log.info("HTML report → %s", html_path)

    print(f"\n  Output directory: {out_dir}/")
    if html_path:
        print(f"  HTML report:      {html_path}")
    if args.open and html_path and html_path.exists():
        webbrowser.open(html_path.as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and report a behavioral session — Day 8",
    )
    parser.add_argument(
        "input",
        help="Session directory (contains session_log.json + samples.csv) "
             "or path to session_log.json directly",
    )
    parser.add_argument(
        "--output-dir", default="outputs/reports",
        help="Root directory for report output (default: outputs/reports/)",
    )
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--no-html",   action="store_true", help="Skip HTML report")
    parser.add_argument("--open",      action="store_true", help="Open HTML report in browser")
    parser.add_argument(
        "--profile", metavar="USER_PROFILE_JSON",
        help="Path to a user_profile.json from run_calibration.py. "
             "When provided, prints a z-score comparison of session signals "
             "against the personal baseline.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
