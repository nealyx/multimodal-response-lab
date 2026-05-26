#!/usr/bin/env python3
"""Stimulus-response behavioral experiment — Day 7.

Runs the full Day 2–6 signal pipeline while presenting a timed sequence of
visual stimuli.  All events (stimulus onsets, behavioral samples, responses)
are recorded with monotonic timestamps and exported to JSON + CSV at session end.

What you see on screen
-----------------------
  Left panel:  engagement dashboard (score, state, component bars, fractions)
  Right panel: head pose, gaze, and blink HUD panels from earlier days
  Bottom bar:  experiment phase (BASELINE / TRIAL N/M / ISI / DONE),
               time elapsed, and last key-press reaction time
  Stimulus:    rendered on top of the webcam feed

Controls
--------
  SPACE  — response key for ReactionPrompt trials
  Q/Esc  — quit immediately (session is still exported)

Temporal synchronization model
--------------------------------
All timestamps use time.monotonic().  The zero point is arbitrary and
irrelevant because only differences (latency = response_ts - onset_ts) matter.

Stimulus onset_ts is set when the presenter first calls stimulus.render(),
which happens inside the main loop just before cv2.imshow().  There is a
systematic display latency of ~8–32 ms between this timestamp and the frame
actually appearing on screen — this is NOT corrected here.  For key-press
responses this offset is a constant systematic error (~15 ms typical); it
can be subtracted as a calibration constant in post-processing.

For gaze- and attention-based responses, there is additional lag from EMA
smoothing (alpha=0.10 → ~200–400 ms) that dominates over display latency.

Usage
-----
    python run_stimulus_experiment.py
    python run_stimulus_experiment.py --experiment configs/experiment_default.yaml
    python run_stimulus_experiment.py --device 1 --debug

Press Q or Esc to quit.  Session files are written to:
    outputs/sessions/<session_id>/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines.face_pipeline import FacePipeline
from src.signals.attention import AttentionDetector
from src.signals.blink_detector import BlinkDetector
from src.signals.engagement import AttentionScorer
from src.signals.eye_metrics import extract_eye_measurements
from src.signals.gaze import GazeDetector, extract_gaze_measurements
from src.signals.head_pose import build_camera_matrix, estimate_head_pose
from src.stimulus.analytics import SessionAnalytics
from src.stimulus.events import BehavioralSample, SessionLog
from src.stimulus.exporter import SessionExporter
from src.stimulus.presenter import build_schedule, StimulusPresenter
from src.stimulus.recorder import EventRecorder
from src.utils.camera import print_camera_candidates
from src.utils.overlay import (
    draw_attention_metrics,
    draw_engagement_dashboard,
    draw_eye_metrics,
    draw_face_mesh,
    draw_fps_counter,
    draw_gaze_metrics,
    draw_head_axes,
    draw_iris_markers,
    draw_status_text,
)


def _setup_logging(cfg: dict, debug: bool) -> None:
    level = logging.DEBUG if debug else getattr(
        logging, cfg.get("level", "INFO").upper(), logging.INFO,
    )
    logging.basicConfig(
        level=level,
        format=cfg.get("format", "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"),
        datefmt=cfg.get("datefmt", "%H:%M:%S"),
    )


def _load_config(base_path: str, experiment_path: str) -> dict:
    with open(base_path) as fh:
        cfg = yaml.safe_load(fh)
    with open(experiment_path) as fh:
        exp_cfg = yaml.safe_load(fh)
    cfg.update(exp_cfg)
    return cfg


def _draw_hud_bar(
    frame,
    phase: str,
    elapsed_s: float,
    trial_n: int,
    trial_total: int,
    last_latency_ms: Optional[float],
) -> None:
    """Render the experiment status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    y    = h - 12
    font = cv2.FONT_HERSHEY_SIMPLEX

    phase_colors = {
        "BASELINE": (80,  80,  80),
        "ISI":      (80,  80,  80),
        "DONE":     (0,  200,  80),
    }
    col = phase_colors.get(phase, (0, 200, 255))

    trial_str = (
        f"TRIAL {trial_n}/{trial_total}"
        if phase not in ("BASELINE", "ISI", "DONE")
        else phase
    )
    lat_str = (
        f"  last RT: {last_latency_ms:.0f} ms"
        if last_latency_ms is not None else ""
    )
    text = f"  {trial_str}  |  {elapsed_s:.1f} s{lat_str}"

    cv2.rectangle(frame, (0, h - 28), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, text, (8, y), font, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (8, y), font, 0.50, col,       1, cv2.LINE_AA)


def run(cfg: dict, debug: bool, output_dir: str) -> None:
    face_cfg    = cfg.get("face_mesh", {})
    capture_cfg = cfg.get("capture", {})
    overlay_cfg = cfg.get("overlay", {})
    signals_cfg = cfg.get("signals", {})
    hp_cfg      = cfg.get("head_pose", {})
    out_cfg     = cfg.get("output", {})

    log = logging.getLogger(__name__)

    print_camera_candidates(capture_cfg)

    # ── Signal pipeline (Days 2–6) ─────────────────────────────────────────
    blink_det     = BlinkDetector(signals_cfg)
    attention_det = AttentionDetector(cfg)
    gaze_det      = GazeDetector(cfg)
    scorer        = AttentionScorer(cfg)

    focal_scale = float(hp_cfg.get("focal_scale", 1.0))
    max_reproj  = float(hp_cfg.get("max_reprojection_error", 10.0))

    # ── Session setup ──────────────────────────────────────────────────────
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log = SessionLog(
        session_id=session_id,
        experiment_name=cfg.get("experiment", {}).get("name", "unnamed"),
        start_ts=time.monotonic(),
        end_ts=None,
        config=cfg,
    )
    recorder = EventRecorder(session_log)

    # Build schedule; t0 is the session start
    trials   = build_schedule(cfg, t0=session_log.start_ts)
    presenter = StimulusPresenter(trials)
    n_trials  = len(trials)

    max_duration = float(cfg.get("experiment", {}).get("max_duration_s", 300.0))

    # ── State tracking ─────────────────────────────────────────────────────
    frame_times:    deque  = deque(maxlen=30)
    no_face_streak: int    = 0
    cam_matrix             = None
    last_blink_a           = None
    last_latency_ms: Optional[float] = None
    trials_seen:     int   = 0

    log.info(
        "Session %s  — %d trials scheduled.  Press Q/Esc to quit.",
        session_id, n_trials,
    )
    print(f"\n  Session: {session_id}  |  {n_trials} trials")
    print(f"  Baseline: {cfg.get('experiment',{}).get('baseline_duration_s',10):.0f} s  "
          f"before first stimulus\n")

    with FacePipeline(face_cfg, capture_cfg) as pipeline:
        while True:
            result = pipeline.process_latest(timeout=0.1)
            if result is None:
                continue

            ts = result.timestamp_process
            elapsed_s = ts - session_log.start_ts

            # Session time cap
            if elapsed_s > max_duration:
                log.info("Session time cap reached (%.0f s).", max_duration)
                break

            frame_times.append(ts)
            fps = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) >= 2 else 0.0
            )

            h, w     = result.frame_raw.shape[:2]
            display  = result.frame_raw.copy()
            fi       = result.frame_index

            if cam_matrix is None:
                cam_matrix = build_camera_matrix(w, h, focal_scale=focal_scale)

            # ── Signal pipeline ────────────────────────────────────────────
            blink_a = att_a = gaze_a = eng = None

            if result.face_detected:
                no_face_streak = 0
                lm = result.face_landmarks[0]

                eye_m   = extract_eye_measurements(lm, w, h, timestamp=ts, frame_index=fi, cfg=signals_cfg)
                blink_a = blink_det.update(eye_m) if eye_m else blink_det.no_face_update()
                last_blink_a = blink_a

                pose  = estimate_head_pose(lm, w, h, cam_matrix, timestamp=ts, frame_index=fi)
                att_a = attention_det.update(pose) if pose else attention_det.no_pose_update()

                gaze_m = extract_gaze_measurements(
                    lm, w, h,
                    mean_ear=eye_m.mean_ear if eye_m else 0.0,
                    timestamp=ts, frame_index=fi, cfg=cfg.get("gaze"),
                )
                if gaze_m:
                    gaze_a = gaze_det.update(
                        gaze_m,
                        head_zone=(att_a.zone.value if att_a else None),
                        head_yaw= (pose.yaw        if pose  else None),
                        head_pitch=(pose.pitch      if pose  else None),
                    )
                else:
                    gaze_a = gaze_det.no_gaze_update()

                eng = scorer.update(
                    blink=blink_a, head=att_a, gaze=gaze_a,
                    face_detected=True, timestamp=ts, frame_index=fi,
                )
            else:
                no_face_streak += 1
                if no_face_streak >= 30:
                    blink_det.reset(); attention_det.reset()
                    gaze_det.reset();  scorer.reset()
                    no_face_streak = 0
                eng = scorer.update(face_detected=False, timestamp=ts, frame_index=fi)

            # ── Stimulus presenter (overlay on frame) ──────────────────────
            display, onset_ev, offset_ev = presenter.update(display, ts)

            if onset_ev:
                recorder.record_stimulus_onset(onset_ev)
                trials_seen += 1
                log.debug("onset  id=%s  trial=%d/%d", onset_ev.stimulus_id, trials_seen, n_trials)
            if offset_ev:
                recorder.record_stimulus_offset(offset_ev)
                log.debug("offset id=%s", offset_ev.stimulus_id)

            # ── Build and record behavioral sample ─────────────────────────
            sample = BehavioralSample(
                frame_index=fi, timestamp=ts,
                active_stimulus_id=presenter.active_stimulus_id,
                engagement_state= eng.state.value,
                smoothed_score=   eng.smoothed_score,
                confidence=       eng.confidence,
                focused_fraction= eng.focused_fraction,
                gaze_zone=    gaze_a.zone.value  if gaze_a  else "unknown",
                is_on_screen= gaze_a.is_on_screen if gaze_a  else False,
                gaze_h=       gaze_a.mean_h       if gaze_a  else 0.5,
                gaze_v=       gaze_a.mean_v        if gaze_a  else 0.5,
                head_zone=    att_a.zone.value     if att_a   else "unknown",
                head_yaw=     att_a.yaw            if att_a   else 0.0,
                head_pitch=   att_a.pitch          if att_a   else 0.0,
                blink_state=  blink_a.state.value  if blink_a else "unknown",
                mean_ear=     blink_a.mean_ear      if blink_a else 0.0,
                blink_rate=   blink_a.blink_rate_per_min if blink_a else 0.0,
                is_fatigued=  eng.is_fatigued,
            )
            recorder.record_sample(sample, presenter.active_event)

            # ── Render HUD ─────────────────────────────────────────────────
            if result.face_detected and lm is not None:
                display = draw_face_mesh(
                    display, lm,
                    draw_tesselation=overlay_cfg.get("draw_tesselation", False),
                    draw_contour=    overlay_cfg.get("draw_contour",     True),
                    draw_eyes=       overlay_cfg.get("draw_eyes",        True),
                    draw_lips=       overlay_cfg.get("draw_lips",        False),
                    draw_irises=     overlay_cfg.get("draw_irises",      False),
                )
                if gaze_m is not None:
                    display = draw_iris_markers(display, gaze_m)
                if pose is not None:
                    display = draw_head_axes(display, pose.rvec, pose.tvec, cam_matrix)

            display = draw_engagement_dashboard(display, eng, blink_a, origin=(10, 20))

            rhs = max(w - 280, 10)
            if att_a is not None:
                display = draw_attention_metrics(display, att_a, origin=(rhs, 60))
            if gaze_a is not None:
                display = draw_gaze_metrics(display, gaze_a, origin=(rhs, 200))
            if blink_a is not None and eye_m is not None:
                display = draw_eye_metrics(
                    display, blink_a,
                    ear_l=eye_m.left_ear, ear_r=eye_m.right_ear,
                    origin=(rhs, 380),
                )
            if not result.face_detected:
                display = draw_status_text(display, "No face detected")

            # Determine phase label for status bar
            if presenter.active_stimulus_id:
                phase = presenter.active_stimulus_id
            elif elapsed_s < cfg.get("experiment", {}).get("baseline_duration_s", 10.0):
                phase = "BASELINE"
            elif presenter.is_finished():
                phase = "DONE"
            else:
                phase = "ISI"

            _draw_hud_bar(display, phase, elapsed_s, trials_seen, n_trials, last_latency_ms)

            display = draw_fps_counter(display, fps)
            cv2.imshow("Stimulus Experiment  |  SPACE=respond  Q/Esc=quit", display)

            # ── Key handling ───────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            if key != 255:  # any key pressed
                resp_key = presenter.notify_key(key, ts)
                if resp_key and presenter.active_event:
                    resp = recorder.record_key_response(presenter.active_event, ts)
                    last_latency_ms = resp.latency_ms
                    log.debug(
                        "key response  stim=%s  latency=%.0f ms",
                        resp.stimulus_id, resp.latency_ms,
                    )

            if presenter.is_finished():
                log.info("All trials complete.")
                # Brief pause so the user can see the final frame
                cv2.waitKey(1500)
                break

    cv2.destroyAllWindows()

    # ── Session end ────────────────────────────────────────────────────────
    session_log.end_ts = time.monotonic()

    # Compute analytics
    analytics  = SessionAnalytics()
    summaries  = analytics.compute_trial_summaries(session_log)
    session_log.trial_summaries = summaries
    agg        = analytics.aggregate_by_type(summaries)
    overall    = analytics.overall_stats(session_log)

    # Print session summary
    _print_summary(session_log, overall, agg, last_latency_ms)

    # Export
    save_json = out_cfg.get("save_json", True)
    save_csv  = out_cfg.get("save_csv",  True)
    out_root  = Path(output_dir) / session_id

    if save_json:
        json_path = str(out_root / "session_log.json")
        SessionExporter.to_json(session_log, json_path,
                                summaries=summaries, aggregate=agg, overall=overall)
        print(f"  JSON → {json_path}")

    if save_csv:
        SessionExporter.to_csv(
            session_log,
            samples_path=str(out_root / "samples.csv"),
            events_path= str(out_root / "events.csv"),
        )
        print(f"  CSV  → {out_root}/")

    if not save_json and not save_csv:
        print("  (export disabled in config)")


def _print_summary(log, overall, agg, last_latency_ms):
    print(f"\n── Session summary  [{log.session_id}] ──────────────────────────")
    d = overall
    if d:
        print(f"  Duration          : {d.get('duration_s', 0):.1f} s")
        print(f"  Frames processed  : {d.get('n_frames', 0)}")
        print(f"  Trials completed  : {d.get('n_trials', 0)}")
        print(f"  Total responses   : {d.get('n_responses', 0)}")
        print(f"  Mean eng. score   : {d.get('mean_score', 0):.3f}")
        print(f"  Focused           : {d.get('focused_pct', 0):.1f}%")
        print(f"  On-screen         : {d.get('on_screen_pct', 0):.1f}%")
        if d.get("mean_key_latency_ms") is not None:
            print(f"  Mean RT (space)   : {d['mean_key_latency_ms']:.0f} ms")
    if agg:
        print("  Per-type stats:")
        for stype, stats in agg.items():
            lat = stats.get("mean_latency_ms")
            lat_s = f"{lat:.0f} ms" if lat else "—"
            print(f"    {stype:<20} n={stats['n_trials']}  "
                  f"hit={stats['hit_rate']*100:.0f}%  mean_RT={lat_s}")
    print("────────────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stimulus-response experiment — Day 7")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--experiment", default="configs/experiment_default.yaml",
                        help="Experiment config (stimuli, timing, output settings)")
    parser.add_argument("--output-dir", default="outputs/sessions",
                        help="Root directory for session output files")
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--device",     type=int, default=None, metavar="N")
    parser.add_argument("--backend",    default=None, metavar="NAME")
    args = parser.parse_args()

    cfg = _load_config(args.config, args.experiment)
    if args.device  is not None:
        cfg.setdefault("capture", {})["device_id"] = args.device
    if args.backend is not None:
        cfg.setdefault("capture", {})["backend"]   = args.backend

    _setup_logging(cfg.get("logging", {}), args.debug)
    run(cfg, debug=args.debug, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
