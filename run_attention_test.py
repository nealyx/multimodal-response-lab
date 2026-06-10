#!/usr/bin/env python3
"""Composite engagement scoring live test — Day 6.

Runs the full pipeline:
  capture → face detection → EAR → blink FSM
                           → head pose → attention zone
                           → iris position → gaze zone
                           → composite engagement score + dashboard overlay

The engagement dashboard (top-left panel) shows:
  - State badge: FOCUSED / DRIFTING / DISTRACTED / FATIGUED / UNRELIABLE
  - Composite smoothed score and confidence progress bars
  - Component sub-scores: gaze / head / eye
  - Rolling focused% and engaged% fractions (60 s window)
  - Blink rate and fatigue label

The smaller HUD panels from earlier days (head pose, gaze, blink) are
preserved on the right side so all layers are visible simultaneously.

Usage
-----
    python run_attention_test.py
    python run_attention_test.py --device 1 --debug

Press Q or Esc to quit.

What to watch for when validating
-----------------------------------
  - Sitting squarely, looking at screen → state badge should settle on FOCUSED
    within ~10–15 s (confidence warm-up period).
  - Looking away deliberately → score drops, state transitions to DISTRACTED
    (after EMA + debounce: ~1–2 s lag is normal).
  - Closing eyes for >0.5 s → FATIGUED badge.
  - Confidence bar should build from 0 to >0.8 after ~10 s.
  - Focused% should approach 90–100% during uninterrupted screen reading.
  - UNRELIABLE appears whenever the face is absent or confidence < 0.20.

Why there is lag in state changes
-----------------------------------
Two layers of intentional lag prevent the badge from flickering:
  1. EMA smoothing (alpha=0.10): a single distracted frame barely moves the
     score; genuine attention changes take several seconds to appear.
  2. State debounce (state_min_frames=5): a new state must be the candidate
     for 5 consecutive frames before the label flips.
This makes the dashboard useful as a session-level signal rather than a
frame-level alert.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from pathlib import Path

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
from src.utils.camera import print_camera_candidates
from src.utils.overlay import (
    draw_attention_metrics,
    draw_demo_overlay,
    draw_diagnostics_panel,
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
        logging, cfg.get("level", "INFO").upper(), logging.INFO
    )
    logging.basicConfig(
        level=level,
        format=cfg.get("format", "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"),
        datefmt=cfg.get("datefmt", "%H:%M:%S"),
    )


def _load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def run(cfg: dict, debug: bool, demo: bool = False) -> None:
    face_cfg    = cfg.get("face_mesh", {})
    capture_cfg = cfg.get("capture", {})
    overlay_cfg = cfg.get("overlay", {})
    signals_cfg = cfg.get("signals", {})
    hp_cfg      = cfg.get("head_pose", {})

    log = logging.getLogger(__name__)

    print_camera_candidates(capture_cfg)

    blink_det     = BlinkDetector(signals_cfg)
    attention_det = AttentionDetector(cfg)
    gaze_det      = GazeDetector(cfg)
    scorer        = AttentionScorer(cfg)

    focal_scale = float(hp_cfg.get("focal_scale", 1.0))
    max_reproj  = float(hp_cfg.get("max_reprojection_error", 10.0))

    frame_times: deque[float] = deque(maxlen=30)
    no_face_streak = 0
    cam_matrix     = None
    last_blink_a   = None  # retained for session summary

    with FacePipeline(face_cfg, capture_cfg) as pipeline:
        log.info("Attention scoring running.  Press Q or Esc to quit.")

        while True:
            result = pipeline.process_latest(timeout=0.1)
            if result is None:
                continue

            frame_times.append(result.timestamp_process)
            fps = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) >= 2 else 0.0
            )

            h, w  = result.frame_raw.shape[:2]
            display = result.frame_raw.copy()

            if cam_matrix is None:
                cam_matrix = build_camera_matrix(w, h, focal_scale=focal_scale)

            if result.face_detected:
                no_face_streak = 0
                lm = result.face_landmarks[0]
                ts = result.timestamp_process
                fi = result.frame_index

                # ── EAR / blink ───────────────────────────────────────────────
                eye_m = extract_eye_measurements(
                    lm, w, h, timestamp=ts, frame_index=fi, cfg=signals_cfg,
                )
                blink_a = (
                    blink_det.update(eye_m) if eye_m is not None
                    else blink_det.no_face_update()
                )
                last_blink_a = blink_a

                # ── Head pose / attention ─────────────────────────────────────
                pose = estimate_head_pose(
                    lm, w, h, cam_matrix, timestamp=ts, frame_index=fi,
                )
                att_a = (
                    attention_det.update(pose) if pose is not None
                    else attention_det.no_pose_update()
                )

                # ── Gaze ──────────────────────────────────────────────────────
                gaze_m = extract_gaze_measurements(
                    lm, w, h,
                    mean_ear=eye_m.mean_ear if eye_m is not None else 0.0,
                    timestamp=ts, frame_index=fi, cfg=cfg.get("gaze"),
                )
                if gaze_m is not None:
                    gaze_a = gaze_det.update(
                        gaze_m,
                        head_zone=(att_a.zone.value if att_a is not None else None),
                        head_yaw= (pose.yaw   if pose is not None else None),
                        head_pitch=(pose.pitch if pose is not None else None),
                    )
                else:
                    gaze_a = gaze_det.no_gaze_update()

                # ── Composite engagement score ─────────────────────────────────
                eng = scorer.update(
                    blink=blink_a,
                    head=att_a,
                    gaze=gaze_a,
                    face_detected=True,
                    timestamp=ts,
                    frame_index=fi,
                )

                if debug:
                    log.debug(
                        "frame=%d  state=%-12s  score=%.3f  conf=%.3f  "
                        "focused=%.0f%%  engaged=%.0f%%",
                        fi, eng.state.value, eng.smoothed_score, eng.confidence,
                        eng.focused_fraction * 100, eng.engaged_fraction * 100,
                    )

                # ── Rendering ─────────────────────────────────────────────────
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

                if not demo and pose is not None:
                    display = draw_head_axes(display, pose.rvec, pose.tvec, cam_matrix)

                if demo:
                    # ── Demo mode: single clean card, no raw signal panels ─
                    display = draw_demo_overlay(
                        display, eng, blink_a, origin=(10, 20),
                    )
                else:
                    # ── Developer mode: full dashboard + all side panels ───
                    display = draw_engagement_dashboard(
                        display, eng, blink_a, origin=(10, 20),
                    )
                    rhs = w - 280
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
                    if debug:
                        # ── Debug mode: confidence diagnostics panel ───────
                        display = draw_diagnostics_panel(
                            display, eng, blink_a, origin=(10, 340),
                        )

            else:
                no_face_streak += 1
                if no_face_streak >= 30:
                    blink_det.reset()
                    attention_det.reset()
                    gaze_det.reset()
                    scorer.reset()
                    no_face_streak = 0

                eng_absent = scorer.update(
                    face_detected=False,
                    timestamp=result.timestamp_process,
                    frame_index=result.frame_index,
                )
                if demo:
                    display = draw_demo_overlay(display, eng_absent, origin=(10, 20))
                else:
                    display = draw_engagement_dashboard(
                        display, eng_absent, origin=(10, 20),
                    )
                display = draw_status_text(display, "No face detected")

            display = draw_fps_counter(display, fps)
            cv2.imshow("Engagement Scoring  |  Q / Esc to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cv2.destroyAllWindows()

    # ── Session summary ───────────────────────────────────────────────────────
    final_gaze  = gaze_det.no_gaze_update()
    final_att   = attention_det.no_pose_update()
    final_eng   = scorer.update(face_detected=False, timestamp=0.0)

    print(f"\n── Session summary ─────────────────────────────────────")
    print(f"  Engagement state      : {final_eng.state.value}")
    print(f"  Smoothed score        : {final_eng.smoothed_score:.3f}")
    print(f"  Confidence            : {final_eng.confidence:.3f}")
    print(f"  Focused fraction      : {final_eng.focused_fraction * 100:.1f}%")
    print(f"  Engaged fraction      : {final_eng.engaged_fraction * 100:.1f}%")
    if final_gaze:
        print(f"  Last gaze zone        : {final_gaze.zone.value}")
        print(f"  On-screen fraction    : {final_gaze.on_screen_fraction * 100:.1f}%")
    if final_att:
        print(f"  Head attention zone   : {final_att.zone.value}")
        print(f"  Head attention frac   : {final_att.attention_fraction * 100:.1f}%")
    if last_blink_a is not None:
        print(f"  Total blinks          : {last_blink_a.blink_count}")
        print(f"  Blink rate            : {last_blink_a.blink_rate_per_min:.1f} / min")
        print(f"  Fatigue indicator     : {last_blink_a.fatigue_label}")
    print(f"────────────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite engagement scoring — Day 6")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--debug",  action="store_true",
                        help="Developer mode + diagnostics panel")
    parser.add_argument("--demo",   action="store_true",
                        help="Presentation mode: clean minimal overlay, "
                             "hides raw EAR/gaze ratios/yaw-pitch/reprojection")
    parser.add_argument(
        "--device", type=int, default=None, metavar="N",
        help="Camera device index (overrides capture.device_id in config)",
    )
    parser.add_argument(
        "--backend", default=None, metavar="NAME",
        help="Camera backend: avfoundation | any | qt | dshow | v4l2",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.device is not None:
        cfg.setdefault("capture", {})["device_id"] = args.device
    if args.backend is not None:
        cfg.setdefault("capture", {})["backend"] = args.backend

    _setup_logging(cfg.get("logging", {}), args.debug)
    run(cfg, debug=args.debug, demo=args.demo)


if __name__ == "__main__":
    main()
