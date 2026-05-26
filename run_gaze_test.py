#!/usr/bin/env python3
"""Live gaze estimation test — Day 5.

Runs the full pipeline:
  capture → face detection → EAR → blink FSM
                           → head pose → attention zone
                           → iris position → gaze zone → on-screen status

Displays iris markers (cyan dots on each eye's iris centre), gaze HUD,
head pose HUD, and blink metrics simultaneously.

Usage
-----
    python run_gaze_test.py
    python run_gaze_test.py --device 1 --debug

Press Q or Esc to quit.

What to watch for when validating
-----------------------------------
  - Two small cyan dots should sit on your irises.  They turn red when your
    eyes are too closed for a reliable reading.
  - Looking straight at the camera → Gaze: CENTER, Screen: ON SCREEN.
  - Looking left/right → zone changes; iris dots shift toward the corners.
  - Looking up/down → UP/DOWN zone; dots shift vertically.
  - Closing eyes for >0.3 s → zone switches to UNRELIABLE.
  - Combined head turn + centred iris → still marked OFF SCREEN because the
    head pose gate overrides the iris reading.
  - On-screen % (last 5 s) should reach ~90–100% during normal screen use.
  - H and V std-dev should be low (<0.05) during still fixation, rising
    during active gaze shifts.
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
from src.signals.eye_metrics import extract_eye_measurements
from src.signals.gaze import GazeDetector, extract_gaze_measurements
from src.signals.head_pose import build_camera_matrix, estimate_head_pose
from src.utils.camera import print_camera_candidates
from src.utils.overlay import (
    draw_attention_metrics,
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


def run(cfg: dict, debug: bool) -> None:
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

    focal_scale = float(hp_cfg.get("focal_scale", 1.0))
    max_reproj  = float(hp_cfg.get("max_reprojection_error", 10.0))

    frame_times: deque[float] = deque(maxlen=30)
    no_face_streak = 0
    cam_matrix     = None

    with FacePipeline(face_cfg, capture_cfg) as pipeline:
        log.info("Gaze test running.  Press Q or Esc to quit.")

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

                if debug:
                    if gaze_m is not None:
                        log.debug(
                            "frame=%d  H=%.2f  V=%.2f  zone=%s  on_screen=%s  "
                            "reliable=%s",
                            fi, gaze_m.mean_h, gaze_m.mean_v,
                            gaze_a.zone.value if gaze_a else "?",
                            gaze_a.is_on_screen if gaze_a else "?",
                            gaze_m.is_reliable,
                        )
                    if pose is not None and pose.reprojection_error > max_reproj:
                        log.debug("  ↳ high reprojection: %.1f px", pose.reprojection_error)

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

                if pose is not None:
                    display = draw_head_axes(display, pose.rvec, pose.tvec, cam_matrix)

                if att_a is not None:
                    display = draw_attention_metrics(display, att_a, origin=(10, 60))

                if gaze_a is not None:
                    display = draw_gaze_metrics(display, gaze_a, origin=(10, 200))

                if blink_a is not None and eye_m is not None:
                    display = draw_eye_metrics(
                        display, blink_a,
                        ear_l=eye_m.left_ear, ear_r=eye_m.right_ear,
                        origin=(10, 380),
                    )

            else:
                no_face_streak += 1
                if no_face_streak >= 30:
                    blink_det.reset()
                    attention_det.reset()
                    gaze_det.reset()
                    no_face_streak = 0
                display = draw_status_text(display, "No face detected")

            display = draw_fps_counter(display, fps)
            cv2.imshow("Gaze Estimation  |  Q / Esc to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cv2.destroyAllWindows()

    # ── Session summary ───────────────────────────────────────────────────────
    final_gaze  = gaze_det.no_gaze_update()
    final_att   = attention_det.no_pose_update()
    final_blink = blink_det._last_analysis
    if final_gaze or final_att or final_blink:
        print(f"\n── Session summary ─────────────────────────────────────")
        if final_gaze:
            print(f"  Last gaze zone        : {final_gaze.zone.value}")
            print(f"  On-screen fraction    : {final_gaze.on_screen_fraction * 100:.1f}%")
            print(f"  Gaze stability H-std  : {final_gaze.stability_h:.3f}")
            print(f"  Gaze stability V-std  : {final_gaze.stability_v:.3f}")
        if final_att:
            print(f"  Head attention zone   : {final_att.zone.value}")
            print(f"  Head attention frac   : {final_att.attention_fraction * 100:.1f}%")
        if final_blink:
            print(f"  Total blinks          : {final_blink.blink_count}")
            print(f"  Blink rate            : {final_blink.blink_rate_per_min:.1f} / min")
            print(f"  Fatigue indicator     : {final_blink.fatigue_label}")
        print(f"────────────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live gaze estimation — Day 5")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--debug", action="store_true")
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
    run(cfg, debug=args.debug)


if __name__ == "__main__":
    main()
