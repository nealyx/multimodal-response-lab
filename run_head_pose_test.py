#!/usr/bin/env python3
"""Live head pose estimation test — Day 4.

Runs the full capture → face detection → EAR extraction → blink FSM →
head pose estimation → attention zone pipeline, displaying all metrics
and a 3-D axis overlay on the face.

Usage
-----
    python run_head_pose_test.py
    python run_head_pose_test.py --config configs/default.yaml --debug

Press Q or Esc to quit.

What to watch for when validating
-----------------------------------
  - 3 coloured arrows should emerge from the nose tip:
      red  = X axis (pointing right from camera's view)
      green = Y axis (pointing down in image)
      blue  = Z axis (pointing toward camera — out of the screen)
  - When looking straight ahead: yaw ≈ 0°, pitch ≈ 0°, roll ≈ 0°.
  - Turn head left → yaw goes negative; turn right → yaw goes positive.
  - Tilt chin down → pitch increases; tilt back → pitch decreases.
  - Tilt head sideways (right ear toward shoulder) → roll increases.
  - Zone badge should read FOCUSED when looking at screen,
    GLANCE when slightly off-centre, LOOKING_AWAY when turned far away.
  - Reprojection error should be < 5 px for a well-detected face;
    values > 10 px indicate noisy landmarks or poor lighting.
  - Stability std devs should be < 3° during still gaze,
    rising to > 8° during active head movement.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines.face_pipeline import FacePipeline
from src.signals.blink_detector import BlinkDetector
from src.signals.eye_metrics import extract_eye_measurements
from src.signals.head_pose import build_camera_matrix, estimate_head_pose
from src.signals.attention import AttentionDetector
from src.utils.camera import print_camera_candidates
from src.utils.overlay import (
    draw_attention_metrics,
    draw_bounding_box,
    draw_eye_metrics,
    draw_face_mesh,
    draw_fps_counter,
    draw_head_axes,
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
    blink_detector     = BlinkDetector(signals_cfg)
    attention_detector = AttentionDetector(cfg)

    max_reproj = float(hp_cfg.get("max_reprojection_error", 10.0))
    focal_scale = float(hp_cfg.get("focal_scale", 1.0))

    frame_times: deque[float] = deque(maxlen=30)
    no_face_streak = 0

    with FacePipeline(face_cfg, capture_cfg) as pipeline:
        log.info("Head pose test running.  Press Q or Esc to quit.")
        cam_matrix = None   # built on first frame (need actual w, h)

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

            # Build camera matrix once (size is constant for a given capture config)
            if cam_matrix is None:
                cam_matrix = build_camera_matrix(w, h, focal_scale=focal_scale)

            if result.face_detected:
                no_face_streak = 0
                lm = result.face_landmarks[0]

                # ── Blink / EAR ───────────────────────────────────────────────
                measurement = extract_eye_measurements(
                    lm, w, h,
                    timestamp=result.timestamp_process,
                    frame_index=result.frame_index,
                    cfg=signals_cfg,
                )
                blink_analysis = (
                    blink_detector.update(measurement)
                    if measurement is not None
                    else blink_detector.no_face_update()
                )

                # ── Head pose ─────────────────────────────────────────────────
                pose = estimate_head_pose(
                    lm, w, h, cam_matrix,
                    timestamp=result.timestamp_process,
                    frame_index=result.frame_index,
                )
                att_analysis = (
                    attention_detector.update(pose)
                    if pose is not None
                    else attention_detector.no_pose_update()
                )

                if debug and pose is not None:
                    log.debug(
                        "frame=%d  yaw=%+.1f  pitch=%+.1f  roll=%+.1f  "
                        "reproj=%.1fpx  zone=%s",
                        result.frame_index, pose.yaw, pose.pitch, pose.roll,
                        pose.reprojection_error,
                        att_analysis.zone.value if att_analysis else "?",
                    )
                    if pose.reprojection_error > max_reproj:
                        log.debug("  ↳ high reprojection error: %.1f px", pose.reprojection_error)

                # ── Rendering ─────────────────────────────────────────────────
                display = draw_face_mesh(
                    display, lm,
                    draw_tesselation=overlay_cfg.get("draw_tesselation", False),
                    draw_contour=    overlay_cfg.get("draw_contour",     True),
                    draw_eyes=       overlay_cfg.get("draw_eyes",        True),
                    draw_lips=       overlay_cfg.get("draw_lips",        False),
                    draw_irises=     overlay_cfg.get("draw_irises",      False),
                )

                if pose is not None:
                    display = draw_head_axes(
                        display, pose.rvec, pose.tvec, cam_matrix,
                        axis_length=60.0,
                    )

                if att_analysis is not None:
                    display = draw_attention_metrics(
                        display, att_analysis, origin=(10, 60),
                    )

                if blink_analysis is not None and measurement is not None:
                    display = draw_eye_metrics(
                        display, blink_analysis,
                        ear_l=measurement.left_ear,
                        ear_r=measurement.right_ear,
                        origin=(10, 240),
                    )

            else:
                no_face_streak += 1
                if no_face_streak >= 30:
                    blink_detector.reset()
                    attention_detector.reset()
                    no_face_streak = 0
                display = draw_status_text(display, "No face detected")

            display = draw_fps_counter(display, fps)
            cv2.imshow("Head Pose  |  Q / Esc to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cv2.destroyAllWindows()

    # ── Session summary ───────────────────────────────────────────────────────
    final_att   = attention_detector.no_pose_update()
    final_blink = blink_detector._last_analysis
    if final_att or final_blink:
        print(f"\n── Session summary ─────────────────────────────────────")
        if final_att:
            print(f"  Last attention zone   : {final_att.zone.value}")
            print(f"  Attention fraction    : {final_att.attention_fraction * 100:.1f}%")
            print(f"  Yaw stability std     : {final_att.yaw_std:.1f}°")
            print(f"  Pitch stability std   : {final_att.pitch_std:.1f}°")
        if final_blink:
            print(f"  Total blinks detected : {final_blink.blink_count}")
            print(f"  Final blink rate      : {final_blink.blink_rate_per_min:.1f} / min")
            print(f"  Fatigue indicator     : {final_blink.fatigue_label}")
        print(f"────────────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live head pose estimation — Day 4")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--device", type=int, default=None,
        metavar="N",
        help="Camera device index (overrides capture.device_id in config)",
    )
    parser.add_argument(
        "--backend", default=None,
        metavar="NAME",
        help="Camera backend: avfoundation | any | qt | dshow | v4l2 "
             "(overrides capture.backend in config)",
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
