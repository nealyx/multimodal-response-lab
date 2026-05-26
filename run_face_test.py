#!/usr/bin/env python3
"""Webcam face mesh visualiser — Day 2 integration test.

Runs the full capture → inference → overlay pipeline in a live window.
Press Q or Escape to quit.

Usage
-----
    python run_face_test.py
    python run_face_test.py --config configs/default.yaml --debug

What this exercises
-------------------
  - CaptureThread writes fresh frames into FrameBuffer at webcam FPS
  - FacePipeline.process_latest() runs MediaPipe on the newest frame
  - draw_face_mesh / draw_fps_counter compose overlays without mutating
    the captured frame
  - Rolling FPS is computed from real wall-clock timestamps, not assumed fps

What this does NOT test
-----------------------
  - State estimation (blinks, gaze direction, expressions) — Day 3+
  - Multi-face tracking (max_faces=1 in default config)
  - Recording or export of landmarks
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from pathlib import Path

import cv2
import yaml

# Make `src.*` importable when the script is run from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines.face_pipeline import FacePipeline
from src.utils.camera import print_camera_candidates
from src.utils.overlay import (
    draw_bounding_box,
    draw_face_mesh,
    draw_fps_counter,
    draw_status_text,
)


def _setup_logging(cfg: dict, debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else getattr(logging, cfg.get("level", "INFO").upper()),
        format=cfg.get("format", "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"),
        datefmt=cfg.get("datefmt", "%H:%M:%S"),
    )


def _load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def run(cfg: dict, debug: bool, debug_vision: bool) -> None:
    face_cfg    = cfg.get("face_mesh", {})
    capture_cfg = cfg.get("capture", {})
    overlay_cfg = cfg.get("overlay", {})

    print_camera_candidates(capture_cfg)

    log = logging.getLogger(__name__)

    # Rolling window of perf_counter timestamps for FPS smoothing.
    # 30 samples ≈ 1 s of history at 30 FPS — long enough to be smooth,
    # short enough to reflect sudden drops quickly.
    frame_times: deque[float] = deque(maxlen=30)
    total_frames   = 0
    total_no_face  = 0

    with FacePipeline(face_cfg, capture_cfg, debug_mode=debug_vision) as pipeline:
        log.info("Display loop started. Press Q or Esc to quit.")

        while True:
            result = pipeline.process_latest(timeout=0.1)
            if result is None:
                continue

            total_frames += 1
            frame_times.append(result.timestamp_process)

            fps = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) >= 2
                else 0.0
            )

            # Start from the raw captured frame; layer overlays on top.
            display = result.frame_raw.copy()

            if result.face_detected:
                for lm in result.face_landmarks:
                    display = draw_face_mesh(
                        display, lm,
                        draw_tesselation=overlay_cfg.get("draw_tesselation", True),
                        draw_contour=    overlay_cfg.get("draw_contour",     True),
                        draw_eyes=       overlay_cfg.get("draw_eyes",        True),
                        draw_lips=       overlay_cfg.get("draw_lips",        True),
                        draw_irises=     overlay_cfg.get("draw_irises",      False),
                    )
                    if overlay_cfg.get("show_bbox", False):
                        display = draw_bounding_box(display, lm)
            else:
                total_no_face += 1
                display = draw_status_text(display, "No face detected")

            if overlay_cfg.get("show_fps", True):
                display = draw_fps_counter(display, fps)

            if debug:
                log.debug(
                    "frame=%d  fps=%.1f  faces=%d  latency=%.1f ms",
                    result.frame_index, fps, result.face_count, result.latency_ms,
                )

            cv2.imshow("Face Mesh  |  Q / Esc to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cv2.destroyAllWindows()

    # ── Session summary ───────────────────────────────────────────────────────
    if total_frames > 0:
        detection_pct = (total_frames - total_no_face) / total_frames * 100
        print(f"\n── Session summary ─────────────────────────────")
        print(f"  Frames processed : {total_frames}")
        print(f"  Detection rate   : {detection_pct:.1f}%")
        print(f"  No-face frames   : {total_no_face}")
        print(f"────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="Face mesh visualiser / Day 2 test")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-vision", action="store_true")
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

    _setup_logging(cfg.get("logging", {}), args.debug or args.debug_vision)

    run(cfg, debug=args.debug, debug_vision=args.debug_vision)


if __name__ == "__main__":
    main()
