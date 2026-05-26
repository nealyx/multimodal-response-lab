#!/usr/bin/env python3
"""Live blink detection test — Day 3.

Runs the full capture → face detection → EAR extraction → FSM pipeline
and displays per-frame eye metrics alongside the face mesh overlay.

Usage
-----
    python run_blink_test.py
    python run_blink_test.py --config configs/default.yaml --debug

Press Q or Esc to quit.

What to watch for when validating
-----------------------------------
  - EAR should be ~0.25–0.35 when your eyes are open.
  - EAR should drop to ~0.05–0.15 when you blink.
  - Blink count increments once per blink, not multiple times.
  - Blink rate approaches 15–20/min after ~1 minute of normal behaviour.
  - The FSM state badge should read CLOSING → CLOSED → OPENING → OPEN
    over the course of one blink, typically within 3–6 frames at 25 FPS.
  - Deliberately stare without blinking for 20 s → 'LOW-BLINK' flag appears.
  - Deliberately close your eyes for > 0.5 s → 'PROLONGED' flag appears.
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
from src.utils.camera import print_camera_candidates
from src.utils.overlay import (
    draw_bounding_box,
    draw_eye_metrics,
    draw_face_mesh,
    draw_fps_counter,
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

    log = logging.getLogger(__name__)

    print_camera_candidates(capture_cfg)
    detector = BlinkDetector(signals_cfg)

    # Rolling FPS window
    frame_times: deque[float] = deque(maxlen=30)
    total_frames   = 0
    no_face_streak = 0     # consecutive no-face frames; reset FSM if too long

    with FacePipeline(face_cfg, capture_cfg) as pipeline:
        log.info("Blink test running.  Press Q or Esc to quit.")

        while True:
            result = pipeline.process_latest(timeout=0.1)
            if result is None:
                continue

            total_frames += 1
            frame_times.append(result.timestamp_process)
            fps = (
                (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                if len(frame_times) >= 2 else 0.0
            )

            h, w = result.frame_raw.shape[:2]
            display = result.frame_raw.copy()

            if result.face_detected:
                no_face_streak = 0
                lm = result.face_landmarks[0]   # first face only

                # ── Extract raw measurements ──────────────────────────────────
                measurement = extract_eye_measurements(
                    lm, w, h,
                    timestamp=result.timestamp_process,
                    frame_index=result.frame_index,
                    cfg=signals_cfg,
                )

                if measurement is not None:
                    analysis = detector.update(measurement)
                else:
                    analysis = detector.no_face_update()

                # ── Render face mesh ──────────────────────────────────────────
                display = draw_face_mesh(
                    display, lm,
                    draw_tesselation=overlay_cfg.get("draw_tesselation", False),
                    draw_contour=    overlay_cfg.get("draw_contour",     True),
                    draw_eyes=       overlay_cfg.get("draw_eyes",        True),
                    draw_lips=       overlay_cfg.get("draw_lips",        False),
                    draw_irises=     overlay_cfg.get("draw_irises",      False),
                )

                # ── Eye metrics HUD ───────────────────────────────────────────
                if analysis is not None and measurement is not None:
                    display = draw_eye_metrics(
                        display, analysis,
                        ear_l=measurement.left_ear,
                        ear_r=measurement.right_ear,
                        origin=(10, 60),
                    )

                if debug and measurement is not None:
                    log.debug(
                        "frame=%d  EAR=%.3f  state=%s  blinks=%d  rate=%.1f/min",
                        result.frame_index,
                        measurement.mean_ear,
                        analysis.state.value if analysis else "?",
                        analysis.blink_count if analysis else 0,
                        analysis.blink_rate_per_min if analysis else 0.0,
                    )

            else:
                no_face_streak += 1
                # After 30 consecutive missing frames (~1 s), reset FSM to
                # prevent stale closure state when face reappears.
                if no_face_streak >= 30:
                    detector.reset()
                    no_face_streak = 0
                analysis = detector.no_face_update()
                display  = draw_status_text(display, "No face detected")

            display = draw_fps_counter(display, fps)

            cv2.imshow("Blink Detection  |  Q / Esc to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cv2.destroyAllWindows()

    # ── Session summary ───────────────────────────────────────────────────────
    final = detector._last_analysis
    if final:
        print(f"\n── Session summary ─────────────────────────────────────")
        print(f"  Total blinks detected : {final.blink_count}")
        print(f"  Final blink rate      : {final.blink_rate_per_min:.1f} / min")
        print(f"  Final EAR             : {final.mean_ear:.3f}")
        print(f"  Fatigue indicator     : {final.fatigue_label}")
        print(f"────────────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live blink detection — Day 3")
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
