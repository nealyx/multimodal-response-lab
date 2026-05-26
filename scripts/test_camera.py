#!/usr/bin/env python3
"""Minimal camera acquisition test — no MediaPipe, no pipeline.

Verifies that the camera opens, warms up, and delivers valid frames using
only the camera module.  Run this first when diagnosing capture issues to
confirm the fix before bringing the full pipeline back up.

Usage
-----
    python scripts/test_camera.py
    python scripts/test_camera.py --device 1 --backend avfoundation
    python scripts/test_camera.py --backend any --warmup 50
    python scripts/test_camera.py --no-display      # headless; just prints stats

Press Q or Esc to quit the preview window.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.camera import is_valid_frame, open_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera acquisition smoke test")
    parser.add_argument("--device",     type=int,   default=0,     help="Camera device index")
    parser.add_argument("--backend",    type=str,   default="auto",help="Backend: auto|avfoundation|any|qt")
    parser.add_argument("--resolution", type=str,   default="640x480", help="WxH e.g. 640x480")
    parser.add_argument("--warmup",     type=int,   default=30,    help="Max warm-up frames")
    parser.add_argument("--timeout",    type=float, default=6.0,   help="Warm-up timeout (s)")
    parser.add_argument("--no-display", action="store_true",       help="Skip preview window")
    args = parser.parse_args()

    w, h = (int(x) for x in args.resolution.split("x"))

    capture_cfg = {
        "device_id":       args.device,
        "backend":         args.backend,
        "resolution":      [w, h],
        "warmup_frames":   args.warmup,
        "warmup_timeout_s":args.timeout,
        "min_frame_mean":  1.0,
    }

    print(f"Opening camera: device={args.device}  backend={args.backend}  {w}x{h}")
    t_open_start = time.perf_counter()

    try:
        cap = open_camera(capture_cfg)
    except RuntimeError as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)

    open_ms = (time.perf_counter() - t_open_start) * 1000
    print(f"Camera ready in {open_ms:.0f} ms")

    if args.no_display:
        # Headless: read 60 frames and report stats.
        valid = blank = 0
        means: list[float] = []
        for _ in range(60):
            ret, frame = cap.read()
            if ret and is_valid_frame(frame):
                valid += 1
                means.append(float(frame.mean()))
            else:
                blank += 1
        cap.release()
        avg_mean = sum(means) / len(means) if means else 0.0
        print(f"60-frame headless test: valid={valid}  blank={blank}  avg_mean={avg_mean:.1f}")
        print("PASS" if valid > 55 else "WARN: unexpectedly high blank rate")
        return

    # Live preview
    frame_times: deque[float] = deque(maxlen=30)
    valid_total = blank_total = 0

    print("Live preview running. Press Q or Esc to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if not is_valid_frame(frame):
            blank_total += 1
            continue

        valid_total += 1
        now = time.perf_counter()
        frame_times.append(now)

        fps = (
            (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            if len(frame_times) >= 2 else 0.0
        )

        display = frame.copy()
        mean_val = float(frame.mean())

        lines = [
            f"backend: {args.backend}  device: {args.device}",
            f"FPS: {fps:.1f}",
            f"frame mean: {mean_val:.1f}",
            f"valid: {valid_total}  blank: {blank_total}",
        ]
        for i, line in enumerate(lines):
            y = 26 + i * 22
            cv2.putText(display, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, (0, 0, 0),   3, cv2.LINE_AA)
            cv2.putText(display, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, (0, 255, 100), 1, cv2.LINE_AA)

        cv2.imshow("Camera Test  |  Q / Esc to quit", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    blank_pct = blank_total / max(1, valid_total + blank_total) * 100
    print(f"\nSession: valid={valid_total}  blank={blank_total}  blank_rate={blank_pct:.1f}%")
    if blank_pct > 5:
        print("WARN: >5% blank frames — consider increasing warmup_frames in config.")
    else:
        print("PASS")


if __name__ == "__main__":
    main()
