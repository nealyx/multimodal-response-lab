#!/usr/bin/env python3
"""Standalone face detection diagnostic.

Tests each failure hypothesis in isolation without going through the
FacePipeline abstraction.  Run this when the live visualiser shows 0 detections
to find exactly which step in the pipeline breaks.

Usage
-----
    python scripts/debug_face_detection.py
    python scripts/debug_face_detection.py --device 1     # try a different camera
    python scripts/debug_face_detection.py --no-flip      # skip horizontal flip
    python scripts/debug_face_detection.py --threshold 0.1  # lower confidence

What each hypothesis tests
--------------------------
H1  Camera warm-up: first frames are black/blank.
    Fix: discard leading blank frames; wait longer before first inference.

H2  Frame format: camera outputs unexpected shape, dtype, or channel order.
    Fix: verify frame.shape == (H, W, 3), dtype == uint8, not all-zero.

H3  Array contiguity: cv2.flip/cvtColor produces non-C-contiguous array.
    Fix: np.ascontiguousarray(rgb) before mp.Image().
    (Unlikely given our test above, but confirmed here.)

H4  IMAGE mode works but VIDEO mode doesn't (timestamp issue).
    Fix: use IMAGE mode, or ensure timestamps are strictly increasing and > 0.

H5  Confidence threshold: face is at the edge of detection range.
    Fix: lower min_detection_confidence and min_presence_confidence to 0.1.

H6  Color channel order: BGR passed as if it were RGB (or vice versa).
    Fix: verify cvtColor conversion produces the correct channel order.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.frame_debug import check_mediapipe_input, frame_info, save_debug_frame

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = ".cache/models/face_landmarker.task"
DUMP_DIR     = "outputs/debug_frames"
WARMUP_FRAMES = 15    # discard this many frames before testing
N_TEST_FRAMES = 10    # frames to test per hypothesis


def _open_camera(device_id: int, w: int = 640, h: int = 480) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"  Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


def _grab_frames(cap: cv2.VideoCapture, n: int, skip: int = 0) -> list[np.ndarray]:
    """Read *skip* frames silently, then collect *n* frames."""
    for _ in range(skip):
        cap.read()
    frames = []
    for _ in range(n):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames


def _make_landmarker(running_mode: RunningMode, threshold: float = 0.5):
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=threshold,
        min_face_presence_confidence=threshold,
        min_tracking_confidence=threshold,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# ── Hypothesis tests ──────────────────────────────────────────────────────────

def h1_camera_warmup(cap: cv2.VideoCapture) -> None:
    """H1: Are early frames blank, improving after warm-up?"""
    print("\n── H1: Camera warm-up ──────────────────────────────────────────────")
    frames_early = _grab_frames(cap, n=3, skip=0)
    frames_late  = _grab_frames(cap, n=3, skip=20)

    for i, f in enumerate(frames_early):
        fi = frame_info(f, label=f"early_{i}")
        print(f"  early frame {i}: all_zero={fi['all_zero']}  "
              f"mean={fi['mean_per_ch']}  max={fi['max']}")

    for i, f in enumerate(frames_late):
        fi = frame_info(f, label=f"late_{i}")
        print(f"  late  frame {i}: all_zero={fi['all_zero']}  "
              f"mean={fi['mean_per_ch']}  max={fi['max']}")

    late_maxes = [frame_info(f)["max"] for f in frames_late]
    if all(m < 5 for m in late_maxes):
        print("  FAIL: Frames are still blank after warm-up.")
        print("  → Check camera permissions (System Preferences → Privacy → Camera).")
        print("  → Try --device 1 if you have multiple cameras.")
    else:
        print("  PASS: Frames have non-trivial pixel values after warm-up.")


def h2_frame_format(cap: cv2.VideoCapture) -> None:
    """H2: Does the frame have the expected shape, dtype, and channels?"""
    print("\n── H2: Frame format ────────────────────────────────────────────────")
    frames = _grab_frames(cap, n=3, skip=10)
    if not frames:
        print("  SKIP: No frames captured.")
        return

    f  = frames[-1]
    fi = frame_info(f, label="bgr_check")
    print(f"  shape={fi['shape']}  dtype={fi['dtype']}  "
          f"C_contiguous={fi['C_contiguous']}")
    print(f"  channel means (B G R): {fi['mean_per_ch']}")

    if f.ndim != 3 or f.shape[2] != 3:
        print("  FAIL: Unexpected channel count.")
    elif f.dtype != np.uint8:
        print("  FAIL: Unexpected dtype.")
    elif fi["all_zero"]:
        print("  FAIL: Frame is all zeros — camera not producing data.")
    else:
        print("  PASS: Shape, dtype, and channels look correct.")

    # Save to disk for visual inspection.
    path = save_debug_frame(f, DUMP_DIR, label="h2_bgr_raw", add_info=True)
    print(f"  Saved: {path}")


def h3_array_contiguity(cap: cv2.VideoCapture, flip: bool = True) -> None:
    """H3: Does flip + cvtColor break C-contiguity?"""
    print("\n── H3: Array contiguity after preprocessing ────────────────────────")
    frames = _grab_frames(cap, n=2, skip=10)
    if not frames:
        print("  SKIP: No frames captured.")
        return

    frame = frames[-1]
    if flip:
        frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    warnings = check_mediapipe_input(rgb, label="h3_rgb")

    print(f"  After flip+cvtColor: C_contiguous={rgb.flags['C_CONTIGUOUS']}  "
          f"dtype={rgb.dtype}  shape={rgb.shape}")

    if warnings:
        print(f"  FAIL: {warnings}")
        print("  → Fix: pass np.ascontiguousarray(rgb) to mp.Image")
    else:
        print("  PASS: Array is suitable for mp.Image.")


def h4_image_vs_video_mode(cap: cv2.VideoCapture, flip: bool = True) -> None:
    """H4: Does IMAGE mode detect faces even though VIDEO mode doesn't?"""
    print("\n── H4: IMAGE mode vs VIDEO mode ────────────────────────────────────")
    frames = _grab_frames(cap, n=N_TEST_FRAMES, skip=WARMUP_FRAMES)
    if not frames:
        print("  SKIP: No frames captured.")
        return

    def _to_mp_image(bgr: np.ndarray) -> mp.Image:
        if flip:
            bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), bgr

    # IMAGE mode
    image_hits = 0
    with _make_landmarker(RunningMode.IMAGE) as lm:
        for bgr in frames:
            mp_img, _ = _to_mp_image(bgr)
            result = lm.detect(mp_img)
            if result.face_landmarks:
                image_hits += 1

    print(f"  IMAGE mode: {image_hits}/{len(frames)} frames detected a face.")

    # VIDEO mode
    video_hits = 0
    t0 = time.perf_counter()
    with _make_landmarker(RunningMode.VIDEO) as lm:
        for bgr in frames:
            mp_img, _ = _to_mp_image(bgr)
            ts = max(1, int((time.perf_counter() - t0) * 1000))
            result = lm.detect_for_video(mp_img, ts)
            if result.face_landmarks:
                video_hits += 1
            time.sleep(0.033)  # simulate 30 FPS spacing between frames

    print(f"  VIDEO mode: {video_hits}/{len(frames)} frames detected a face.")

    if image_hits > 0 and video_hits == 0:
        print("  DIAGNOSIS: IMAGE works, VIDEO doesn't → timestamp problem.")
        print("  → Fix applied below: timestamps now start at 1ms, not 0ms.")
    elif image_hits == 0 and video_hits == 0:
        print("  DIAGNOSIS: Neither mode detects a face.")
        print("  → Check H5 (thresholds) and H6 (color channels).")
    elif image_hits > 0 and video_hits > 0:
        print("  PASS: Both modes work — pipeline wiring issue, not model/format.")
    else:
        print(f"  Unexpected: IMAGE={image_hits} VIDEO={video_hits}")


def h5_confidence_thresholds(cap: cv2.VideoCapture, flip: bool = True) -> None:
    """H5: Does lowering thresholds to 0.1 unlock detections?"""
    print("\n── H5: Confidence threshold sensitivity ────────────────────────────")
    frames = _grab_frames(cap, n=N_TEST_FRAMES, skip=WARMUP_FRAMES)
    if not frames:
        print("  SKIP: No frames captured.")
        return

    for threshold in (0.5, 0.3, 0.1):
        hits = 0
        with _make_landmarker(RunningMode.IMAGE, threshold=threshold) as lm:
            for bgr in frames:
                if flip:
                    bgr = cv2.flip(bgr, 1)
                rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = lm.detect(mp_img)
                if result.face_landmarks:
                    hits += 1
        print(f"  threshold={threshold:.1f}: {hits}/{len(frames)} detections")

    if hits > 0:
        print("  DIAGNOSIS: Lower thresholds unlock detections.")
        print("  → Set min_detection_confidence / min_presence_confidence to 0.3 in config.")
    else:
        print("  PASS: Thresholds are not the blocker (0.1 still finds nothing).")


def h6_color_channels(cap: cv2.VideoCapture, flip: bool = True) -> None:
    """H6: Does swapping color channels (testing wrong BGR→RGB path) matter?"""
    print("\n── H6: Color channel order ─────────────────────────────────────────")
    frames = _grab_frames(cap, n=5, skip=WARMUP_FRAMES)
    if not frames:
        print("  SKIP: No frames captured.")
        return

    bgr   = frames[-1]
    if flip:
        bgr = cv2.flip(bgr, 1)
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    results = {}
    with _make_landmarker(RunningMode.IMAGE, threshold=0.1) as lm:
        for label, arr in [("correct_RGB", rgb), ("wrong_BGR_as_RGB", bgr)]:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            r      = lm.detect(mp_img)
            results[label] = len(r.face_landmarks) if r.face_landmarks else 0

    print(f"  RGB input (correct):    {results['correct_RGB']} face(s) detected")
    print(f"  BGR input (wrong):      {results['wrong_BGR_as_RGB']} face(s) detected")

    # Save both versions for visual comparison.
    save_debug_frame(rgb[:, :, ::-1].copy(), DUMP_DIR, label="h6_rgb_saved_as_bgr")
    save_debug_frame(bgr, DUMP_DIR, label="h6_raw_bgr")
    print(f"  Frames saved to {DUMP_DIR}/ for comparison.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Face detection diagnostics")
    parser.add_argument("--device",    type=int,   default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-flip",   action="store_true")
    args = parser.parse_args()

    flip = not args.no_flip

    print(f"Face detection diagnostic  |  device={args.device}  flip={flip}")
    print(f"Model: {MODEL_PATH}")
    print("=" * 68)

    cap = _open_camera(args.device)

    try:
        h1_camera_warmup(cap)
        h2_frame_format(cap)
        h3_array_contiguity(cap, flip=flip)
        h4_image_vs_video_mode(cap, flip=flip)
        h5_confidence_thresholds(cap, flip=flip)
        h6_color_channels(cap, flip=flip)
    finally:
        cap.release()

    print("\n" + "=" * 68)
    print(f"Debug frames saved to: {DUMP_DIR}/")
    print("Open them in Preview/Finder to see what the camera is actually capturing.")


if __name__ == "__main__":
    main()
