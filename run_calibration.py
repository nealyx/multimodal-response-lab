#!/usr/bin/env python3
"""Calibration session: establish a personal behavioral baseline profile.

Two modes
----------
webcam (default)
    Opens the webcam, runs a guided 30-second rest recording, and saves
    ``outputs/calibration/user_profile.json``.  The user simply sits
    naturally and looks at the screen — no task, no stimuli.

survey  (--survey <session_dir>)
    Skips the webcam entirely and runs a post-session questionnaire in the
    terminal.  Saves ``<session_dir>/labels.json``.

Calibration workflow
---------------------
  1. Warm-up phase (configurable, default 5 s)
       Alerts the system; data discarded.
  2. Rest phase (default 25 s)
       Collect CalibrationSamples at the live frame rate.
  3. Compute UserProfile via CalibrationProfiler.
  4. Save to ``outputs/calibration/user_profile.json``.

Usage
-----
    # Webcam calibration (saves user_profile.json)
    python run_calibration.py

    # Custom duration
    python run_calibration.py --duration 60

    # Post-session survey (saves labels.json next to session_log.json)
    python run_calibration.py --survey outputs/sessions/20241201_120000/

    # Specify custom output path for the profile
    python run_calibration.py --output outputs/calibration/my_profile.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

_DEFAULT_DURATION_S = 30
_DEFAULT_WARMUP_S   = 5
_DEFAULT_OUTPUT     = "outputs/calibration/user_profile.json"


# ── Survey mode ───────────────────────────────────────────────────────────────

def run_survey_mode(session_dir: str) -> None:
    """Run the post-session survey and save labels.json inside session_dir."""
    from src.calibration.labeler import SessionLabeler
    from src.calibration.survey import run_survey_cli

    session_path = Path(session_dir)
    if not session_path.exists():
        print(f"Session directory not found: {session_dir}")
        sys.exit(1)

    survey = run_survey_cli(session_dir=str(session_path))
    label  = SessionLabeler.from_survey(survey, session_dir=str(session_path))

    out_path = session_path / "labels.json"
    label.save(str(out_path))

    print(f"\nLabels saved to: {out_path}")
    print("  engagement : " + (label.label_for("engagement") or "ambiguous (excluded)"))
    print("  fatigue    : " + (label.label_for("fatigue")    or "ambiguous (excluded)"))
    print("  distraction: " + (label.label_for("distraction") or "ambiguous (excluded)"))

    survey_path = session_path / "survey.json"
    survey.session_dir = str(session_path)
    survey.save(str(survey_path))
    print(f"\nRaw survey saved to: {survey_path}")


# ── Webcam calibration mode ───────────────────────────────────────────────────

def run_webcam_calibration(
    duration_s:  float,
    warmup_s:    float,
    output_path: str,
    config_path: str,
) -> None:
    """Run the webcam calibration and save a UserProfile."""
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) is required for webcam calibration.")
        print("Install with: pip install opencv-python")
        sys.exit(1)

    import yaml
    from src.calibration.collector import CalibrationCollector, CalibrationSample
    from src.calibration.profiler import CalibrationProfiler
    from src.signals.attention import AttentionDetector
    from src.signals.blink_detector import BlinkDetector
    from src.signals.engagement import AttentionScorer
    from src.signals.eye_metrics import extract_eye_measurements
    from src.signals.gaze import GazeDetector, extract_gaze_measurements
    from src.signals.head_pose import build_camera_matrix, estimate_head_pose

    try:
        import mediapipe as mp
    except ImportError:
        print("MediaPipe is required. Install with: pip install mediapipe")
        sys.exit(1)

    # ── Load config ───────────────────────────────────────────────────────
    cfg_path = Path(config_path)
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        log.warning("Config not found at %s; using defaults", config_path)
        cfg = {}

    # ── Set up signal components ──────────────────────────────────────────
    signals_cfg      = cfg.get("signals", cfg)
    focal_scale      = float(cfg.get("head_pose", {}).get("focal_scale", 1.0))
    blink_detector   = BlinkDetector(signals_cfg)
    attention_det    = AttentionDetector(cfg)
    gaze_detector    = GazeDetector(cfg)
    attention_scorer = AttentionScorer(cfg)

    collector = CalibrationCollector()

    # ── Open camera ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        sys.exit(1)

    # ── FaceMesh setup ────────────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=       False,
        max_num_faces=           1,
        refine_landmarks=        True,
        min_detection_confidence=0.5,
        min_tracking_confidence= 0.5,
    )

    start_ts      = time.monotonic()
    phase         = "warmup"
    total_s       = warmup_s + duration_s
    cam_matrix    = None
    frame_idx     = 0

    print()
    print("=" * 60)
    print("  Calibration Session")
    print(f"  Warm-up : {warmup_s:.0f} s  |  Recording: {duration_s:.0f} s")
    print("  Sit naturally and look at the screen.")
    print("  Press Q to abort.")
    print("=" * 60)
    print()

    try:
        while True:
            elapsed = time.monotonic() - start_ts

            if elapsed >= total_s:
                break

            ret, frame = cap.read()
            if not ret:
                log.warning("Frame capture failed")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = face_mesh.process(frame_rgb)
            now       = time.monotonic()

            h, w = frame.shape[:2]
            face_detected = results.multi_face_landmarks is not None

            # Phase management
            if elapsed < warmup_s:
                if phase != "warmup":
                    phase = "warmup"
                remaining = warmup_s - elapsed
                phase_text = f"Warming up... {remaining:.0f}s"
                color = (0, 165, 255)
            else:
                if phase != "recording":
                    phase = "recording"
                    print("  Recording started. Stay still and relaxed.")
                remaining = total_s - elapsed
                phase_text = f"Recording... {remaining:.0f}s"
                color = (0, 200, 0)

            # ── Draw overlay ──────────────────────────────────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, phase_text, (12, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            n_rec = collector.n_samples
            if n_rec > 0:
                cv2.putText(frame, f"frames: {n_rec}", (w - 160, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # ── Extract signals ───────────────────────────────────────────
            ear        = 0.28
            blink_rate = 0.0
            head_yaw   = 0.0
            head_pitch = 0.0
            gaze_h     = 0.5
            gaze_v     = 0.5
            eng_score  = 0.0
            is_eyes_open = True

            if face_detected:
                lm = results.multi_face_landmarks[0]

                if cam_matrix is None:
                    cam_matrix = build_camera_matrix(w, h, focal_scale=focal_scale)

                eye_m = extract_eye_measurements(
                    lm, w, h, timestamp=now, frame_index=frame_idx, cfg=signals_cfg,
                )
                blink_a = (
                    blink_detector.update(eye_m) if eye_m is not None
                    else blink_detector.no_face_update()
                )
                if eye_m is not None:
                    ear = float(eye_m.mean_ear)
                    is_eyes_open = ear >= cfg.get("blink", {}).get("ear_close_threshold", 0.20)
                if blink_a is not None:
                    blink_rate = float(blink_a.blink_rate_per_min)

                pose = estimate_head_pose(
                    lm, w, h, cam_matrix, timestamp=now, frame_index=frame_idx,
                )
                att_a = (
                    attention_det.update(pose) if pose is not None
                    else attention_det.no_pose_update()
                )
                if pose is not None:
                    head_yaw   = float(pose.yaw)
                    head_pitch = float(pose.pitch)

                gaze_m = extract_gaze_measurements(
                    lm, w, h,
                    mean_ear=eye_m.mean_ear if eye_m is not None else 0.0,
                    timestamp=now, frame_index=frame_idx, cfg=cfg.get("gaze"),
                )
                if gaze_m is not None:
                    gaze_a = gaze_detector.update(
                        gaze_m,
                        head_zone=(att_a.zone.value if att_a is not None else None),
                        head_yaw=(pose.yaw if pose is not None else None),
                        head_pitch=(pose.pitch if pose is not None else None),
                    )
                    gaze_h = float(gaze_a.h_ratio)
                    gaze_v = float(gaze_a.v_ratio)
                else:
                    gaze_a = gaze_detector.no_gaze_update()

                eng = attention_scorer.update(
                    blink=blink_a,
                    head=att_a,
                    gaze=gaze_a,
                    face_detected=True,
                    timestamp=now,
                    frame_index=frame_idx,
                )
                eng_score = float(eng.smoothed_score)

            frame_idx += 1

            # ── Collect sample (only during recording phase) ──────────────
            if phase == "recording" and face_detected:
                sample = CalibrationSample(
                    timestamp=       now,
                    ear=             ear,
                    blink_rate=      blink_rate,
                    head_yaw=        head_yaw,
                    head_pitch=      head_pitch,
                    gaze_h=          gaze_h,
                    gaze_v=          gaze_v,
                    engagement_score=eng_score,
                    is_face_detected=True,
                    is_eyes_open=    is_eyes_open,
                )
                collector.add(sample)
            elif phase == "recording":
                # No face: record a placeholder so duration tracking works
                sample = CalibrationSample(
                    timestamp=       now,
                    ear=             0.0,
                    blink_rate=      0.0,
                    head_yaw=        0.0,
                    head_pitch=      0.0,
                    gaze_h=          0.5,
                    gaze_v=          0.5,
                    engagement_score=0.0,
                    is_face_detected=False,
                    is_eyes_open=    False,
                )
                collector.add(sample)

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nCalibration aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

    # ── Compute and save profile ──────────────────────────────────────────
    print()
    n_valid = collector.n_face_detected
    det_rate = collector.face_detection_rate * 100.0
    print(f"  Collected {collector.n_samples} frames  "
          f"({n_valid} with face detected, {det_rate:.0f}%)")

    if n_valid < 10:
        print("  WARNING: Very few valid frames. "
              "Ensure your face was visible and well lit.")

    profile = CalibrationProfiler.compute(collector)
    profile.save(output_path)

    print()
    print(f"  Profile saved to: {output_path}")
    print(f"  Profile ID       : {profile.profile_id}")
    print(f"  Duration         : {profile.duration_s:.1f} s")
    print(f"  Blink rate (mean): {profile.blink.rate_per_min:.1f} /min")
    print(f"  Open EAR (mean)  : {profile.blink.open_ear:.3f}")
    print(f"  Head yaw center  : {profile.head.yaw_center:.1f}°")
    print(f"  Engagement mean  : {profile.engagement.score_mean:.3f}")
    print()
    print("  Pass --profile outputs/calibration/user_profile.json")
    print("  to analyze_session.py to enable baseline comparison.")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration: establish personal baseline or run post-session survey",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--survey", metavar="SESSION_DIR",
        help="Run post-session survey for SESSION_DIR (saves labels.json). "
             "Skips webcam entirely.",
    )
    parser.add_argument(
        "--duration", type=float, default=_DEFAULT_DURATION_S,
        help=f"Recording duration in seconds (default: {_DEFAULT_DURATION_S})",
    )
    parser.add_argument(
        "--warmup", type=float, default=_DEFAULT_WARMUP_S,
        help=f"Warm-up phase duration in seconds (default: {_DEFAULT_WARMUP_S})",
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT,
        help=f"Path to save user_profile.json (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Signal config file (default: config/default.yaml)",
    )
    args = parser.parse_args()

    if args.survey:
        run_survey_mode(args.survey)
    else:
        run_webcam_calibration(
            duration_s=  args.duration,
            warmup_s=    args.warmup,
            output_path= args.output,
            config_path= args.config,
        )


if __name__ == "__main__":
    main()
