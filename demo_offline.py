#!/usr/bin/env python3
"""Generate an explicitly synthetic session through the real scorer and reporter.

No camera, microphone, display, model weights, personal files, or network calls.
The input analyses are scripted fixtures, not detections or participant evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.signals.attention import AttentionAnalysis, AttentionZone
from src.signals.blink_detector import BlinkAnalysis, BlinkState
from src.signals.engagement import AttentionScorer
from src.signals.gaze import GazeAnalysis, GazeZone
from src.stimulus.analytics import SessionAnalytics
from src.stimulus.events import BehavioralSample, ResponseEvent, SessionLog, StimulusEvent
from src.stimulus.exporter import SessionExporter


SESSION_ID = "synthetic-orientation-v1"
FPS = 10
DURATION_S = 60
PHASES = [
    {"start_s": 0, "end_s": 10, "name": "centered inputs during confidence warmup"},
    {"start_s": 10, "end_s": 20, "name": "centered inputs"},
    {"start_s": 20, "end_s": 35, "name": "off-center gaze and head inputs"},
    {"start_s": 35, "end_s": 45, "name": "scripted prolonged-closure flag"},
    {"start_s": 45, "end_s": 50, "name": "missing face and all channels"},
    {"start_s": 50, "end_s": 60, "name": "return to centered inputs"},
]


def build_synthetic_session() -> SessionLog:
    """Deterministic input fixtures and relative timestamps; no real clock."""
    config = {
        "engagement": {
            "weights": {"gaze": 0.50, "head": 0.20, "eye": 0.30},
            "smoothing": {"alpha": 0.10, "state_min_frames": 5},
            "confidence": {"min_history_s": 10.0},
        },
        "provenance": {
            "synthetic": True,
            "generator": "demo_offline.build_synthetic_session",
            "scenario": SESSION_ID,
            "description": "Scripted signal analyses and responses; no participant data.",
            "timestamp_basis": "simulated relative seconds",
            "samples_per_simulated_second": FPS,
            "phases": PHASES,
        },
    }
    session = SessionLog(
        session_id=SESSION_ID,
        experiment_name="SYNTHETIC DEMO — no participant recording",
        start_ts=0.0,
        end_ts=float(DURATION_S),
        config=config,
    )
    for trial, (onset, latency) in enumerate(((20.0, 350.0), (40.0, 450.0))):
        stimulus_id = f"synthetic-prompt-{trial}"
        session.stimulus_events.append(StimulusEvent(
            stimulus_id, trial, "reaction_prompt", onset, onset + 3.0, 3000.0,
            {"synthetic": True, "text": "Scripted event; never displayed"},
        ))
        session.response_events.append(ResponseEvent(
            stimulus_id, trial, "key_press", onset + latency / 1000.0, latency, True,
            {"synthetic": True, "description": "Scripted latency; no key was pressed"},
        ))

    scorer = AttentionScorer(config)
    for index in range(DURATION_S * FPS):
        t = index / FPS
        away = 20 <= t < 35
        closed = 35 <= t < 45
        missing = 45 <= t < 50
        head = AttentionAnalysis(
            AttentionZone.LOOKING_AWAY if away else AttentionZone.FOCUSED,
            40.0 if away else 2.0, 1.0, 0.0, 1.0, 1.0,
            0.1 if away else 0.95, 2.0, t, index,
        )
        gaze = GazeAnalysis(
            GazeZone.RIGHT if away else GazeZone.CENTER,
            0.85 if away else 0.5, 0.5,
            not (away or closed), not closed, 0.01, 0.01,
            0.05 if away or closed else 0.95, t, index,
        )
        blink = BlinkAnalysis(
            BlinkState.CLOSED if closed else BlinkState.OPEN,
            0, 14.0, 0.10 if closed else 0.30, 0.05 if closed else 0.90,
            None, False, False, closed, False, t,
        )
        score = scorer.update(
            blink=None if missing else blink,
            head=None if missing else head,
            gaze=None if missing else gaze,
            face_detected=not missing,
            timestamp=t,
            frame_index=index,
        )
        active = next(
            (e.stimulus_id for e in session.stimulus_events if e.onset_ts <= t < e.offset_ts),
            None,
        )
        session.samples.append(BehavioralSample(
            frame_index=index,
            timestamp=t,
            active_stimulus_id=active,
            engagement_state=score.state.value,
            smoothed_score=score.smoothed_score,
            confidence=score.confidence,
            focused_fraction=score.focused_fraction,
            gaze_zone=GazeZone.UNRELIABLE.value if missing else gaze.zone.value,
            is_on_screen=False if missing else gaze.is_on_screen,
            gaze_h=0.0 if missing else gaze.mean_h,
            gaze_v=0.0 if missing else gaze.mean_v,
            head_zone="unreliable" if missing else head.zone.value,
            head_yaw=0.0 if missing else head.yaw,
            head_pitch=0.0 if missing else head.pitch,
            blink_state="unknown" if missing else blink.state.value,
            mean_ear=0.0 if missing else blink.mean_ear,
            blink_rate=0.0 if missing else blink.blink_rate_per_min,
            is_fatigued=score.is_fatigued,
        ))
    session.trial_summaries = SessionAnalytics().compute_trial_summaries(session)
    return session


def write_demo(output_dir: Path) -> Path:
    """Write a new demo directory, refusing to overwrite any existing output."""
    output_dir.mkdir(parents=True, exist_ok=False)
    session = build_synthetic_session()
    session_dir = output_dir / "session"
    SessionExporter.to_json(
        session, str(session_dir / "session_log.json"), summaries=session.trial_summaries,
    )
    SessionExporter.to_csv(
        session, str(session_dir / "samples.csv"), str(session_dir / "events.csv"),
    )

    # Exercise the same saved-file analytics path as a recorded session.
    from analyze_session import run
    run(argparse.Namespace(
        input=str(session_dir),
        output_dir=str(output_dir / "reports"),
        no_charts=False, no_html=False, open=False, profile=None,
    ))
    summary = {
        "synthetic": True,
        "session_id": SESSION_ID,
        "duration_s": DURATION_S,
        "samples": len(session.samples),
        "trials": len(session.stimulus_events),
        "scripted_key_latencies_ms": [e.latency_ms for e in session.response_events],
        "phases": PHASES,
        "scope": "Software demonstration only; no camera measurement or empirical accuracy claim.",
    }
    (output_dir / "demo_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return output_dir / "reports" / SESSION_ID / "report.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    try:
        report = write_demo(args.output_dir)
    except FileExistsError:
        parser.error(f"{args.output_dir} already exists; choose a new --output-dir.")
    print(f"\nSYNTHETIC demonstration complete. Open {report}")
    print("The session, signals, events, and key latencies were generated, not observed.")


if __name__ == "__main__":
    main()
