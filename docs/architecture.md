# Architecture

This document describes the full design of multimodal-response-lab: data flow, module responsibilities, temporal synchronization model, and the key engineering decisions made at each layer.

---

## System overview

The system is a **layered signal pipeline**. Each layer takes the raw output of the layer below, applies one behavioral inference step, and emits a typed data object that the next layer can consume. No layer reaches up or sideways — each module depends only on its own configuration and the typed input it receives.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Input: Webcam @ 30 FPS                           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ raw BGR frames
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 0 — FacePipeline  (src/pipelines/face_pipeline.py)               │
│                                                                         │
│  MediaPipe FaceLandmarker 0.10+ (Task API, float16, ~3 MB)              │
│  Output: 478 normalized (x, y, z) landmarks per face                   │
│  Threading: capture thread queues frames; caller calls process_latest() │
└──────────┬──────────────────────┬──────────────────────────────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌────────────────┐  ┌─────────────────────────┐  ┌────────────────────────┐
│ Layer 1a       │  │ Layer 1b                 │  │ Layer 1c               │
│ Eye metrics    │  │ Head pose                │  │ Iris position          │
│ (eye_metrics)  │  │ (head_pose)              │  │ (gaze, extract_*)      │
│                │  │                          │  │                        │
│ EAR (left,     │  │ 6-point solvePnP model   │  │ H/V ratios per eye     │
│ right, mean)   │  │ → rvec, tvec             │  │ from indices 468/473   │
│ openness [0,1] │  │ → RQDecomp3x3 Euler      │  │ is_reliable (EAR gate) │
│                │  │ → PoseMeasurement        │  │ → GazeMeasurement      │
└───────┬────────┘  └───────────┬─────────────┘  └──────────┬─────────────┘
        │                       │                            │
        ▼                       ▼                            ▼
┌────────────────┐  ┌─────────────────────────┐  ┌────────────────────────┐
│ Layer 2a       │  │ Layer 2b                 │  │ Layer 2c               │
│ Blink Detector │  │ Attention Detector       │  │ Gaze Detector          │
│ (blink_detector│  │ (attention)              │  │ (gaze)                 │
│                │  │                          │  │                        │
│ 4-state FSM    │  │ Zone classification      │  │ 9-zone classification  │
│ rolling rate   │  │ rolling yaw/pitch std    │  │ on-screen heuristic    │
│ fatigue flags  │  │ attention_fraction       │  │ rolling H/V stability  │
│ → BlinkAnalysis│  │ → AttentionAnalysis      │  │ → GazeAnalysis         │
└───────┬────────┘  └───────────┬─────────────┘  └──────────┬─────────────┘
        │                       │                            │
        └───────────────────────┼────────────────────────────┘
                                │ 3 orthogonal behavioral channels
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3 — AttentionScorer  (src/signals/engagement.py)                  │
│                                                                         │
│  Weighted sum:  score = 0.50·gaze + 0.20·head + 0.30·eye               │
│  EMA:           smoothed = α·raw + (1−α)·prev  (α=0.10, ~0.4s τ)       │
│  Debounce:      new state confirmed after 5 consecutive matching frames  │
│  Confidence:    penalised for missing iris, high reproj error, short Δt │
│                                                                         │
│  Output: EngagementScore (state, smoothed_score, confidence,            │
│          focused_fraction, component scores, flags)                     │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────────────┐
          ▼              ▼                       ▼
┌──────────────┐  ┌─────────────────┐  ┌────────────────────────────────┐
│ Live overlay │  │ Stimulus layer  │  │ Reporting layer                │
│ (overlay.py) │  │ (src/stimulus/) │  │ (src/reporting/)               │
│              │  │                 │  │                                │
│ HUD panels   │  │ StimulusPresent-│  │ SessionReporter:               │
│ drawn on     │  │ er + EventReco- │  │ SessionLog → SessionMetrics    │
│ each frame   │  │ rder + exporter │  │                                │
│              │  │                 │  │ charts.py → PNG figures         │
│              │  │ JSON/CSV output │  │ html_report.py → HTML          │
└──────────────┘  └─────────────────┘  └────────────────────────────────┘
```

---

## Temporal synchronization model

**All timestamps in the system are `time.monotonic()`.**

Monotonic time is immune to NTP corrections and daylight-saving adjustments. This matters because reaction latency is computed as `response_ts − onset_ts` — if either used wall-clock time and a time correction happened mid-session, latencies would be corrupted.

### The timing chain

```
time.monotonic()
    │
    ├── frame.timestamp_process  ← set by FacePipeline when frame is dequeued
    │                              (not when the sensor captured it — ~30–100 ms earlier)
    │
    ├── StimulusEvent.onset_ts  ← set when StimulusPresenter first renders the stimulus
    │                              (before cv2.imshow — ~8–32 ms before display)
    │
    ├── ResponseEvent.response_ts  ← set at key press detection in main loop
    │
    └── BehavioralSample.timestamp  ← same as frame.timestamp_process
```

**Reaction latency** = `response_ts − onset_ts`  
This is stimulus-to-key, not stimulus-on-screen-to-key. Display latency (~8–32 ms typical LCD) is a known systematic offset, documented but not corrected.

**Why not use wall-clock time?**  
`time.time()` can jump backward or skip during NTP corrections. A 20 ms NTP adjustment in a 300 ms reaction time window produces a −6.7% to +6.7% latency error. Monotonic clocks never jump.

---

## Module responsibilities

### `src/pipelines/face_pipeline.py` — FacePipeline

- Manages the capture thread and the MediaPipe inference thread independently
- Exposes a single `process_latest(timeout)` call to the main loop
- Handles camera warm-up (blank-frame rejection), backend selection, and device index
- Detects Continuity Camera (iPhone webcam via macOS) using `system_profiler` and warns

### `src/signals/eye_metrics.py` — EAR extraction

- Computes Eye Aspect Ratio (EAR) from 6 landmark indices per eye
- Normalises to `mean_openness ∈ [0, 1]` using configurable `ear_closed_ref` and `ear_open_ref`
- Returns `Optional[EyeMeasurement]` — None when fewer than 468 landmarks are present

### `src/signals/blink_detector.py` — BlinkDetector (4-state FSM)

States: `OPEN → CLOSING → CLOSED → OPENING → OPEN`

- **Hysteresis:** separate close threshold (0.20) and open threshold (0.25) prevent boundary bouncing
- **Debounce:** 2+ consecutive frames required to confirm state transitions
- **Duration gate:** only counts as a blink if the closed interval is 60–500 ms
- **Rolling rate:** blinks/minute over a 60 s window with a 15 s warm-up guard
- **Fatigue flags:** `is_prolonged_closure` (>500 ms), `is_low_blink_rate` (<8/min), `is_drowsy` (compound)

Why an FSM over a simple threshold: a threshold can't distinguish a 50 ms landmark jitter from a 200 ms real blink. The FSM adds duration, debounce, and hysteresis at minimal computational cost.

### `src/signals/head_pose.py` — solvePnP pose estimation

6-point 3D canonical face model in **Y-down convention** (critical: OpenCV camera coordinates use Y-down):

```
Point     3D (mm)               Landmark index
─────────────────────────────────────────────
Nose tip  [ 0.0,   0.0,   0.0]  1
Chin      [ 0.0,  63.6, -12.5]  152
L eye     [-43.3,-32.7, -26.0]  33
R eye     [ 43.3,-32.7, -26.0]  263
L mouth   [-28.9,  28.9,-24.1]  61
R mouth   [ 28.9,  28.9,-24.1]  291
```

`RQDecomp3x3` extracts Euler angles (pitch, yaw, roll) in degrees from the rotation matrix.
Reprojection error is logged; >10 px indicates poor landmark quality and penalises the confidence score.

Why not use the landmark-provided face geometry directly: solvePnP gives an explicit 6-DOF pose in a metric space (millimetres), which makes the angle thresholds in `attention.py` geometrically meaningful. Direct 2D landmark positions are projective and frame-size-dependent.

### `src/signals/gaze.py` — GazeDetector

Iris indices: `473` (subject's left iris centre), `468` (subject's right iris centre).
With `flip_horizontal=True` (default), both irises move in the same image-pixel direction.

H-ratio: `(iris_x − inner_corner_x) / (outer_corner_x − inner_corner_x)` ∈ [0, 1]  
→ 0.5 = centered, <0.5 = looking toward image-left, >0.5 = toward image-right

V-ratio: `(iris_y − lid_top_y) / (lid_bottom_y − lid_top_y)` ∈ [0, 1]  
→ 0.25–0.70 = normal range; <0.25 = looking up; >0.70 = looking down

**on-screen heuristic:** iris in CENTER zone AND `|yaw| ≤ 25°` AND `|pitch| ≤ 20°`  
Head pose gates the iris reading because large head rotations distort the apparent iris position in the image plane even when the eyes are looking forward.

### `src/signals/engagement.py` — AttentionScorer

Three component scores → weighted sum → EMA → debounced state:

```
gaze_score  = gaze.on_screen_fraction   (if reliable)
              OR head.attention_fraction × 0.65  (head-only fallback)

head_score  = head.attention_fraction × (1 − stability_penalty)
              stability_penalty = min(0.5, (yaw_std + pitch_std) / 30°)

eye_score   = blink.mean_openness
              × 0.15 if prolonged_closure
              × 0.35 if drowsy
              × 0.80 if low_blink_rate
              × 0.85 if high_blink_rate

raw = 0.50 × gaze_score + 0.20 × head_score + 0.30 × eye_score
smoothed = 0.10 × raw + 0.90 × prev_smoothed

candidate_state = classify(smoothed, is_fatigued, confidence)
if candidate == prev_candidate: candidate_frames += 1
if candidate_frames >= 5: current_state = candidate  ← debounce
```

**Why three channels with these weights:**  
Gaze is the primary signal because it directly measures where attention is directed. Eye alertness (EAR/blink) captures fatigue independently of orientation. Head pose is a lower-weight supplement — it becomes important when iris data is unreliable (glasses, poor lighting), but adds little when gaze is reliable because the gaze component already incorporates a head-pose gate.

---

## Stimulus layer (Day 7)

### Scheduling model

The `build_schedule(cfg, t0)` function converts the YAML stimulus list into a flat chronological list of `ScheduledTrial` objects with absolute monotonic onset times:

```
t_trial_k = t0 + baseline_s + Σ_{i=0}^{k-1} (duration_i/1000 + max(0.5, isi + jitter_i))
```

`StimulusPresenter.update(frame, ts)` checks whether `ts ≥ trial.start_ts` on every frame. This means timing is **pull-based** — no background threads, no timers. A dropped frame simply delays the next check by one frame period (~33 ms), which is within the frame-rate granularity of the system anyway.

### Response detection

`EventRecorder.record_sample(sample, active_event)` detects two implicit response types:
- `gaze_shift`: `is_on_screen` transitions from `True → False` while a stimulus is active
- `attention_change`: `engagement_state` value changes while a stimulus is active

Key-press responses (`key_press`) are detected explicitly in the main loop via `presenter.notify_key(key, ts)`.

All response events carry `latency_ms = (response_ts − onset_ts) × 1000`.

---

## Reporting layer (Day 8)

The reporting layer has **zero runtime dependency on OpenCV or MediaPipe**. It reads session files (JSON + CSV) and produces derived artifacts.

### SessionMetrics schema (abbreviated)

```python
SessionMetrics
├── session_id, experiment_name, generated_at
├── duration_s, n_frames, n_trials, n_responses
│
├── EngagementMetrics
│   ├── score_summary: SignalSummary  (mean, std, min, max, p25, p75, n)
│   └── *_fraction: float  (focused, drifting, distracted, fatigued, unreliable)
│
├── GazeMetrics
│   ├── on_screen_fraction
│   └── gaze_h_summary, gaze_v_summary: SignalSummary
│
├── HeadMetrics
│   ├── head_focused_fraction
│   └── yaw_summary, pitch_summary: SignalSummary
│
├── BlinkMetrics
│   ├── blink_rate_summary: SignalSummary
│   ├── fatigue_fraction
│   └── ear_summary: SignalSummary
│
└── stimulus_types: Dict[str, StimulusTypeMetrics]
    └── StimulusTypeMetrics
        ├── n_trials, hit_rate
        ├── mean/median/std latency_ms
        ├── mean_baseline_score, mean_during_score, score_delta
        └── mean_recovery_s
```

### Per-trial analysis

For each stimulus trial, `SessionAnalytics` computes:
- `baseline_score`: mean engagement in the 5 s window before onset
- `during_score`: mean engagement during the stimulus
- `score_delta`: `during − baseline` (negative = stimulus was disruptive)
- `recovery_s`: seconds after stimulus offset for score to return to ≥90% of baseline

This structure parallels event-related analysis in EEG/fMRI research — measuring signals relative to event epochs rather than as raw time series.

---

## Key design decisions

### 1. All timestamps are monotonic

`time.monotonic()` everywhere. No `datetime.now()`, no `time.time()`. Reaction latency is simply `response_ts − onset_ts`.

### 2. Raw measurements are strictly separated from inferred states

`GazeMeasurement` (iris ratios, pixel coordinates) is separate from `GazeAnalysis` (zone classification, on-screen heuristic). `EyeMeasurement` (EAR values) is separate from `BlinkAnalysis` (FSM state, fatigue flags). This separation makes each layer independently testable.

### 3. Every module accepts `None` inputs gracefully

If iris landmarks are absent, `GazeDetector.no_gaze_update()` returns the last valid analysis. If head pose fails, `AttentionDetector.no_pose_update()` returns the last valid result. `AttentionScorer.update()` can be called with all signals as `None` and will return a valid `EngagementScore` (with `confidence=0.0`). This prevents a single bad frame from crashing the pipeline.

### 4. Configuration is always a dict, never a file path

Modules receive `cfg: dict` at construction and never import YAML or JSON files directly. This makes unit testing trivial — pass `{"engagement": {"smoothing": {"alpha": 1.0}}}` and the module immediately reflects it, no file I/O.

### 5. Tests inject typed mock objects, not raw landmarks

Tests build `BlinkAnalysis`, `GazeAnalysis`, and `AttentionAnalysis` objects directly from Python constructors. No webcam, no MediaPipe model, no OpenCV display. The full suite runs in ~1 second.

### 6. The reporting layer is dependency-isolated

`src/reporting/` imports only `src/stimulus/events.py` (pure dataclasses) and `src/stimulus/analytics.py` (pure computation). It never imports OpenCV, MediaPipe, or any live-pipeline module. This means reports can be generated on any machine with `numpy` and `matplotlib` — no webcam stack required.

---

## Signal interaction and failure modes

```
Signal interaction matrix:

              Gaze score  Head score  Eye score
              ──────────  ──────────  ─────────
Iris absent      ↓ (head    —           —
                 fallback)
Head absent      ↓ (no      ↓ (neutral  —
                 gate)      0.5)
Blink occludes   ↓ (iris    —           ↓↓↓
                 unreliable)             (×0.15)
High reproj err  ↓ (gaze    ↓ (penalty  —
(confidence)     penalty)   in conf.)
No face          0.0        0.5         0.5
                 (gaze=0    (neutral)   (neutral)
                 fallback)

Note: head=0.5 and eye=0.5 when absent because "no evidence of
distraction" ≠ "evidence of distraction". Only gaze defaults to
0.0 when absent because a missing face is more likely off-screen
than unknown.
```

---

## File naming conventions

| Pattern | Meaning |
|---------|---------|
| `run_*.py` | Live script — requires webcam, enters main loop, calls cv2.imshow |
| `src/signals/*.py` | Signal processing module — pure Python, no I/O, testable offline |
| `src/stimulus/*.py` | Experiment infrastructure — timestamps, events, schedule |
| `src/reporting/*.py` | Offline analytics — no OpenCV/MediaPipe dependency |
| `tests/test_*.py` | Unit tests — no webcam, no display, no file I/O (except loader tests) |
| `configs/*.yaml` | Configuration — all runtime parameters |
| `outputs/sessions/` | Session data written by live experiment scripts |
| `outputs/reports/` | Analytics output written by analyze_session.py |
