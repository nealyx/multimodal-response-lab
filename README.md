# multimodal-response-lab

> Real-time behavioral engagement analysis from a single consumer webcam — no special hardware, no EEG, no wearables.

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-285%20passing-27ae60)](tests/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)](https://github.com/nealyx/multimodal-response-lab)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-FF6F00)](https://mediapipe.dev)

The system fuses eye aspect ratio (blink/fatigue), head-pose estimation (solvePnP), and iris-ratio gaze direction into a composite engagement score that runs at **25–30 FPS** on a MacBook CPU. It supports configurable visual stimulus experiments, synchronized event logging with monotonic timestamps, and offline session analytics with exported charts and HTML reports.

**This is approximate behavioral orientation inference, not cognitive measurement.**
It tells you where the eyes and head are pointing, not what the person is thinking.
All processing is local — no data leaves the machine.

---

## What the system produces

Every frame, the pipeline emits:

| Output | Type | Description |
|--------|------|-------------|
| `EngagementState` | enum | `FOCUSED` / `DRIFTING` / `DISTRACTED` / `FATIGUED` / `UNRELIABLE` |
| `smoothed_score` | float [0, 1] | EMA-smoothed composite signal — suitable for time-series logging |
| `confidence` | float [0, 1] | Data-quality gate; low during warm-up or when signals are absent |
| `focused_fraction` | float [0, 1] | Rolling 60-second FOCUSED proportion |
| `gaze_zone` | enum | `CENTER` / `LEFT` / `RIGHT` / `UP` / `DOWN` / `UNRELIABLE` + diagonals |
| `head_zone` | enum | `FOCUSED` / `GLANCE` / `LOOKING_AWAY` |
| `blink_state` | enum | `OPEN` / `CLOSING` / `CLOSED` / `OPENING` |
| `fatigue_label` | str | `LOW` / `MODERATE` / `HIGH` |
| `reaction_latency_ms` | float | Time from stimulus onset to behavioral response (experiment mode) |

---

## Quick demo

```bash
# 1. Set up (one-time, ~3 minutes)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .cache/models && curl -L -o .cache/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# 2. Run the full engagement dashboard (press Q to quit)
python run_attention_test.py

# 3. Run a stimulus experiment and generate a report
python run_stimulus_experiment.py
python analyze_session.py outputs/sessions/<session_id>/ --open
```

---

## What has been built

| Day | Component | Entry point | Key output type |
|-----|-----------|-------------|-----------------|
| 2 | Face landmark detection | `run_face_test.py` | 478-point face mesh @ 25–30 FPS |
| 3 | EAR / blink FSM | `run_blink_test.py` | `BlinkAnalysis` (rate, drowsiness, fatigue) |
| 4 | Head pose + attention zone | `run_head_pose_test.py` | `PoseMeasurement` + `AttentionAnalysis` |
| 5 | Iris gaze estimation | `run_gaze_test.py` | `GazeAnalysis` (zone, on-screen fraction, stability) |
| 6 | Composite engagement scoring | `run_attention_test.py` | `EngagementScore` (state, score, confidence) |
| 7 | Stimulus-response experiments | `run_stimulus_experiment.py` | `SessionLog` (events, latencies, samples) |
| 8 | Offline analytics + reporting | `analyze_session.py` | `SessionMetrics` + charts + HTML report |

**285 unit tests, zero webcam dependencies** — all signal modules are tested with injected mock objects.

---

## Architecture

```
Webcam @ 30 FPS
    │
    ▼
┌─────────────────────────────────────┐
│  FacePipeline                        │  MediaPipe FaceLandmarker 0.10+
│  478 normalized landmarks per frame  │  float16 model, CPU-only, ~3 MB
└────────┬──────────────┬─────────────┘
         │              │              │
         ▼              ▼              ▼
   ┌──────────┐  ┌───────────┐  ┌───────────────┐
   │ EAR layer│  │ Head pose │  │ Iris position │
   │          │  │ (solvePnP)│  │ (H/V ratios)  │
   └────┬─────┘  └─────┬─────┘  └───────┬───────┘
        │              │                 │
        ▼              ▼                 ▼
   ┌──────────┐  ┌───────────┐  ┌───────────────┐
   │  Blink   │  │ Attention │  │ Gaze          │
   │ Detector │  │ Detector  │  │ Detector      │
   │  (FSM)   │  │           │  │               │
   └────┬─────┘  └─────┬─────┘  └───────┬───────┘
        │              │                 │
        └──────────────┼─────────────────┘
                       │  3 orthogonal channels
                       ▼
              ┌────────────────┐
              │ AttentionScorer│  EMA α=0.10, 5-frame debounce
              │                │  weighted sum: gaze 50% + eye 30% + head 20%
              │ EngagementScore│
              └────────┬───────┘
                       │
           ┌───────────┼────────────────┐
           ▼           ▼                ▼
    ┌──────────┐ ┌──────────────┐ ┌──────────────────┐
    │ Live HUD │ │ Stimulus     │ │ Session Reporter │
    │ overlay  │ │ Experiment   │ │ (offline)        │
    └──────────┘ └──────┬───────┘ └────────┬─────────┘
                        │                   │
                        ▼                   ▼
                 ┌────────────┐     ┌──────────────────┐
                 │ SessionLog │     │ SessionMetrics   │
                 │ JSON + CSV │     │ charts + HTML    │
                 └────────────┘     └──────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full design walkthrough: data flow, temporal synchronization model, key design decisions, and signal interaction.

---

## Setup

**Requirements:** Python 3.9+, macOS (Apple Silicon or Intel) or Linux. A webcam is required only for live scripts; all 285 tests run offline.

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
```

**Apple Silicon (M1/M2/M3):** Install the MPS-enabled PyTorch build:
```bash
pip install torch torchvision torchaudio
```

**Intel Mac / Linux:** Same command — PyPI selects the correct CPU wheel automatically.

### 2. MediaPipe face model (one-time, ~3 MB)

```bash
mkdir -p .cache/models
curl -L -o .cache/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

### 3. Verify the setup

```bash
python -m pytest tests/ -q        # should print "285 passed"
```

### macOS camera permissions

On first run, macOS will prompt for camera access. If the prompt does not appear or the camera returns a black frame, go to **System Settings → Privacy & Security → Camera** and enable access for Terminal (or your IDE).

---

## Running the live pipeline

All live scripts share the same CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--device N` | `0` | Camera device index |
| `--backend NAME` | `avfoundation` | OpenCV backend (`avfoundation` / `any` / `v4l2`) |
| `--debug` | off | Print per-frame values to terminal |
| `--config PATH` | `configs/default.yaml` | Alternate config file |

Press **Q** or **Esc** to quit. A session summary prints on exit.

### Layer by layer

```bash
# Day 2 — face landmark mesh (validates MediaPipe is working)
python run_face_test.py

# Day 3 — blink detector + EAR metrics + fatigue label
python run_blink_test.py

# Day 4 — head pose (solvePnP) + attention zone + 3D axes overlay
python run_head_pose_test.py

# Day 5 — iris gaze estimation + on-screen heuristic
python run_gaze_test.py

# Day 6 — full composite engagement dashboard  ← start here
python run_attention_test.py
```

### What the engagement dashboard shows

```
Engagement: FOCUSED
Score  0.87  [████████████████░░░░]
Conf   0.94  [██████████████████░░]
-- components --
Gaze  0.90   [██████████████████░░]
Head  0.82   [████████████████░░░░]
Eye   0.86   [█████████████████░░░]
Focused 82%  Engaged 95%
Blink 13.4/min  [LOW]
```

The right side retains the head-pose, gaze, and blink HUD panels from earlier days so all signal layers are visible at once.

---

## Stimulus-response experiments (Day 7)

```bash
# Run an experiment with the default configuration (18 trials)
python run_stimulus_experiment.py

# Custom experiment config
python run_stimulus_experiment.py --experiment configs/experiment_default.yaml

# With camera override
python run_stimulus_experiment.py --device 1 --debug
```

**During the experiment:**
- The engagement dashboard remains visible on the left
- Stimuli appear on the webcam feed: color flashes, shapes, moving targets, reaction prompts
- Press **SPACE** when you see a `PRESS SPACE` prompt (reaction-time trials)
- Press **Q** or **Esc** to quit; the session is exported automatically

**Output files** written to `outputs/sessions/<session_id>/`:
```
session_log.json    — stimulus events, responses, trial summaries, config snapshot
samples.csv         — per-frame behavioral time series (one row per frame)
events.csv          — merged stimulus + response event table
```

### Experiment configuration

Edit [`configs/experiment_default.yaml`](configs/experiment_default.yaml) to change:

```yaml
experiment:
  baseline_duration_s: 12.0        # warm-up before first stimulus
  inter_stimulus_interval_s: 5.0   # mean gap between trials
  isi_jitter_s: 1.5                 # ±jitter to prevent anticipation
  randomise_order: false

  stimuli:
    - type: reaction_prompt
      text: "PRESS SPACE"
      response_key: space
      duration_ms: 3000
      n_trials: 5
```

Supported stimulus types: `color_flash`, `shape`, `moving_target`, `reaction_prompt`.

---

## Session analytics and reporting (Day 8)

```bash
# Analyze a session directory (JSON + CSV)
python analyze_session.py outputs/sessions/<session_id>/

# Open the HTML report in the browser immediately
python analyze_session.py outputs/sessions/<session_id>/ --open

# Specify output directory
python analyze_session.py outputs/sessions/<session_id>/ --output-dir outputs/reports/

# Skip charts (fast, JSON/CSV only)
python analyze_session.py outputs/sessions/<session_id>/ --no-charts
```

**Report output** written to `outputs/reports/<session_id>/`:
```
session_metrics.json    — SessionMetrics dataclass serialized (20+ behavioral metrics)
timeline.csv            — sample-by-sample time series with relative timestamps
charts/
  engagement_timeline.png    — smoothed score + state shading + stimulus markers
  state_distribution.png     — FOCUSED/DRIFTING/DISTRACTED/FATIGUED/UNRELIABLE fractions
  signal_channels.png        — gaze, head yaw, blink rate over time
  stimulus_comparison.png    — per-type baseline vs. during score (if stimuli present)
  reaction_latency.png       — mean RT per stimulus type (if key-press responses)
report.html             — self-contained HTML with all charts embedded as base64
```

The reporting layer has **no dependency on OpenCV or MediaPipe** — it reads exported files and runs on any machine with NumPy and matplotlib.

---

## Tests

```bash
python -m pytest tests/ -q          # 285 tests, ~1 second
python -m pytest tests/ -v          # verbose: see every test name
python -m pytest tests/test_engagement.py -v   # single module
```

| Test file | What it covers | Tests |
|-----------|---------------|-------|
| `test_eye_metrics.py` | EAR normalisation, openness computation | 18 |
| `test_blink_detector.py` | Blink FSM, rate thresholds, fatigue labels | 32 |
| `test_head_pose.py` | solvePnP pose, Euler angles, attention zones | 27 |
| `test_gaze.py` | Iris H/V ratios, zone classification, on-screen heuristic | 44 |
| `test_engagement.py` | EMA smoothing, state debounce, confidence, fractions | 45 |
| `test_stimulus_events.py` | Event schemas, stimulus rendering, `build_schedule` | 52 |
| `test_analytics.py` | Session analytics, event recorder, exporter | 28 |
| `test_reporting.py` | Metrics computation, loader, HTML report, charts | 43 |

**All tests are offline.** They inject typed mock objects directly into signal modules — no webcam, no MediaPipe model, no OpenCV display required. This makes the suite fast (~1 s), deterministic, and runnable in CI.

---

## Configuration reference

All parameters live in [`configs/default.yaml`](configs/default.yaml). Modules receive a config dict at construction and never import the file directly, making any value trivially overridable in tests.

```yaml
capture:
  device_id: 0             # camera index (0 = first AVFoundation device)
  backend: avfoundation    # macOS native; use "any" on Linux
  fps: 30
  flip_horizontal: true    # mirror mode for selfie-camera view

signals:
  ear_closed_ref: 0.15     # EAR reference for fully closed eye
  ear_open_ref:   0.35     # EAR reference for fully open eye
  blink:
    ear_close_threshold:      0.20   # below → CLOSING state
    ear_open_threshold:       0.25   # above → OPENING state (hysteresis gap)
    drowsy_closure_ms:        500.0  # single closure ≥ this → drowsy flag
    low_blink_rate_threshold:  8.0   # blinks/min below → staring/early fatigue
    high_blink_rate_threshold: 25.0  # blinks/min above → eye strain

head_pose:
  focal_scale: 1.0          # focal_length = focal_scale × frame_width
  attention:
    yaw_focus_deg:    20.0  # |yaw| ≤ this → FOCUSED zone
    pitch_focus_deg:  15.0
    yaw_glance_deg:   35.0  # |yaw| ≤ this → GLANCE zone
    pitch_glance_deg: 25.0

gaze:
  center_h_lo: 0.35         # iris H-ratio band for CENTER zone
  center_h_hi: 0.65
  center_v_lo: 0.25
  center_v_hi: 0.70
  ear_reliable_min: 0.15    # below → iris measurement unreliable
  head_yaw_max_on_screen: 25.0   # head-pose gate for is_on_screen

engagement:
  weights: {gaze: 0.50, head: 0.20, eye: 0.30}
  thresholds:
    focused:         0.72
    drifting:        0.45
    min_confidence:  0.20
  smoothing:
    alpha:            0.10   # EMA decay — ~0.4 s half-life at 25 FPS
    state_min_frames: 5      # debounce before state badge changes
  confidence:
    min_history_s: 10.0      # warm-up seconds before confidence reaches 1.0
  rolling_window_s: 60.0
```

---

## Project structure

```
multimodal-response-lab/
├── configs/
│   ├── default.yaml              # runtime parameters for live pipeline
│   └── experiment_default.yaml   # stimulus experiment configuration
│
├── src/
│   ├── pipelines/
│   │   └── face_pipeline.py      # MediaPipe FaceLandmarker wrapper (threaded capture)
│   ├── signals/
│   │   ├── eye_metrics.py        # EAR computation + normalisation
│   │   ├── blink_detector.py     # blink FSM + rolling rate + fatigue labels
│   │   ├── head_pose.py          # solvePnP 6-DOF pose estimation
│   │   ├── attention.py          # attention zone classifier + stability metrics
│   │   ├── gaze.py               # iris H/V ratios, 9-zone classification, on-screen heuristic
│   │   └── engagement.py         # composite engagement scorer (AttentionScorer)
│   ├── stimulus/
│   │   ├── events.py             # StimulusEvent, ResponseEvent, BehavioralSample schemas
│   │   ├── stimuli.py            # ColorFlash, ShapeStimulus, MovingTarget, ReactionPrompt
│   │   ├── presenter.py          # StimulusPresenter + build_schedule
│   │   ├── recorder.py           # EventRecorder (onset/offset/response detection)
│   │   ├── analytics.py          # per-trial latency, recovery, aggregates
│   │   └── exporter.py           # SessionExporter: JSON + CSV
│   ├── reporting/
│   │   ├── metrics.py            # SessionMetrics, SignalSummary, channel metric dataclasses
│   │   ├── reporter.py           # SessionReporter.compute(log) → SessionMetrics
│   │   ├── loader.py             # load_session_dir / load_session_json from disk
│   │   ├── charts.py             # matplotlib chart generators (5 chart types)
│   │   └── html_report.py        # self-contained HTML report with base64 charts
│   └── utils/
│       ├── camera.py             # camera enumeration + Continuity Camera detection
│       ├── overlay.py            # HUD rendering: mesh, axes, metrics, dashboard
│       └── landmarks.py          # MediaPipe landmark index constants
│
├── tests/                        # 285 offline unit tests (no webcam required)
├── docs/
│   └── architecture.md           # full design walkthrough + architecture diagram
│
├── run_face_test.py              # Day 2 — face mesh
├── run_blink_test.py             # Day 3 — blink + EAR metrics
├── run_head_pose_test.py         # Day 4 — head pose + attention zone
├── run_gaze_test.py              # Day 5 — iris gaze estimation
├── run_attention_test.py         # Day 6 — composite engagement dashboard
├── run_stimulus_experiment.py    # Day 7 — stimulus-response experiment
└── analyze_session.py            # Day 8 — offline analytics + HTML report
```

---

## Limitations

**This is a behavioral engineering instrument, not a neuroscience tool.**

| Limitation | Detail |
|------------|--------|
| **Webcam gaze is approximate** | Expected accuracy ±4–10°. Cannot resolve fine within-screen gaze (e.g. which paragraph is being read). |
| **Motivated deception** | Looking at the camera while mentally absent scores as FOCUSED. |
| **Dual monitors** | Legitimate gaze to a second monitor reads as DISTRACTED. |
| **Session warm-up** | Confidence builds over the first ~10 s; initial states should be treated as provisional. |
| **Population thresholds** | All EAR, yaw/pitch, and score thresholds are population averages. Per-user calibration would reduce false alarms. |
| **Glasses / extreme lighting** | Iris tracking degrades; the system falls back to head-pose-only gaze estimation. |
| **Display latency** | Stimulus onset timestamps capture when the frame was passed to OpenCV, not when photons reached the screen (~8–32 ms systematic offset). |
| **Frame-rate granularity** | Behavioral changes are detectable to the nearest frame (~33 ms at 30 FPS). |
| **EMA lag** | The EMA smoother adds ~200–400 ms latency to engagement state changes. Key-press latencies are unaffected; gaze-shift and attention-change latencies include this offset. |

**Results must not be used to make individual assessments of cognition, attention disorders, productivity, or mental state.**

---

## Ethics and privacy

- **All processing is local.** No video, audio, landmarks, or behavioral data is transmitted to any server. Nothing leaves the machine.
- **No persistent biometric storage.** Landmark data is processed in memory and discarded each frame. Session exports contain only aggregate behavioral metrics (scores, states, timestamps) — no images or raw landmark coordinates.
- **Opt-in only.** The system requires an explicit decision to run a live script. There is no background monitoring capability.
- **Informed use.** Any use of this system in a context where another person is monitored requires their informed consent. The session summary printed at script exit makes the data collection transparent.
- **This is not a medical or cognitive diagnostic tool.** Do not use outputs to make decisions about individuals' health, ability, or employment.

---

## Troubleshooting

### `python: command not found`
macOS does not have a `python` binary by default. Use `python3`, or activate the venv (`source .venv/bin/activate`) which adds a `python` alias.

### `ModuleNotFoundError: No module named 'cv2'`
The virtual environment is not active. Run `source .venv/bin/activate` and try again.

### Black or blank camera frames on macOS
AVFoundation emits blank frames during sensor warm-up. The pipeline discards them automatically (`min_frame_mean: 1.0` in config). If the problem persists, increase `warmup_frames` in `configs/default.yaml`. Check that Terminal has camera permission in **System Settings → Privacy & Security → Camera**.

### iPhone selected instead of built-in webcam
macOS Continuity Camera can claim device index 0. All live scripts print a camera candidate list on startup:
```
[Camera 0] iPhone Camera  (Continuity Camera) ← WARNING
[Camera 1] FaceTime HD Camera  640×480
```
Pass `--device 1` (or whichever index shows the FaceTime camera) to override.

### `[WARNING] MediaPipe face landmarker: no faces detected`
This is MediaPipe's internal logging, not an error. It appears when the face is briefly out of frame. Suppress it by setting `logging.level: WARNING` in `configs/default.yaml`.

### `Error: face_landmarker.task not found`
Run the model download command from the setup section. The model must be at `.cache/models/face_landmarker.task` (configurable via `face_mesh.model_path` in `configs/default.yaml`).

### Engagement score stuck at UNRELIABLE
Normal for the first ~10 seconds. The confidence score builds with session history (`min_history_s: 10.0`). Wait for the confidence bar to reach >0.20 before expecting meaningful state labels.

### Tests fail with import errors
Ensure the venv is active and you are running `python -m pytest`, not `pytest`. The latter may use the system Python which lacks the project dependencies.

---

## Roadmap

| Priority | Feature | Rationale |
|----------|---------|-----------|
| Next | Per-user EAR calibration | Population-average thresholds cause false alarms; baseline calibration during the first 30 s of a session would reduce these significantly |
| Next | Gaze-on-screen calibration | A simple 5-point screen calibration would map iris ratios to screen coordinates, enabling reading-line tracking |
| Medium | Webcam latency compensation | Measure and subtract the camera-to-display round-trip for more accurate reaction time estimates |
| Medium | Cross-session comparison | A session database layer to compare focused_fraction trends across days |
| Medium | `asyncio` / threaded inference | Decouple capture from inference to maintain frame rate under CPU load |
| Long | EEG / physiological fusion | Add heart rate (webcam rPPG) or EEG (OpenBCI) as a fourth orthogonal channel |
| Long | Adaptive stimulus generation | Use engagement score to dynamically adjust stimulus difficulty or pacing |
| Long | Multi-person support | Extend FacePipeline to track `max_faces > 1` and maintain per-subject state |

---

## Resume bullets

Use these to describe the project in a CV or cover letter:

- **Designed and implemented a 4-stage multimodal behavioral signal pipeline** (EAR/blink → solvePnP head pose → iris gaze → composite engagement) operating at 25–30 FPS on a MacBook CPU using MediaPipe, OpenCV, and NumPy
- **Built a stimulus-response experiment framework** supporting 4 stimulus types with ±33 ms temporal precision via monotonic clock synchronization across all signal channels; reaction latencies are stamped on the same clock as stimulus onsets
- **Implemented 3-signal fusion with EMA smoothing and state debouncing** — combining gaze direction (50%), eye alertness (30%), and head orientation (20%) into a composite engagement score with configurable confidence gating
- **Achieved 285 unit tests with zero webcam or display dependencies** by designing injectable mock signal objects; the full suite runs in ~1 second and is suitable for CI
- **Produced a complete offline analytics pipeline** that loads exported JSON/CSV sessions, computes 20+ behavioral metrics with distribution summaries, generates matplotlib charts, and exports self-contained HTML reports
- **Resolved macOS Continuity Camera device-index conflicts** via `system_profiler SPCameraDataType` integration with automated fallback warnings and per-script CLI override support

---

## Technical pitch

*For interviews or engineering conversations about attention-sensing systems:*

**The core problem:** Single-channel behavioral sensors (gaze alone, blink rate alone, head pose alone) all have high false-positive rates. Gaze degrades with glasses and head movement. Head pose can't distinguish looking-at-screen from staring at your coffee. Blink rate has too many confounders. The key insight behind this system is that these three channels are largely *uncorrelated in their noise* — so fusing them multiplicatively means the composite only approaches zero when all three channels agree that attention is absent.

**The interesting engineering problems:**

1. *Temporal synchronization* — every frame, landmark coordinate, and stimulus event is timestamped with `time.monotonic()`. Stimulus onset times and response times are on the same reference clock, so reaction latencies are simply `response_ts − onset_ts`. No NTP jitter, no wall-clock discontinuities.

2. *Signal architecture* — raw measurements (EAR, iris ratios, head angles) are separated from inferred states (BlinkAnalysis, GazeAnalysis, AttentionAnalysis) which are separated from session-level analytics (SessionMetrics). Each layer is independently testable and replaceable without breaking the others.

3. *Honest uncertainty quantification* — the confidence score degrades gracefully when signals are missing or unreliable (no iris landmarks → gaze weight penalty; high reprojection error → head penalty; short session → history penalty). A high-confidence LOW score means something; a low-confidence score means we don't know.

**Why this approaches research-grade tooling on a consumer device:** The system structures its output the same way a research paradigm would — baseline period, timed stimulus events, synchronized behavioral responses, per-trial baseline/during/recovery score comparisons. That structure is what allows you to ask causal questions ("did this specific stimulus type disrupt attention?") rather than passive monitoring questions ("was this person generally engaged?").

**Current limitations:** Frame-rate granularity limits temporal precision to ~33 ms. Display latency (~15 ms typical) is a systematic offset in reaction times. EMA smoothing adds 200–400 ms to state-change detection. Per-user calibration is not yet implemented. This is behavioral orientation inference, not cognitive engagement measurement — a person can look straight at the camera while mentally absent.
