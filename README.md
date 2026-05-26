# multimodal-response-lab

A real-time behavioral engagement analyzer built incrementally on a single webcam feed.
The system infers attention and fatigue from face geometry — no special hardware, no wearables.

## What it does

The pipeline runs at 25–30 FPS on a MacBook and produces, every frame:

| Output | Description |
|---|---|
| `EngagementState` | Categorical label: FOCUSED / DRIFTING / DISTRACTED / FATIGUED / UNRELIABLE |
| `smoothed_score` | Continuous score 0–1 (EMA-smoothed, suitable for time-series logging) |
| `focused_fraction` | Rolling % of recent frames classified FOCUSED (60 s window) |
| `confidence` | Data-quality gate — low during warm-up or poor visibility |
| Component scores | Gaze, head-pose, and eye-alertness sub-scores for debugging |

The engagement score fuses three orthogonal behavioral signals so that individual noise cancels out:

- **Gaze (50%)** — where the iris is pointing relative to the screen
- **Eye alertness (30%)** — EAR (eye aspect ratio) and blink patterns
- **Head orientation (20%)** — head-pose attention zone and rotational stability

---

## Signal pipeline

```
Raw pixels
  └─ FaceLandmarker (MediaPipe 0.10+, 478-point model)
       ├─ EAR / blink FSM          → BlinkAnalysis (rate, drowsy flag, fatigue label)
       ├─ solvePnP head pose       → PoseMeasurement (yaw, pitch, roll)
       │    └─ AttentionDetector   → AttentionAnalysis (zone, fraction, stability)
       ├─ Iris H/V ratios          → GazeMeasurement (H/V position per eye)
       │    └─ GazeDetector        → GazeAnalysis (zone, on-screen fraction, stability)
       └─ AttentionScorer          → EngagementScore (state, score, confidence, fractions)
```

Each layer is independent and gracefully handles missing upstream input (e.g. iris unreliable → falls back to head pose only for the gaze component).

---

## What you see in each live script

### `run_face_test.py` — Day 2: face mesh
Face contour, eye outlines, eyebrow outlines, and lip contour drawn over the webcam feed.
Validates that MediaPipe landmarks are being extracted correctly.

### `run_blink_test.py` — Day 3: blink detector
Adds a left-side HUD panel showing:
- Eye state badge (OPEN / CLOSING / CLOSED / OPENING)
- Left and right EAR values
- Openness bar
- Blink count, blink rate (per minute)
- Fatigue indicator: LOW / MODERATE / HIGH

### `run_head_pose_test.py` — Day 4: head pose + attention zone
Adds to the face mesh:
- Three projected 3D axes at the nose tip (X=red, Y=green, Z=blue)
- HUD panel with yaw/pitch/roll angles, attention zone, stability std-devs, reprojection error

Attention zones:
- `FOCUSED` (|yaw| ≤ 20°, |pitch| ≤ 15°)
- `GLANCE` (|yaw| ≤ 35°, |pitch| ≤ 25°)
- `LOOKING_AWAY` (beyond glance thresholds)

### `run_gaze_test.py` — Day 5: iris gaze estimation
Adds to the head pose view:
- Two cyan dots on each iris centre (turn red when EAR is too low for reliable detection)
- Gaze zone HUD: CENTER / LEFT / RIGHT / UP / DOWN / diagonal / UNRELIABLE
- On-screen heuristic: iris centred AND head within ±25° yaw / ±20° pitch → ON SCREEN
- Rolling on-screen fraction (last 5 s)
- H/V gaze stability (std-dev over 5 s)

### `run_attention_test.py` — Day 6: composite engagement dashboard
Full pipeline. Left side shows the engagement dashboard:

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

Right side retains the individual head-pose, gaze, and blink HUD panels from earlier days.

---

## Setup

**Requirements:** Python 3.9+, macOS (Apple Silicon or Intel) / Linux. A webcam is required to run the live scripts; tests run offline.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the MediaPipe FaceLandmarker model (once, ~3 MB)
mkdir -p .cache/models
curl -L -o .cache/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

**Apple Silicon note:** Install the CPU/MPS build of PyTorch, not the CUDA wheel:
```bash
pip install torch torchvision torchaudio
```

---

## Running the live scripts

```bash
# Day 2 — face mesh
python run_face_test.py

# Day 3 — blink detector
python run_blink_test.py

# Day 4 — head pose + attention zone
python run_head_pose_test.py

# Day 5 — iris gaze estimation
python run_gaze_test.py

# Day 6 — composite engagement dashboard  ← start here for the full pipeline
python run_attention_test.py
```

All scripts accept:

| Flag | Default | Description |
|---|---|---|
| `--device N` | 0 | Camera device index (override `capture.device_id` in config) |
| `--backend NAME` | avfoundation | OpenCV backend: `avfoundation` / `any` / `qt` / `dshow` / `v4l2` |
| `--debug` | off | Print per-frame signal values to the terminal |
| `--config PATH` | `configs/default.yaml` | Alternate config file |

Press **Q** or **Esc** to quit any script. A session summary is printed on exit.

**macOS Continuity Camera:** If an iPhone is detected as the default camera (index 0), the scripts print a warning and list all available cameras. Pass `--device 1` (or the correct index shown) to select the built-in FaceTime HD camera.

---

## Running tests

```bash
# All 162 tests
python -m pytest tests/ -q

# Single module
python -m pytest tests/test_engagement.py -v
```

Test modules:

| File | What it covers |
|---|---|
| `test_eye_metrics.py` | EAR normalisation, openness computation |
| `test_blink_detector.py` | Blink FSM, rate thresholds, fatigue labels |
| `test_head_pose.py` | solvePnP pose, Euler angles, attention zones |
| `test_gaze.py` | Iris H/V ratios, zone classification, on-screen heuristic |
| `test_engagement.py` | Component scores, state classification, EMA smoothing, debounce, confidence, rolling fractions |

All tests inject pre-built signal objects — no webcam or landmark data is needed.

---

## Configuration

All runtime parameters live in [`configs/default.yaml`](configs/default.yaml). Modules receive a config dict and never import the file directly, so overriding any value for testing is trivial.

Key sections:

```yaml
capture:
  device_id: 0        # camera index
  backend: avfoundation
  fps: 30
  flip_horizontal: true

signals:
  blink:
    ear_close_threshold: 0.20   # EAR below this → eye closing
    drowsy_closure_ms:   500.0  # closure longer than this → drowsy flag

head_pose:
  attention:
    yaw_focus_deg:   20.0   # |yaw| within this → FOCUSED zone
    pitch_focus_deg: 15.0

gaze:
  center_h_lo: 0.35   # iris H-ratio band for CENTER zone
  center_h_hi: 0.65
  head_yaw_max_on_screen: 25.0   # head-pose gate for is_on_screen

engagement:
  weights:
    gaze: 0.50
    head: 0.20
    eye:  0.30
  thresholds:
    focused:   0.72   # smoothed score >= this → FOCUSED
    drifting:  0.45   # score >= this → DRIFTING (else DISTRACTED)
  smoothing:
    alpha:            0.10   # EMA decay; ~0.4 s half-life at 25 FPS
    state_min_frames: 5      # debounce: candidate must persist this many frames
  confidence:
    min_history_s: 10.0     # seconds before confidence reaches 1.0
  rolling_window_s: 60.0    # window for focused_fraction / engaged_fraction
```

---

## Project structure

```
multimodal-response-lab/
├── configs/
│   └── default.yaml              # all runtime parameters
├── src/
│   ├── pipelines/
│   │   └── face_pipeline.py      # MediaPipe FaceLandmarker wrapper (threaded capture)
│   ├── signals/
│   │   ├── eye_metrics.py        # EAR computation + normalisation
│   │   ├── blink_detector.py     # blink FSM, rate, drowsiness, fatigue
│   │   ├── head_pose.py          # solvePnP pose estimation
│   │   ├── attention.py          # attention zone classifier + stability
│   │   ├── gaze.py               # iris H/V ratios, gaze zone, on-screen heuristic
│   │   └── engagement.py         # composite engagement scorer (AttentionScorer)
│   └── utils/
│       ├── camera.py             # camera enumeration, Continuity Camera detection
│       ├── overlay.py            # HUD rendering: mesh, axes, metrics, dashboard
│       ├── landmarks.py          # landmark index constants and helpers
│       └── device.py             # CUDA / MPS / CPU device selection
├── tests/                        # 162 offline unit tests
├── run_face_test.py              # Day 2 live script
├── run_blink_test.py             # Day 3 live script
├── run_head_pose_test.py         # Day 4 live script
├── run_gaze_test.py              # Day 5 live script
└── run_attention_test.py         # Day 6 live script (full pipeline)
```

---

## Design principles and limitations

**This is approximate behavioral inference, not cognitive measurement.**
The system infers whether someone is *oriented toward the screen*, not whether they are thinking or comprehending.

Known failure modes:
- **Motivated deception** — looking at the camera while mentally absent scores as FOCUSED
- **Dual monitors** — legitimate off-screen gaze is classified as DISTRACTED
- **Glasses / extreme lighting** — iris tracking degrades; falls back to head-pose-only gaze
- **Session warm-up** — confidence builds over the first 10 s; initial states should be treated as provisional
- **Population averages** — all thresholds are generic; per-user calibration would reduce false alarms

**Why fuse three signals?**
Each channel has a high false-positive rate alone. Gaze, head orientation, and blink patterns are largely uncorrelated in their noise, so their product approaches zero only when all three simultaneously indicate distraction.

**Why smooth and debounce?**
A single blink drops the eye score to near zero for ~100 ms. EMA (α=0.10, ~0.4 s half-life) absorbs transients; state debouncing (5 frames) prevents the displayed badge from flickering on single-frame anomalies. Genuine attention changes take seconds to develop and are still captured promptly.
