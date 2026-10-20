# Validation record — 20 October 2026

These checks validate software behavior. They do not validate attention, cognitive states, clinical meaning, participant generalization, webcam accuracy, or live throughput.

## Observed checks

| Check | Actual result |
|---|---|
| Existing suite before changes | 538 passed; four K-Means convergence warnings |
| Final suite, `python -m pytest tests/ -q -p no:cacheprovider` | **543 passed in 10.62 s**; the same four warnings; zero skips |
| Fresh offline-only environment | Installed only `requirements-offline.txt`; OpenCV, MediaPipe, PyTorch, sounddevice, and pyaudio were absent |
| `python demo_offline.py` | Completed: 600 samples, 60 simulated seconds, two scripted events/responses, five PNG charts, HTML, JSON, and CSV |
| Network/hardware import boundary | End-to-end subprocess test rejected network attempts and imports of vision/audio hardware packages; demo still completed |
| Deterministic input fixture | Two generated `SessionLog` objects were identical; confidence stayed separate from numeric scores during warmup and face loss |
| Overwrite protection | Rerunning the demo into an existing output directory failed without replacing its manifest |
| Relative-path `analyze_session.py --open` | Mocked browser received an absolute file URI; no real browser was opened by the test |
| Calibration CLI configuration | Default resolves to the checked-in `configs/default.yaml`; camera function replaced by a test stub |
| Normal report CLI | Reanalyzed `outputs/demo/session` and generated five charts and HTML |
| Feature CLI | Extracted 116 synthetic windows with 18 features and generated embedding artifacts |
| Offline CLI help | Dataset, classifier, and calibration help completed without the vision stack |
| Visual inspection | Generated timeline is readable and explicitly labeled synthetic; the README preview is that generated PNG |
| Whitespace and documentation links | `git diff --check` and local Markdown-target checks passed |

The four warnings come from tests that intentionally provide duplicate feature vectors while requesting more clusters than distinct points. They were also present in the unchanged baseline. Reported wall time is one local run, not a performance benchmark or CI guarantee.

## Environment

- macOS, Apple Silicon; CPython 3.12.14.
- NumPy 2.5.4, matplotlib 3.11.2, scikit-learn 1.9.1, SciPy 1.18.1.
- Full test environment additionally used OpenCV headless 5.0.0.93 and pytest 9.1.1.
- Exact installed package snapshot: [`validation-environment.txt`](validation-environment.txt).
- No MediaPipe model, personal recordings, profiles, survey responses, or participant catalog was used.

The small requirements files specify compatibility floors, not a frozen cross-platform environment. The package snapshot records this tested installation; it is not evidence that every permitted version or platform works.

## Reproduce

From a fresh checkout, create separate environments as shown in the README:

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q -p no:cacheprovider
```

For an installation without native vision dependencies:

```bash
python -m pip install -r requirements-offline.txt
python demo_offline.py --output-dir outputs/demo-check
python analyze_session.py outputs/demo-check/session --output-dir outputs/reanalysis-check
python embed_session.py outputs/demo-check/session --output-dir outputs/embeddings-check
```

Do not install the test OpenCV wheel into the live GUI environment. The synthetic fixture uses fixed relative times and prescribed signals; its 350/450 ms key latencies are chosen values. `generated_at` fields vary, and figure bytes can vary with plotting versions.

## Changes covered

- Replaced stale README milestones and unsupported test/FPS/accuracy claims with the implemented feature map and scoped limits.
- Corrected weighted-sum fusion versus multiplicative confidence descriptions in the README, architecture, and scorer documentation.
- Documented the implemented baseline/survey/offline-comparison workflow separately from unimplemented automatic live personalization.
- Made rendering package exports lazy and pose annotations type-only so the offline path does not import OpenCV.
- Added a synthetic walkthrough and explicit HTML/chart provenance notices.
- Removed the report's causal interpretation of a negative event-relative score change.
- Fixed relative output paths for `--open` and the calibration CLI's misspelled default config directory.

## Not verified in this pass

Live camera/display capture, microphone input, MediaPipe inference/model downloads, angular error, real reaction timing, Linux/Windows execution, external CI, survey administration, participant-based model evaluation, and calibrated uncertainty were not tested. The live model URL still references `latest`; exact live reproduction needs an archived model hash. No license was added or inferred.
