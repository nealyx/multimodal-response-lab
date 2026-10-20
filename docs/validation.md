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

## Continuous checks

The [Offline checks workflow](../.github/workflows/offline.yml) defines four jobs:
the full suite and a minimal-dependency synthetic walkthrough, each on
`ubuntu-latest` and `macos-latest` with Python 3.12. Every job creates a fresh
virtual environment, installs the appropriate requirements file, checks
dependency consistency, and records installed versions in its log.

After installation, the workflow enables a Python socket guard through
`PYTHONPATH`, including in child Python processes. The minimal walkthrough also
rejects imports of OpenCV, MediaPipe, PyTorch, sounddevice, and pyaudio. It
generates the synthetic session, reruns normal reporting and feature CLIs, then
checks provenance, frame/window counts, prescribed latencies, and output files.
The guard catches accidental Python network calls; it is not an operating-system
network sandbox. Live capture and native-library network behavior are outside
this check.

The badge links to [actual hosted runs](https://github.com/nealyx/multimodal-response-lab/actions/workflows/offline.yml).
The local CI rehearsal passed **543 tests with the same four warnings**, plus the
minimal walkthrough and explicit checks that the network/import guard rejects
access and propagates to child Python processes.

The [first hosted run](https://github.com/nealyx/multimodal-response-lab/actions/runs/39440536448)
completed successfully on 20 October 2026 at commit `bb8636e`:

| Hosted job | Observed result |
|---|---|
| [Full suite · Ubuntu 24.04.5](https://github.com/nealyx/multimodal-response-lab/actions/runs/39440536448/job/118688876288) | 543 passed, four existing convergence warnings, zero skips |
| [Full suite · macOS 26 arm64](https://github.com/nealyx/multimodal-response-lab/actions/runs/39440536448/job/118688877326) | 543 passed, the same four warnings, zero skips |
| [Minimal demo · Linux](https://github.com/nealyx/multimodal-response-lab/actions/runs/39440536448/job/118688876309) | Demo, reanalysis, feature extraction, and artifact checks passed |
| [Minimal demo · macOS](https://github.com/nealyx/multimodal-response-lab/actions/runs/39440536448/job/118688875869) | Demo, reanalysis, feature extraction, and artifact checks passed |

The hosted jobs used CPython 3.12; their logs record the exact patch version,
runner image, and resolved packages. `*-latest` runner images and dependency
floors will move over time. These results describe the linked run, not all
possible environments. The workflow pins checkout v7.0.1 and setup-python v7.0.0
to their official release commit hashes.

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

Live camera/display capture, microphone input, MediaPipe inference/model downloads, angular error, real reaction timing, Windows execution, survey administration, participant-based model evaluation, and calibrated uncertainty were not tested. Linux and macOS hosted checks cover only the offline paths listed above. The live model URL still references `latest`; exact live reproduction needs an archived model hash. No license was added or inferred.
