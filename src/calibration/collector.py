"""Raw signal accumulator for calibration sessions.

CalibrationSample is intentionally simpler than BehavioralSample:
  - No stimulus metadata (calibration has no stimuli)
  - No engagement state (state classification uses population thresholds;
    calibration is specifically needed to set personal thresholds)
  - Just the raw signal values that the profiler needs to compute means/stds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CalibrationSample:
    """One frame of raw signal measurements captured during calibration.

    All fields use the same units as the rest of the pipeline so that the
    profiler can directly compare calibration stats to live-session values.
    """
    timestamp:       float          # monotonic time (same clock as BehavioralSample)
    ear:             float          # mean EAR across both eyes [0, ~0.4]
    blink_rate:      float          # rolling blink rate at this frame (blinks/min)
    head_yaw:        float          # head yaw in degrees
    head_pitch:      float          # head pitch in degrees
    gaze_h:          float          # normalised horizontal iris position [0, 1]
    gaze_v:          float          # normalised vertical iris position [0, 1]
    engagement_score: float         # composite engagement score [0, 1]
    is_face_detected: bool
    is_eyes_open:    bool           # True when EAR > close_threshold (used for open-EAR baseline)


class CalibrationCollector:
    """Accumulates CalibrationSamples during a calibration session.

    Thread-safety: not thread-safe; call from a single inference thread.
    """

    def __init__(self) -> None:
        self._samples: List[CalibrationSample] = []

    def add(self, sample: CalibrationSample) -> None:
        """Append *sample* to the collection."""
        self._samples.append(sample)

    @property
    def samples(self) -> List[CalibrationSample]:
        """All collected samples (read-only view)."""
        return list(self._samples)

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def n_face_detected(self) -> int:
        return sum(1 for s in self._samples if s.is_face_detected)

    @property
    def duration_s(self) -> float:
        """Elapsed time from first to last sample in seconds."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1].timestamp - self._samples[0].timestamp

    @property
    def face_detection_rate(self) -> float:
        """Fraction of frames where a face was detected."""
        if not self._samples:
            return 0.0
        return self.n_face_detected / len(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def valid_samples(self) -> List[CalibrationSample]:
        """Samples where a face was detected — used by the profiler."""
        return [s for s in self._samples if s.is_face_detected]

    def open_eye_samples(self) -> List[CalibrationSample]:
        """Samples where eyes were open — for EAR baseline computation."""
        return [s for s in self._samples if s.is_face_detected and s.is_eyes_open]
