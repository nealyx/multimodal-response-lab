"""FSM-based blink detection with rolling rate and fatigue indicators.

Why a state machine instead of a raw threshold
-----------------------------------------------
A simple "EAR < 0.20 → blink" rule fails in three common ways:

1.  Jitter false positives.  Even a fully-open eye has frame-to-frame EAR
    variation of ±0.02 from landmark instability.  A single low frame fires
    a "blink" every few seconds on a completely still face.

2.  No duration information.  A threshold cannot distinguish a normal blink
    (80–200 ms), a slow deliberate close (200–400 ms), and a prolonged
    closure (> 500 ms) that may indicate drowsiness.

3.  No hysteresis.  At threshold boundary, the signal bounces on/off on
    consecutive frames, producing multiple spurious "blinks" per real event.

The FSM fixes all three:
    - Debounce:   require N consecutive frames below threshold to confirm closure.
    - Hysteresis: separate close (lower) and open (higher) thresholds prevent
                  boundary bouncing.
    - Duration gate: only register a blink if the closed interval falls within
                  [min_blink_ms, max_blink_ms].

FSM states and transitions
---------------------------

        EAR < close_thr           N frames below
    ┌──────────────────► CLOSING ──────────────────► CLOSED
    │                      │                            │
  OPEN ◄──────────────────┘                            │ EAR ≥ open_thr
    ▲       EAR ≥ close_thr                             │
    │       (false alarm)                               ▼
    │                                                OPENING
    │                                                   │
    └─────────────── M frames above ───────────────────┘
         (register blink if duration in [min, max])
              EAR < open_thr from OPENING → back to CLOSED

Why rolling blink rate matters
-------------------------------
Cumulative blink count is monotone and tells you nothing about the current
moment.  Rate over a rolling window (default 60 s) captures the *present*
behavioral pattern:

    Healthy adult:   ~15–20 blinks / minute
    Screen focus:    ~3–8 / minute  ← suppressed blink rate, early fatigue sign
    Eye strain:      ~25–35 / minute ← compensating for dryness
    Possible sleep:  0 blinks in > 10 s window

A 60-second window smooths individual burst variations while responding to
real behavioral shifts within ~1 minute.

What can cause false positives
-------------------------------
    1.  Head rotation > ~30°: landmark accuracy degrades → EAR dips
    2.  Lighting transitions: camera auto-exposure lag causes transient blur
    3.  Glasses reflections: systematic EAR bias on the affected eye
    4.  Partial occlusion: hand, hair, or out-of-frame
    5.  Yawning: mouth opening causes facial deformation that can move eye landmarks

Mitigations: the duration gate (no sub-50 ms "blinks"), the debounce counter,
and per-eye vs mean-EAR selection all reduce these artefacts.

Config keys (under signals.blink in default.yaml)
--------------------------------------------------
    ear_close_threshold:       0.20   EAR below this starts closure
    ear_open_threshold:        0.25   EAR above this starts opening (hysteresis gap)
    min_close_frames:          2      debounce for closure confirmation
    min_open_frames:           2      debounce for re-open confirmation
    min_blink_ms:              60.0   reject closures shorter than this
    max_blink_ms:              500.0  closures longer → prolonged / drowsy
    rate_window_s:             60.0   rolling window for blinks-per-minute
    low_blink_rate_threshold:  8.0    below → staring / early fatigue
    high_blink_rate_threshold: 25.0   above → eye strain
    drowsy_closure_ms:         500.0  single closure longer than this → drowsy flag
    min_history_s:             15.0   ignore rate flags before this many seconds
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.signals.eye_metrics import EyeMeasurement

logger = logging.getLogger(__name__)


# ── State enum ────────────────────────────────────────────────────────────────

class BlinkState(str, Enum):
    OPEN    = "open"
    CLOSING = "closing"   # below close_thr, accumulating debounce
    CLOSED  = "closed"    # confirmed closed; duration accumulating
    OPENING = "opening"   # above open_thr, accumulating debounce


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlinkEvent:
    """A single confirmed blink (eye fully closed then fully re-opened)."""
    timestamp:   float   # time.perf_counter() when the eye re-opened
    duration_ms: float   # milliseconds between close-confirmed and open-confirmed


@dataclass
class BlinkAnalysis:
    """Inferred behavioral snapshot at one frame.

    Combines raw measurements with FSM-derived state and computed fatigue flags.
    """
    state:                BlinkState
    blink_count:          int
    blink_rate_per_min:   float
    mean_ear:             float
    mean_openness:        float
    ms_since_last_blink:  Optional[float]   # None until first blink
    is_low_blink_rate:    bool
    is_high_blink_rate:   bool
    is_prolonged_closure: bool              # single closure > drowsy_closure_ms
    is_drowsy:            bool              # compound indicator
    timestamp:            float
    has_sufficient_data:  bool = True       # False during min_history_s warmup

    @property
    def fatigue_label(self) -> str:
        if self.is_drowsy:
            return "HIGH"
        if self.is_low_blink_rate or self.is_high_blink_rate or self.is_prolonged_closure:
            return "MODERATE"
        return "LOW"


# ── FSM ───────────────────────────────────────────────────────────────────────

class BlinkDetector:
    """Finite-state-machine blink detector.

    Call update() once per frame with an EyeMeasurement.
    Call no_face_update() when the face was not detected in a frame — this
    avoids corrupting the FSM state with missing data.

    Not thread-safe: call from one thread only (the same thread as inference).
    """

    def __init__(self, cfg: dict) -> None:
        bc = cfg.get("blink", {})

        # Thresholds
        self._ear_close: float = bc.get("ear_close_threshold", 0.20)
        self._ear_open:  float = bc.get("ear_open_threshold",  0.25)

        # Debounce frame counts
        self._min_close_frames: int = bc.get("min_close_frames", 2)
        self._min_open_frames:  int = bc.get("min_open_frames",  2)

        # Duration gate (ms)
        self._min_blink_ms: float = bc.get("min_blink_ms",   60.0)
        self._max_blink_ms: float = bc.get("max_blink_ms",  500.0)

        # Rolling window
        self._rate_window_s: float = bc.get("rate_window_s", 60.0)

        # Fatigue thresholds
        self._low_rate_thr:    float = bc.get("low_blink_rate_threshold",   8.0)
        self._high_rate_thr:   float = bc.get("high_blink_rate_threshold",  25.0)
        self._drowsy_close_ms: float = bc.get("drowsy_closure_ms",         500.0)
        self._min_history_s:   float = bc.get("min_history_s",              15.0)

        # FSM internal state
        self._state:          BlinkState    = BlinkState.OPEN
        self._frames_closing: int           = 0
        self._frames_opening: int           = 0
        self._close_ts:       Optional[float] = None   # perf_counter when CLOSED entered

        # History
        self._blink_count:  int                  = 0
        self._blink_events: deque[BlinkEvent]    = deque()
        self._start_ts:     Optional[float]      = None  # set on first update
        self._last_analysis: Optional[BlinkAnalysis] = None

    # ── Public interface ───────────────────────────────────────────────────────

    def update(self, m: EyeMeasurement) -> BlinkAnalysis:
        """Advance FSM and return updated analysis.  Called once per frame."""
        prev_state = self._state
        self._tick(m.mean_ear, m.timestamp)

        if self._state != prev_state:
            logger.debug(
                "blink FSM: %s → %s  EAR=%.3f  count=%d",
                prev_state.value, self._state.value, m.mean_ear, self._blink_count,
            )

        analysis = self._build_analysis(m.mean_ear, m.mean_openness, m.timestamp)
        self._last_analysis = analysis
        return analysis

    def no_face_update(self) -> Optional[BlinkAnalysis]:
        """Return last known analysis without advancing the FSM.

        Advancing the FSM on missing data would corrupt accumulated state —
        e.g. a single missed frame could reset a legitimate prolonged closure.
        """
        return self._last_analysis

    def reset(self) -> None:
        """Reset FSM to OPEN.  Call when face is lost for several consecutive frames."""
        self._state          = BlinkState.OPEN
        self._frames_closing = 0
        self._frames_opening = 0
        self._close_ts       = None
        logger.debug("blink FSM reset to OPEN")

    # ── FSM tick ──────────────────────────────────────────────────────────────

    def _tick(self, ear: float, now: float) -> None:
        """Single-frame FSM advance.  All state mutation lives here."""

        if self._state in (BlinkState.OPEN, BlinkState.CLOSING):
            if ear < self._ear_close:
                self._frames_closing += 1
                if self._frames_closing >= self._min_close_frames:
                    # Closure confirmed.
                    self._state          = BlinkState.CLOSED
                    self._close_ts       = now
                    self._frames_closing = 0
                    self._frames_opening = 0
                else:
                    self._state = BlinkState.CLOSING
            else:
                # EAR recovered before debounce completed — false alarm.
                if self._frames_closing > 0:
                    logger.debug(
                        "blink FSM: closure cancelled after %d frame(s)  EAR=%.3f",
                        self._frames_closing, ear,
                    )
                self._frames_closing = 0
                self._state = BlinkState.OPEN

        elif self._state in (BlinkState.CLOSED, BlinkState.OPENING):
            if ear >= self._ear_open:
                self._frames_opening += 1
                if self._frames_opening >= self._min_open_frames:
                    # Re-open confirmed → potential blink.
                    self._state          = BlinkState.OPEN
                    self._frames_opening = 0
                    self._frames_closing = 0
                    self._maybe_register_blink(now)
                else:
                    self._state = BlinkState.OPENING
            else:
                # EAR dropped again before debounce completed — still closed.
                if self._state == BlinkState.OPENING:
                    logger.debug("blink FSM: re-close during OPENING  EAR=%.3f", ear)
                self._frames_opening = 0
                self._state = BlinkState.CLOSED

    def _maybe_register_blink(self, now: float) -> None:
        """Record a BlinkEvent if duration falls within the valid blink window."""
        if self._close_ts is None:
            return

        duration_ms   = (now - self._close_ts) * 1000.0
        self._close_ts = None

        if duration_ms < self._min_blink_ms:
            logger.debug(
                "blink candidate rejected: %.0f ms < min %.0f ms",
                duration_ms, self._min_blink_ms,
            )
            return

        if duration_ms > self._max_blink_ms:
            logger.debug(
                "prolonged closure: %.0f ms (max %.0f ms) — not counted as blink",
                duration_ms, self._max_blink_ms,
            )
            return

        self._blink_count += 1
        self._blink_events.append(BlinkEvent(timestamp=now, duration_ms=duration_ms))
        logger.debug("blink #%d: %.0f ms", self._blink_count, duration_ms)

    # ── Analysis builder ──────────────────────────────────────────────────────

    def _build_analysis(
        self,
        mean_ear:      float,
        mean_openness: float,
        now:           float,
    ) -> BlinkAnalysis:
        # Prune events outside the rolling window before computing rate.
        cutoff = now - self._rate_window_s
        while self._blink_events and self._blink_events[0].timestamp < cutoff:
            self._blink_events.popleft()

        if self._start_ts is None:
            self._start_ts = now
        elapsed_s = max(0.0, now - self._start_ts)

        # Use elapsed time as denominator during warmup to avoid artificially
        # low rates (e.g. 1 blink in 5 s reported as 1/min instead of 12/min).
        # Once elapsed exceeds the full window, use the window for consistency.
        effective_window = min(max(elapsed_s, 0.1), self._rate_window_s)
        rate = len(self._blink_events) / effective_window * 60.0

        ms_since: Optional[float] = None
        if self._blink_events:
            ms_since = (now - self._blink_events[-1].timestamp) * 1000.0

        # Only flag rate-based anomalies after enough history has accumulated.
        rate_flags_active = elapsed_s >= self._min_history_s

        prolonged = (
            self._state == BlinkState.CLOSED
            and self._close_ts is not None
            and (now - self._close_ts) * 1000.0 > self._drowsy_close_ms
        )

        low_rate  = rate_flags_active and rate < self._low_rate_thr
        high_rate = rate_flags_active and rate > self._high_rate_thr
        drowsy    = prolonged or (
            rate_flags_active
            and rate < self._low_rate_thr / 2.0
            and self._blink_count > 10
        )

        return BlinkAnalysis(
            state=                self._state,
            blink_count=          self._blink_count,
            blink_rate_per_min=   rate,
            mean_ear=             mean_ear,
            mean_openness=        mean_openness,
            ms_since_last_blink=  ms_since,
            is_low_blink_rate=    low_rate,
            is_high_blink_rate=   high_rate,
            is_prolonged_closure= prolonged,
            is_drowsy=            drowsy,
            timestamp=            now,
            has_sufficient_data=  rate_flags_active,
        )
