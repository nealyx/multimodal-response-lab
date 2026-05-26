"""EventRecorder — collects all session events into a SessionLog.

Threading note
--------------
The live experiment loop runs entirely on the main thread (cv2.imshow is not
thread-safe), so simple list appends are safe here.  If a future version adds a
background inference thread, replace list with a deque + threading.Lock.

Response detection philosophy
------------------------------
Key-press responses are detected explicitly (the caller passes the key and ts).
Gaze and attention responses are detected by comparing consecutive frame
states — they require the previous frame's state to detect a transition, which
the recorder tracks.

Why track prev_on_screen and prev_eng_state here rather than in the run script?
Because the recorder is the single owner of "what changed" logic.  The run
script just calls record_sample() and the recorder decides whether a gaze_shift
or attention_change response event should be emitted.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.stimulus.events import (
    BehavioralSample, ResponseEvent, SessionLog, StimulusEvent,
)

log = logging.getLogger(__name__)


class EventRecorder:
    """Accumulates all events for one experiment session."""

    def __init__(self, session_log: SessionLog) -> None:
        self._log = session_log
        # Previous-frame state for transition detection
        self._prev_on_screen:  Optional[bool] = None
        self._prev_eng_state:  Optional[str]  = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def session_log(self) -> SessionLog:
        return self._log

    # ── Stimulus events ───────────────────────────────────────────────────────

    def record_stimulus_onset(self, event: StimulusEvent) -> None:
        self._log.stimulus_events.append(event)
        log.debug("record onset  id=%s  ts=%.3f", event.stimulus_id, event.onset_ts)

    def record_stimulus_offset(self, event: StimulusEvent) -> None:
        # The StimulusEvent object is already in the list (appended at onset);
        # its offset_ts is mutated in place by the presenter.  We just log it.
        log.debug(
            "record offset  id=%s  duration=%.0f ms",
            event.stimulus_id,
            event.actual_duration_ms or 0.0,
        )

    # ── Response events ───────────────────────────────────────────────────────

    def record_response(self, event: ResponseEvent) -> None:
        self._log.response_events.append(event)
        log.debug(
            "record response  stim=%s  type=%s  latency=%.0f ms",
            event.stimulus_id, event.response_type, event.latency_ms,
        )

    def record_key_response(
        self,
        active_event: StimulusEvent,
        ts: float,
    ) -> ResponseEvent:
        """Emit a key_press ResponseEvent for the active ReactionPrompt."""
        latency_ms = (ts - active_event.onset_ts) * 1000.0
        ev = ResponseEvent(
            stimulus_id=  active_event.stimulus_id,
            trial_index=  active_event.trial_index,
            response_type="key_press",
            response_ts=  ts,
            latency_ms=   latency_ms,
            hit=          latency_ms >= 0,
            metadata=     {},
        )
        self.record_response(ev)
        return ev

    # ── Behavioral samples + implicit response detection ──────────────────────

    def record_sample(
        self,
        sample:       BehavioralSample,
        active_event: Optional[StimulusEvent],
    ) -> None:
        """Append the sample and check for implicit behavioral responses.

        Implicit responses detected:
          gaze_shift      — is_on_screen flips from True to False during a trial
          attention_change — engagement_state changes during a trial
        """
        self._log.samples.append(sample)

        if active_event is not None:
            # Gaze shift: on-screen → off-screen
            if (
                self._prev_on_screen is True
                and not sample.is_on_screen
            ):
                latency_ms = (sample.timestamp - active_event.onset_ts) * 1000.0
                if latency_ms >= 0:
                    self.record_response(ResponseEvent(
                        stimulus_id=  active_event.stimulus_id,
                        trial_index=  active_event.trial_index,
                        response_type="gaze_shift",
                        response_ts=  sample.timestamp,
                        latency_ms=   latency_ms,
                        hit=          True,
                        metadata=     {"direction": "off_screen"},
                    ))

            # Attention state change
            if (
                self._prev_eng_state is not None
                and self._prev_eng_state != sample.engagement_state
            ):
                latency_ms = (sample.timestamp - active_event.onset_ts) * 1000.0
                if latency_ms >= 0:
                    self.record_response(ResponseEvent(
                        stimulus_id=  active_event.stimulus_id,
                        trial_index=  active_event.trial_index,
                        response_type="attention_change",
                        response_ts=  sample.timestamp,
                        latency_ms=   latency_ms,
                        hit=          True,
                        metadata={
                            "from_state": self._prev_eng_state,
                            "to_state":   sample.engagement_state,
                        },
                    ))

        self._prev_on_screen = sample.is_on_screen
        self._prev_eng_state = sample.engagement_state
