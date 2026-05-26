"""StimulusPresenter — schedules and renders visual stimuli during a session.

Why separate scheduling from rendering?
----------------------------------------
The schedule (which stimuli appear, in what order, at what times) is computed
once at session start from the experiment config.  The renderer simply checks
whether the current monotonic timestamp has crossed a trial's start time.  This
keeps the hot-path update() method free of allocation and branching logic.

Timing model
------------
All times are absolute monotonic (time.monotonic()).  Trial start times are
pre-computed as:

    start_ts = t0 + baseline_s + Σ(prev_durations_s + ISI_s)

where ISI includes optional uniform jitter so consecutive trials don't form a
predictable rhythm (which would allow anticipatory responses).

The presenter does NOT detect key presses or compute latency — that is the
EventRecorder's responsibility.  The presenter only knows what is on screen.

Dropped frames
--------------
If a frame is dropped and the update() call is delayed, the elapsed_ms
calculation inside render() still uses the actual monotonic onset time, so
animated stimuli (moving target, pulsing prompt) self-correct without any
special handling.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.stimulus.events import StimulusEvent
from src.stimulus.stimuli import (
    ColorFlash, MovingTarget, ReactionPrompt, ShapeStimulus, Stimulus,
)

log = logging.getLogger(__name__)


# ── Schedule data structure ───────────────────────────────────────────────────

@dataclass
class ScheduledTrial:
    trial_index: int
    start_ts:    float     # absolute monotonic onset time
    stimulus:    Stimulus


# ── Presenter ─────────────────────────────────────────────────────────────────

class StimulusPresenter:
    """Manages the trial schedule and renders stimuli onto frames.

    Call update(frame, ts) once per frame.  The presenter handles onset and
    offset transitions automatically.
    """

    def __init__(self, trials: List[ScheduledTrial]) -> None:
        self._trials          = sorted(trials, key=lambda t: t.start_ts)
        self._next_idx        = 0
        self._active_trial:   Optional[ScheduledTrial] = None
        self._active_onset_ts: float = 0.0
        self._active_event:   Optional[StimulusEvent]  = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def active_stimulus_id(self) -> Optional[str]:
        return self._active_trial.stimulus.stimulus_id if self._active_trial else None

    @property
    def active_event(self) -> Optional[StimulusEvent]:
        return self._active_event

    @property
    def trials_remaining(self) -> int:
        return max(0, len(self._trials) - self._next_idx)

    def is_finished(self) -> bool:
        return self._next_idx >= len(self._trials) and self._active_trial is None

    def update(
        self,
        frame: np.ndarray,
        ts:    float,
    ) -> Tuple[np.ndarray, Optional[StimulusEvent], Optional[StimulusEvent]]:
        """Advance the schedule and render the active stimulus.

        Returns:
            (rendered_frame, onset_event, offset_event)
            onset_event  — StimulusEvent whose offset_ts was just set to ts
            offset_event — StimulusEvent that just started this frame
        Both may be non-None in the same call if one trial ends and another
        starts simultaneously (rare but handled cleanly).
        """
        onset_event:  Optional[StimulusEvent] = None
        offset_event: Optional[StimulusEvent] = None

        # ── Expire current stimulus if duration elapsed ───────────────────
        if self._active_trial is not None:
            elapsed_ms = (ts - self._active_onset_ts) * 1000.0
            if elapsed_ms >= self._active_trial.stimulus.duration_ms:
                self._active_event.offset_ts = ts
                offset_event = self._active_event
                log.debug(
                    "stimulus offset  id=%s  actual=%.0f ms",
                    self._active_event.stimulus_id, elapsed_ms,
                )
                self._active_trial = None
                self._active_event = None

        # ── Start next trial if its scheduled time has arrived ────────────
        if (
            self._active_trial is None
            and self._next_idx < len(self._trials)
            and ts >= self._trials[self._next_idx].start_ts
        ):
            trial = self._trials[self._next_idx]
            self._next_idx       += 1
            self._active_trial    = trial
            self._active_onset_ts = ts
            self._active_event    = StimulusEvent(
                stimulus_id=  trial.stimulus.stimulus_id,
                trial_index=  trial.trial_index,
                stimulus_type=trial.stimulus.stimulus_type,
                onset_ts=     ts,
                offset_ts=    None,
                duration_ms=  trial.stimulus.duration_ms,
                metadata=     trial.stimulus.metadata(),
            )
            onset_event = self._active_event
            log.debug(
                "stimulus onset  id=%s  type=%s  ts=%.3f  remaining=%d",
                trial.stimulus.stimulus_id,
                trial.stimulus.stimulus_type,
                ts,
                self.trials_remaining,
            )

        # ── Render ────────────────────────────────────────────────────────
        if self._active_trial is not None:
            elapsed_ms = (ts - self._active_onset_ts) * 1000.0
            frame = self._active_trial.stimulus.render(frame, elapsed_ms)

        return frame, onset_event, offset_event

    def notify_key(self, key: int, ts: float) -> Optional[str]:
        """Return the response_key if this key matches the active ReactionPrompt.

        Returns None when no ReactionPrompt is active or the key doesn't match.
        """
        if self._active_trial is None:
            return None
        stim = self._active_trial.stimulus
        if not isinstance(stim, ReactionPrompt):
            return None
        key_map = {32: "space", 13: "enter", 27: "esc"}
        pressed = key_map.get(key) or (chr(key) if 0 < key < 128 else None)
        if pressed == stim.response_key:
            return stim.response_key
        return None


# ── Schedule builder ──────────────────────────────────────────────────────────

def build_schedule(cfg: Dict[str, Any], t0: float) -> List[ScheduledTrial]:
    """Construct a flat, chronologically ordered list of ScheduledTrials.

    t0 is the monotonic session start time.  Trial onset times are:
        t0 + baseline_s + Σ(prev_duration_s + ISI_s ± jitter)

    Trials are ordered as they appear in cfg["experiment"]["stimuli"], repeated
    n_trials times each.  Set randomise_order: true to shuffle the flat list.
    """
    exp = cfg.get("experiment", {})
    baseline_s   = float(exp.get("baseline_duration_s",       10.0))
    isi_s        = float(exp.get("inter_stimulus_interval_s",  5.0))
    isi_jitter_s = float(exp.get("isi_jitter_s",               1.0))

    # Flatten stimuli × n_trials into individual specs
    flat: List[Tuple[Dict[str, Any], int]] = []
    for spec in exp.get("stimuli", []):
        n = max(1, int(spec.get("n_trials", 1)))
        for i in range(n):
            flat.append((spec, i))

    if exp.get("randomise_order", False):
        random.shuffle(flat)

    trials: List[ScheduledTrial] = []
    cursor = t0 + baseline_s

    for global_idx, (spec, local_idx) in enumerate(flat):
        stim_type = spec.get("type", "color_flash")
        prefix    = spec.get("id_prefix", stim_type)
        stim_id   = f"{prefix}_{local_idx:02d}"
        duration  = float(spec.get("duration_ms", 250.0))

        stim = _build_stimulus(stim_id, duration, spec, cfg)
        if stim is None:
            continue

        trials.append(ScheduledTrial(
            trial_index=global_idx,
            start_ts=cursor,
            stimulus=stim,
        ))

        jitter = random.uniform(-isi_jitter_s, isi_jitter_s)
        cursor += duration / 1000.0 + max(0.5, isi_s + jitter)

    if trials:
        log.info(
            "Scheduled %d trials  first=+%.1f s  last=+%.1f s  total=+%.1f s",
            len(trials),
            trials[0].start_ts  - t0,
            trials[-1].start_ts - t0,
            cursor - t0,
        )
    return trials


def _build_stimulus(
    stim_id:    str,
    duration_ms: float,
    spec:       Dict[str, Any],
    cfg:        Dict[str, Any],
) -> Optional[Stimulus]:
    t = spec.get("type", "color_flash")
    try:
        if t == "color_flash":
            raw_region = spec.get("region")
            region = None if (raw_region is None or raw_region == "full") \
                     else tuple(int(v) for v in raw_region)
            return ColorFlash(
                stim_id, duration_ms, cfg,
                color=tuple(int(c) for c in spec.get("color", [0, 100, 255])),
                alpha=float(spec.get("alpha", 0.45)),
                region=region,
            )
        elif t == "shape":
            return ShapeStimulus(
                stim_id, duration_ms, cfg,
                shape=      spec.get("shape", "circle"),
                color=      tuple(int(c) for c in spec.get("color", [50, 50, 255])),
                size=       int(spec.get("size", 60)),
                position=   spec.get("position", "center"),
                border_only=bool(spec.get("border_only", False)),
            )
        elif t == "moving_target":
            return MovingTarget(
                stim_id, duration_ms, cfg,
                color=          tuple(int(c) for c in spec.get("color", [50, 220, 50])),
                size=           int(spec.get("size", 30)),
                speed_px_per_s= float(spec.get("speed_px_per_s", 120.0)),
                path=           spec.get("path", "horizontal"),
            )
        elif t == "reaction_prompt":
            return ReactionPrompt(
                stim_id, duration_ms, cfg,
                text=         spec.get("text", "PRESS SPACE"),
                color=        tuple(int(c) for c in spec.get("color", [0, 220, 255])),
                response_key= spec.get("response_key", "space"),
            )
        else:
            log.warning("Unknown stimulus type %r — skipping %r", t, stim_id)
            return None
    except Exception as exc:
        log.error("Failed to build stimulus %r: %s", stim_id, exc)
        return None
