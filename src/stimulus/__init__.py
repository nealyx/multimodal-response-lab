"""Day 7: stimulus-response experiment framework."""

from src.stimulus.analytics import SessionAnalytics
from src.stimulus.events import (
    BehavioralSample,
    ResponseEvent,
    SessionLog,
    StimulusEvent,
    TrialSummary,
)
from src.stimulus.exporter import SessionExporter
from src.stimulus.presenter import ScheduledTrial, StimulusPresenter, build_schedule
from src.stimulus.recorder import EventRecorder
from src.stimulus.stimuli import (
    ColorFlash,
    MovingTarget,
    ReactionPrompt,
    ShapeStimulus,
    Stimulus,
)

__all__ = [
    "BehavioralSample",
    "ColorFlash",
    "EventRecorder",
    "MovingTarget",
    "ReactionPrompt",
    "ResponseEvent",
    "ScheduledTrial",
    "SessionAnalytics",
    "SessionExporter",
    "SessionLog",
    "ShapeStimulus",
    "Stimulus",
    "StimulusEvent",
    "StimulusPresenter",
    "TrialSummary",
    "build_schedule",
]
