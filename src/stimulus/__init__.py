"""Stimulus events and analytics, with rendering imported only when requested."""

from importlib import import_module

from src.stimulus.analytics import SessionAnalytics
from src.stimulus.events import (
    BehavioralSample,
    ResponseEvent,
    SessionLog,
    StimulusEvent,
    TrialSummary,
)
from src.stimulus.exporter import SessionExporter
from src.stimulus.recorder import EventRecorder

_RENDERING_EXPORTS = {
    "ScheduledTrial": "presenter",
    "StimulusPresenter": "presenter",
    "build_schedule": "presenter",
    "ColorFlash": "stimuli",
    "MovingTarget": "stimuli",
    "ReactionPrompt": "stimuli",
    "ShapeStimulus": "stimuli",
    "Stimulus": "stimuli",
}


def __getattr__(name):
    """Keep the public rendering API without loading OpenCV for file analysis."""
    if name not in _RENDERING_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_RENDERING_EXPORTS[name]}"), name)
    globals()[name] = value
    return value

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
