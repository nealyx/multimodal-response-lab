"""Personal baseline recording, survey labels, and offline comparison.

A recorded profile summarizes a person's signals under the supplied conditions;
it does not validate the proxy or automatically retune the live scorer. Survey
ratings are not computed from sensor features, but they are subjective,
retrospective, and may apply one session-level label to many overlapping windows.
Any predictive claim requires an appropriate independent evaluation.
"""

from src.calibration.baseline import (
    UserProfile,
    BlinkBaseline,
    HeadPoseBaseline,
    GazeBaseline,
    EngagementBaseline,
)
from src.calibration.collector import CalibrationSample, CalibrationCollector
from src.calibration.profiler import CalibrationProfiler
from src.calibration.survey import SurveyResponse, run_survey_cli
from src.calibration.labeler import SessionLabel, SessionLabeler
from src.calibration.comparison import SignalDelta, SessionComparison, BaselineComparator

__all__ = [
    "UserProfile", "BlinkBaseline", "HeadPoseBaseline",
    "GazeBaseline", "EngagementBaseline",
    "CalibrationSample", "CalibrationCollector",
    "CalibrationProfiler",
    "SurveyResponse", "run_survey_cli",
    "SessionLabel", "SessionLabeler",
    "SignalDelta", "SessionComparison", "BaselineComparator",
]
