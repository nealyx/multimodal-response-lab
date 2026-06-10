"""Day 12: user calibration, human label collection, and baseline comparison.

Why calibration matters
------------------------
Population-average thresholds (blink rate < 8/min = low, EAR = 0.20 = closed)
work tolerably on a large sample but fail for individuals who deviate from the
mean: heavy-glasses wearers have systematically lower EAR; touch-typists blink
half as often as average; people with wide-set eyes have different gaze ratios.

A 30-second personal baseline session corrects for these differences by
measuring YOUR neutral state — not an imagined average person's.

Why human labels reduce circularity
-------------------------------------
Day 11's supervised labels are heuristic: "high attention = engagement_mean >
0.65" derives from the same signals used as features.  A model trained on these
labels learns to reproduce a formula, not to predict anything independent.

Self-reported survey labels (after a session: "I was very focused, 4/5") are
independent of the sensor signals.  Training on them measures whether the
behavioral signals actually correlate with how engaged the person felt — which
is the real scientific question.

Why self-reported labels are still noisy
-----------------------------------------
Post-session retrospective ratings conflate multiple episodes of varied
engagement into a single number.  People are poor at recalling moment-to-moment
variation.  A single-number overall rating assigned to all windows from a session
ignores within-session dynamics.  For better labels you'd need either:
  - moment-to-moment thought-probe ratings (pressing a key when attention lapses)
  - an external objective measure aligned to the timeline (e.g. comprehension quiz)

This module is a step away from fully circular and toward research-grade, not a
fully rigorous experimental design.

How this connects to Egra-style evaluation
-------------------------------------------
An Egra product must demonstrate that its behavioral signals predict something
independently measurable — not just that its own signals are internally
consistent.  Day 12's human labels provide the first such independent anchor:
  - Calibrate → establish personal baseline
  - Collect session with stimulus experiments
  - Rate the session after (survey)
  - Train a model to predict survey rating from the 18-dim feature vectors
  - If the model outperforms a trivial baseline (majority class), the system has
    demonstrated that behavioural signals correlate with subjective experience
  - This is the minimum credible bar for Egra evaluation conversations
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
