"""Day 13: multi-session behavioral dataset management.

Why datasets matter more than individual sessions
--------------------------------------------------
A single session from one person tells you what THAT person did for 5-10
minutes under those specific conditions.  You cannot generalise:
  - Is the high blink rate a personal trait or a fatigue indicator?
  - Does the engagement score pattern replicate across days?
  - Do the supervised model's accuracy numbers hold for other people?

A multi-session, multi-participant dataset lets you answer these questions:
  - Within-person variation: same person across multiple days, tasks, or
    lighting conditions. Shows what is stable vs what fluctuates.
  - Between-person variation: different people under the same conditions.
    Shows what is universal vs what requires personalisation (Day 12).
  - Longitudinal trends: does a person's engagement score shift over a week?
    Is that a real cognitive change or a system drift?

Without a dataset, the system is a real-time tool. With one, it becomes a
research instrument.

Why metadata quality matters
-----------------------------
The most common failure mode in behavioural datasets is ambiguous provenance:
"we have 47 sessions but we don't know which participant did which session,
whether they were calibrated, or which software version was used."

Good metadata means every session entry answers:
  - Who recorded it? (participant ID, never PII)
  - When? (timestamp)
  - With what configuration? (experiment name, config snapshot)
  - Were files complete? (quality flags)
  - Were independent labels collected? (has_labels)
  - Was personalisation active? (has_calibration)

A model trained on a dataset with poor metadata produces results you cannot
interpret or reproduce.

Why participant tracking matters
---------------------------------
Anonymous participant IDs (participant_001, not "Alice") allow:
  - Grouping sessions from the same person for within-person analysis
  - Applying the correct personal calibration profile per session
  - Measuring inter-session reproducibility per participant
  - Excluding a participant's data if consent is withdrawn (delete one dir)

Participant tracking does NOT mean collecting PII. The system stores only
age_range (e.g., "25-34"), an optional notes field, and a calibration path.

How this enables future ML research
-------------------------------------
With N participants × M sessions each you can:
  - Leave-one-out cross-validation: train on N-1 people, test on 1 →
    measures generalisation to new individuals
  - Temporal hold-out: train on weeks 1-3, test on week 4 → measures
    whether behavioural patterns stay stable
  - Modality fusion: when EEG or biosignal data is added later, the same
    participant_id links the webcam and EEG session logs

The manifest.json is the schema contract for all of this. Future modalities
only need to add fields to SessionEntry, not restructure the whole system.
"""

from src.dataset.participant import Participant
from src.dataset.catalog import SessionEntry, SessionCatalog
from src.dataset.manifest import DatasetManifest
from src.dataset.registry import DatasetRegistry
from src.dataset.ingestor import SessionIngestor
from src.dataset.stats import DatasetStats, DatasetStatsComputer
from src.dataset.checker import QualityIssue, DatasetChecker
from src.dataset.exporter import DatasetExporter

__all__ = [
    "Participant",
    "SessionEntry", "SessionCatalog",
    "DatasetManifest",
    "DatasetRegistry",
    "SessionIngestor",
    "DatasetStats", "DatasetStatsComputer",
    "QualityIssue", "DatasetChecker",
    "DatasetExporter",
]
