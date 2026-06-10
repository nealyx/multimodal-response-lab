"""Day 11: supervised predictive ML layer over behavioral embeddings."""

from src.supervised.labels import LabelConfig, assign_labels
from src.supervised.dataset import load_windows_csv, build_dataset
from src.supervised.trainer import BehaviorModelTrainer, TrainResult
from src.supervised.evaluator import evaluate, EvalResult
from src.supervised.exporter import ModelExporter

__all__ = [
    "LabelConfig",
    "assign_labels",
    "load_windows_csv",
    "build_dataset",
    "BehaviorModelTrainer",
    "TrainResult",
    "evaluate",
    "EvalResult",
    "ModelExporter",
]
