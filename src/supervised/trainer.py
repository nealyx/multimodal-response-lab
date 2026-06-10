"""Model training pipeline for behavioral prediction.

Why this is a baseline, not a production model
-----------------------------------------------
Three constraints make this a prototype:

1. Heuristic labels (see labels.py): the targets are derived from the same
   signals used as features, so the model learns to reproduce a rule, not to
   predict an independent measure of cognitive state.

2. Small-n problem: a single 5–20 minute session produces 60–300 windows.
   Reliable generalization estimates require held-out sessions, not a 20-sample
   test split.  See the WARNING printed when n_test < 30.

3. No cross-session or cross-person validation: individual differences in
   baseline EAR, gaze range, and blink rate mean that a model trained on person
   A will not generalize to person B without recalibration or fine-tuning.

What the baseline is good for:
- Validating that the 18-dim feature space contains discriminative signal
- Identifying which features drive the most variance in behavioral state
- Providing a reproducible benchmark that future models must beat

How feature importance guides engineering:
  If 'focused_frac' and 'engagement_mean' dominate importance, the engagement
  signal is the bottleneck — invest in reducing UNRELIABLE frames and
  recalibrating EAR.  If gaze features dominate, the gaze calibration (Day 4)
  is the place to improve.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.supervised.dataset import BehavioralDataset, train_test_split_dataset

# Minimum test samples before we emit a data-quality warning
_MIN_TEST_SAMPLES = 20


@dataclass
class TrainResult:
    """Everything produced by one training run.

    Attributes
    ----------
    model : fitted sklearn estimator
    model_name : str
    train_dataset, test_dataset : BehavioralDataset splits
    train_score : float  — accuracy on train set
    test_score  : float  — accuracy on test set
    config      : dict   — all hyperparameters used
    warnings    : list   — data-quality / reliability warnings
    """
    model:          Any
    model_name:     str
    train_dataset:  BehavioralDataset
    test_dataset:   BehavioralDataset
    train_score:    float
    test_score:     float
    config:         Dict[str, Any]    = field(default_factory=dict)
    warnings_list:  List[str]         = field(default_factory=list)


def _make_logistic_regression(random_state: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )


def _make_random_forest(random_state: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
    )


def _make_gradient_boosting(random_state: int) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
    )


MODEL_FACTORIES = {
    "logistic_regression":  _make_logistic_regression,
    "random_forest":        _make_random_forest,
    "gradient_boosting":    _make_gradient_boosting,
}


class BehaviorModelTrainer:
    """Train one or more classifiers on a BehavioralDataset.

    Parameters
    ----------
    model_names : list[str]
        Which models to train. Defaults to all three.
    test_size : float
        Fraction of windows held out for testing (default 0.20).
    split_strategy : str
        ``"random"`` (stratified) or ``"temporal"`` (time-ordered split).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        model_names:     Optional[List[str]] = None,
        test_size:       float               = 0.20,
        split_strategy:  str                 = "random",
        random_state:    int                 = 42,
    ) -> None:
        self.model_names    = model_names or list(MODEL_FACTORIES.keys())
        self.test_size      = test_size
        self.split_strategy = split_strategy
        self.random_state   = random_state

    def train_all(self, dataset: BehavioralDataset) -> List[TrainResult]:
        """Train all configured models on *dataset* and return results."""
        train_ds, test_ds = train_test_split_dataset(
            dataset,
            test_size=self.test_size,
            strategy=self.split_strategy,
            random_state=self.random_state,
        )
        results = []
        for name in self.model_names:
            result = self._train_one(name, train_ds, test_ds)
            results.append(result)
        return results

    def _train_one(
        self,
        model_name: str,
        train_ds:   BehavioralDataset,
        test_ds:    BehavioralDataset,
    ) -> TrainResult:
        warns: List[str] = []

        n_train = train_ds.n_samples
        n_test  = test_ds.n_samples

        if n_train < 10:
            warns.append(
                f"Only {n_train} training samples — model will overfit badly. "
                "Record a longer session or merge multiple sessions."
            )
        if n_test < _MIN_TEST_SAMPLES:
            warns.append(
                f"Only {n_test} test samples — accuracy estimate has very wide "
                "confidence intervals (~±10 pp for 20 samples). "
                "Reported metrics are illustrative only."
            )

        # Check class balance
        unique, counts = np.unique(train_ds.y, return_counts=True)
        if len(unique) < 2:
            warns.append(
                "Training data has only one class — classifier cannot learn. "
                "Try a different label task or record a more varied session."
            )

        factory = MODEL_FACTORIES.get(model_name)
        if factory is None:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODEL_FACTORIES.keys())}"
            )

        model = factory(self.random_state)

        # Suppress convergence warnings for small datasets
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n_train > 0 and len(unique) >= 2:
                model.fit(train_ds.X, train_ds.y)
                train_score = float(model.score(train_ds.X, train_ds.y))
                test_score  = float(model.score(test_ds.X, test_ds.y)) if n_test > 0 else float("nan")
            else:
                train_score = float("nan")
                test_score  = float("nan")

        cfg = {
            "model_name":     model_name,
            "test_size":      self.test_size,
            "split_strategy": self.split_strategy,
            "random_state":   self.random_state,
            "n_train":        n_train,
            "n_test":         n_test,
            "task":           train_ds.task,
            "label_names":    train_ds.label_names,
            "class_counts_train": {
                train_ds.label_names[i]: int(c)
                for i, c in zip(unique.tolist(), counts.tolist())
            },
        }

        return TrainResult(
            model=model,
            model_name=model_name,
            train_dataset=train_ds,
            test_dataset=test_ds,
            train_score=train_score,
            test_score=test_score,
            config=cfg,
            warnings_list=warns,
        )
