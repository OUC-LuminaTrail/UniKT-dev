"""Callback integrating Optuna with the training loop."""

import math

from optuna.trial import Trial

from utils.training import Callback


class OptunaTrialCallback(Callback):
    """Report validation metrics to an Optuna trial and track the best value.

    Supports pruning: sets the ``pruned`` flag and lets the training loop
    end gracefully via ``should_stop`` (preserving resource cleanup in
    ``on_train_end`` / ``_finish``). The caller is expected to raise
    ``TrialPruned`` after ``run()`` returns.
    """

    def __init__(self, trial: Trial, metric_name: str, maximize: bool):
        """Initialise the Optuna trial callback.

        Args:
            trial: Optuna trial object.
            metric_name: Name of the metric to optimise (case-insensitive).
            maximize: Whether to maximise (True) or minimise (False) the metric.
        """
        self.trial = trial
        self.metric_name = metric_name.lower()
        self.maximize = maximize
        self.best_value: float | None = None
        self.pruned = False

    def _extract_value(self, metrics: dict, loss: float | None) -> float | None:
        """Extract the raw metric value from the validation metrics dictionary.

        Args:
            metrics: Dictionary of validation metrics.
            loss: Current loss value.

        Returns:
            The metric value as a float, or None if missing or non-finite.
        """
        value = loss if self.metric_name == "loss" else metrics.get(self.metric_name)
        if value is None or not math.isfinite(value):
            return None
        return float(value)

    def _is_better(self, current: float) -> bool:
        """Check whether the current value is better than the stored best.

        Args:
            current: Current metric value.

        Returns:
            True if the current value is better.
        """
        if self.best_value is None:
            return True
        return current > self.best_value if self.maximize else current < self.best_value

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Handle end-of-phase events: report validation metrics to Optuna.

        Reports the metric value to the trial at the current epoch and
        checks for pruning.

        Args:
            epoch: Current epoch number.
            phase: Phase name ('train' or 'val').
            loss: Current loss value.
            metrics: Dictionary of current metrics.
            **kwargs: Additional keyword arguments passed by the callback manager.
        """
        if phase != "val":
            return

        value = self._extract_value(metrics, loss)
        if value is None:
            return

        if self._is_better(value):
            self.best_value = value

        self.trial.report(value, epoch)
        if self.trial.should_prune():
            self.pruned = True

    def should_stop(self, **kwargs) -> bool:
        """Check whether training should stop due to pruning.

        Returns:
            True if the trial has been pruned.
        """
        return self.pruned


__all__ = ["OptunaTrialCallback"]
