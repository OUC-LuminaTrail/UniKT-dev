"""Early-stopping algorithm: monitors a metric and signals when to stop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with utils.config
    from utils.config import EarlyStoppingConfig


class EarlyStopping:
    """General-purpose early stopping monitor.

    Tracks the best value of a monitored metric and recommends stopping after
    ``patience`` epochs without improvement (tolerance ``min_delta``).

    Usage:
        >>> es = EarlyStopping(EarlyStoppingConfig(patience=5, monitor="auc", mode="max"))
        >>> should_stop = es.step(current_val_metric)
    """

    def __init__(self, config: EarlyStoppingConfig):
        """Initialize the early stopping monitor with the given configuration."""
        self.cfg = config
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.num_bad_epochs = 0
        self.best_metrics: dict | None = None

        mode = self.cfg.mode.lower()
        if mode not in ("min", "max"):
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")
        self._cmp_sign = -1.0 if mode == "min" else 1.0

    def _is_improved(self, current: float, best: float) -> bool:
        # Unify comparison direction by multiplying with sign
        return (current - best) * self._cmp_sign > self.cfg.min_delta

    def step(
        self, current: float, epoch: int | None = None, metrics: dict | None = None
    ) -> bool:
        """Feed the current validation metric and return whether to stop early.

        Args:
            current: Current epoch's monitored metric value
            epoch: Current epoch number (optional)
            metrics: Full metrics dictionary for the current epoch (optional) {auc, acc, rmse}

        Returns:
            Whether training should stop
        """
        if self.best_score is None:
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            self.best_metrics = metrics.copy() if metrics else None
            return False

        if self._is_improved(current, self.best_score):
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            self.best_metrics = metrics.copy() if metrics else None
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.cfg.patience


__all__ = ["EarlyStopping"]
