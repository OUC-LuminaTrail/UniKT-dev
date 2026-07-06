"""Training configuration module.

Provides training-related configurations, including early stopping configuration.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration.

    Attributes:
        monitor: Metric to monitor, one of 'auc', 'acc', 'rmse', 'loss'
        mode: Optimization mode, 'max' for auc/acc, 'min' for rmse/loss
        patience: Number of epochs to tolerate without improvement
        min_delta: Minimum improvement threshold
    """

    monitor: str = "auc"
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.0


class EarlyStopping:
    """General-purpose early stopping utility.

    Usage:
        >>> es = EarlyStopping(EarlyStoppingConfig(patience=5, monitor='auc', mode='max'))
        >>> should_stop = es.step(current_val_metric)

    Features:
    - Supports min/max mode
    - Supports min_delta tolerance
    - Records best metric value and corresponding epoch
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


@dataclass
class TrainingConfig:
    """Training configuration for epochs and device.

    Attributes:
        epochs: Number of training epochs
        seed: Random seed for reproducibility
        device: Device to train on (cuda/cpu)
        checkpoint_path: Path to save/load checkpoints
    """

    epochs: int = 150
    seed: int = 42
    device: torch.device | None = None
    checkpoint_path: str | None = None


@dataclass
class DataConfig:
    """Data configuration for training and validation datasets.

    Attributes:
        train_data: Training DataLoader
        val_data: Validation DataLoader
        batch_size: Batch size for DataLoader
        collate_fn: Optional custom collate function for training
        val_collate_fn: Optional custom collate function for validation
    """

    train_data: Any = None
    val_data: Any = None
    test_data: Any = None
    batch_size: int = 128
    collate_fn: Callable | None = None
    val_collate_fn: Callable | None = None
    test_collate_fn: Callable | None = None


@dataclass
class OptimizationConfig:
    """Optimization configuration for optimizer, loss, and scheduler.

    Attributes:
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        max_clip_grad_norm: Optional max norm for gradient clipping
        lr_scheduler: Optional learning rate scheduler
        early_stopping: Optional early stopping configuration
    """

    optimizer: Any = None
    loss_fn: Any = None
    max_clip_grad_norm: float | None = None
    lr_scheduler: Any = None
    early_stopping: EarlyStoppingConfig | None = None


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration.

    Attributes:
        exp_manager: Experiment manager instance
        hyperparams: Hyperparameters dictionary or object
        no_swanlab: Disable SwanLab tracking (local CSV logging is always on)
        log_batch_metrics: Record per-batch loss to batch_metrics_<phase>.csv
        model_name: Name of the model
        dataset_name: Name of the dataset
    """

    exp_manager: Any = None
    hyperparams: Any = None
    no_swanlab: bool = False
    log_batch_metrics: bool = False
    model_name: str = ""
    dataset_name: str = ""


__all__ = [
    "DataConfig",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ExperimentConfig",
    "OptimizationConfig",
    "TrainingConfig",
]
