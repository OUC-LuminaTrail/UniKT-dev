"""训练配置模块

提供训练相关配置，包括早停配置。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EarlyStoppingConfig:
    """早停配置。

    Attributes:
        monitor: 监控的指标，可选 'auc', 'acc', 'rmse', 'loss'
        mode: 优化模式，'max' 用于 auc/acc，'min' 用于 rmse/loss
        patience: 容忍的 epoch 数
        min_delta: 最小改善阈值
    """

    monitor: str = "auc"
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.0


class EarlyStopping:
    """通用早停工具。

    用法：
        >>> es = EarlyStopping(EarlyStoppingConfig(patience=5, monitor='auc', mode='max'))
        >>> should_stop = es.step(current_val_metric)

    特性：
    - 支持 min/max 模式
    - 支持 min_delta 容忍区间
    - 记录最佳指标值与对应 epoch
    """

    def __init__(self, config: EarlyStoppingConfig):
        self.cfg = config
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.num_bad_epochs: int = 0

        mode = self.cfg.mode.lower()
        if mode not in ("min", "max"):
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")
        self._cmp_sign = -1.0 if mode == "min" else 1.0

    def _is_improved(self, current: float, best: float) -> bool:
        # 通过乘以 sign 统一比较方向
        return (current - best) * self._cmp_sign > self.cfg.min_delta

    def step(self, current: float, epoch: int | None = None) -> bool:
        """输入本轮验证指标，返回是否需要早停。

        Args:
            current: 当前 epoch 的指标值
            epoch: 当前 epoch 编号（可选）

        Returns:
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return False

        if self._is_improved(current, self.best_score):
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
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
    batch_size: int = 128
    collate_fn: Callable | None = None
    val_collate_fn: Callable | None = None


@dataclass
class OptimizationConfig:
    """Optimization configuration for optimizer, loss, and scheduler.

    Attributes:
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        lr_scheduler: Optional learning rate scheduler
        early_stopping: Optional early stopping configuration
    """

    optimizer: Any = None
    loss_fn: Any = None
    lr_scheduler: Any = None
    early_stopping: EarlyStoppingConfig | None = None


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration.

    Attributes:
        exp_manager: Experiment manager instance
        hyperparams: Hyperparameters dictionary or object
        use_swanlab: Whether to use SwanLab for logging
        model_name: Name of the model
        dataset_name: Name of the dataset
    """

    exp_manager: Any = None
    hyperparams: Any = None
    use_swanlab: bool = True
    model_name: str = ""
    dataset_name: str = ""


__all__ = [
    "EarlyStopping",
    "EarlyStoppingConfig",
    "TrainingConfig",
    "DataConfig",
    "OptimizationConfig",
    "ExperimentConfig",
]
