"""训练模块

提供训练器基类、训练循环、指标计算、回调和检查点管理。
"""

from .base_trainer import BaseTrainer
from .multi_trainer import MultiTrainer, StageConfig
from .metrics import MetricsAccumulator
from .callbacks import (
    Callback,
    CallbackManager,
    EarlyStoppingCallback,
    CheckpointCallback,
    MemoryCleanupCallback,
)

__all__ = [
    "BaseTrainer",
    "MultiTrainer",
    "StageConfig",
    "MetricsAccumulator",
    "Callback",
    "CallbackManager",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "MemoryCleanupCallback",
]
