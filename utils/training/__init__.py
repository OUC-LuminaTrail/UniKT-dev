"""训练模块

提供训练器基类、训练循环、指标计算、回调和检查点管理。
"""

from .base_trainer import BaseTrainer
from .callbacks import (
    Callback,
    CallbackManager,
    CheckpointCallback,
    EarlyStoppingCallback,
    FunctionCallback,
    MemoryCleanupCallback,
)
from .metrics import MetricsAccumulator
from .multi_trainer import MultiTrainer, StageConfig

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
    "FunctionCallback",
]
