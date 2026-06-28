"""训练模块

提供训练器基类、训练循环、指标计算、回调和检查点管理。
"""

from .base_trainer import BaseTrainer, StageResult
from .callbacks import (
    Callback,
    CallbackManager,
    CheckpointCallback,
    EarlyStoppingCallback,
    FunctionCallback,
    MemoryCleanupCallback,
)
from .metric_logger import (
    LocalMetricLogger,
    MetricLogger,
    SwanLabMetricLogger,
    build_default_metric_loggers,
    get_metric_logger,
)
from .metrics import MetricsAccumulator
from .multi_trainer import MultiTrainer, StageComponents, StageConfig

__all__ = [
    "BaseTrainer",
    "MultiTrainer",
    "StageConfig",
    "StageComponents",
    "StageResult",
    "MetricsAccumulator",
    "Callback",
    "CallbackManager",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "MemoryCleanupCallback",
    "FunctionCallback",
    "MetricLogger",
    "LocalMetricLogger",
    "SwanLabMetricLogger",
    "get_metric_logger",
    "build_default_metric_loggers",
]
