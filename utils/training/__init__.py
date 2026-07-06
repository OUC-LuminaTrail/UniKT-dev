"""Training module.

Provides trainer base classes, training loops, metric computation,
callbacks, and checkpoint management.
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
    "Callback",
    "CallbackManager",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "FunctionCallback",
    "LocalMetricLogger",
    "MemoryCleanupCallback",
    "MetricLogger",
    "MetricsAccumulator",
    "MultiTrainer",
    "StageComponents",
    "StageConfig",
    "StageResult",
    "SwanLabMetricLogger",
    "build_default_metric_loggers",
    "get_metric_logger",
]
