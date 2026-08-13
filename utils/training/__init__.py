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
    TestEvaluationCallback,
)
from .early_stopping import EarlyStopping
from .metric_logger import (
    LocalMetricLogger,
    MetricLogger,
    SwanLabMetricLogger,
    WandbMetricLogger,
    build_default_metric_loggers,
    get_metric_logger,
)
from .metrics import MetricsAccumulator
from .multi_trainer import MultiTrainer, StageComponents, StageConfig
from .runtime_components import RuntimeComponents

__all__ = [
    "BaseTrainer",
    "Callback",
    "CallbackManager",
    "CheckpointCallback",
    "EarlyStopping",
    "EarlyStoppingCallback",
    "FunctionCallback",
    "LocalMetricLogger",
    "MemoryCleanupCallback",
    "MetricLogger",
    "MetricsAccumulator",
    "MultiTrainer",
    "RuntimeComponents",
    "StageComponents",
    "StageConfig",
    "StageResult",
    "SwanLabMetricLogger",
    "TestEvaluationCallback",
    "WandbMetricLogger",
    "build_default_metric_loggers",
    "get_metric_logger",
]
