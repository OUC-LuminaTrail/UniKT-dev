"""核心基础设施模块

提供统一注册表、日志系统和随机种子设置。
"""

from .logger import get_logger, reset_loggers, set_log_level
from .random import seed_everything
from .registry import (
    ANALYZERS,
    COMPONENTS,
    DATA_SOURCES,
    METRIC_LOGGERS,
    MODELS,
    PARAM_CONFIGS,
    TRAINERS,
    UniversalRegistry,
    register_analyzer,
    register_data_source,
    register_metric_logger,
    register_model,
    register_trainer,
)

__all__ = [
    # Registry
    "MODELS",
    "TRAINERS",
    "DATA_SOURCES",
    "COMPONENTS",
    "PARAM_CONFIGS",
    "ANALYZERS",
    "METRIC_LOGGERS",
    "UniversalRegistry",
    "register_model",
    "register_trainer",
    "register_data_source",
    "register_analyzer",
    "register_metric_logger",
    # Logger
    "get_logger",
    "set_log_level",
    "reset_loggers",
    # Random
    "seed_everything",
]
