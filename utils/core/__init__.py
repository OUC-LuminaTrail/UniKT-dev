"""核心基础设施:统一注册表、日志、随机种子。"""

from .discovery import discover_registrations
from .logger import get_logger, reset_loggers, set_log_level
from .random import seed_everything
from .registry import (
    ANALYZERS,
    DATA_SOURCES,
    METRIC_LOGGERS,
    PARAM_CONFIGS,
    TRAINERS,
    UniversalRegistry,
    register_analyzer,
    register_data_source,
    register_metric_logger,
    register_model_params,
    register_trainer,
)

__all__ = [
    # Registries
    "TRAINERS",
    "PARAM_CONFIGS",
    "DATA_SOURCES",
    "ANALYZERS",
    "METRIC_LOGGERS",
    "UniversalRegistry",
    # Decorators
    "register_trainer",
    "register_model_params",
    "register_data_source",
    "register_analyzer",
    "register_metric_logger",
    # Discovery
    "discover_registrations",
    # Logger
    "get_logger",
    "set_log_level",
    "reset_loggers",
    # Random
    "seed_everything",
]
