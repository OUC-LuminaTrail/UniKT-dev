"""核心基础设施模块

提供统一注册表、日志系统和随机种子设置。
"""

from .registry import (
    MODELS,
    TRAINERS,
    DATA_SOURCES,
    COMPONENTS,
    PARAM_CONFIGS,
    ABLATION_STRATEGIES,
    UniversalRegistry,
    register_model,
    register_trainer,
    register_data_source,
)
from .logger import get_logger, set_log_level, reset_loggers
from .random import seed_everything

__all__ = [
    # Registry
    "MODELS",
    "TRAINERS",
    "DATA_SOURCES",
    "COMPONENTS",
    "PARAM_CONFIGS",
    "ABLATION_STRATEGIES",
    "UniversalRegistry",
    "register_model",
    "register_trainer",
    "register_data_source",
    # Logger
    "get_logger",
    "set_log_level",
    "reset_loggers",
    # Random
    "seed_everything",
]
