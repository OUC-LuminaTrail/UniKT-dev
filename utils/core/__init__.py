"""Core infrastructure: unified registry, logger, and random seed utilities.

This module re-exports the primary components of the utils.core package,
including registries (trainers, model params, data sources, analyzers,
metric loggers), logging utilities, and seed setting.
"""

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
    "ANALYZERS",
    "DATA_SOURCES",
    "METRIC_LOGGERS",
    "PARAM_CONFIGS",
    # Registries
    "TRAINERS",
    "UniversalRegistry",
    # Discovery
    "discover_registrations",
    # Logger
    "get_logger",
    "register_analyzer",
    "register_data_source",
    "register_metric_logger",
    "register_model_params",
    # Decorators
    "register_trainer",
    "reset_loggers",
    # Random
    "seed_everything",
    "set_log_level",
]
