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


def get_supported_datasets() -> list[str]:
    """Return all registered dataset names from ``DATA_SOURCES``.

    Lazily triggers discovery for ``utils.data_process`` if the registry
    is still empty (handles the case where ``utils.data_process`` was never
    imported, e.g. in the web backend).
    """
    if not DATA_SOURCES:
        from pathlib import Path

        discover_registrations(
            Path(__file__).resolve().parent.parent / "data_process",
            "utils.data_process",
        )
    return sorted(DATA_SOURCES.keys())


def get_supported_models() -> list[str]:
    """Return all registered model names from ``TRAINERS``.

    Lazily triggers discovery for ``model`` if the registry is still empty.
    """
    if not TRAINERS:
        from pathlib import Path

        discover_registrations(
            Path(__file__).resolve().parent.parent.parent / "model",
            "model",
        )
    return sorted(TRAINERS.keys())


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
    # Helpers
    "get_supported_datasets",
    "get_supported_models",
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
