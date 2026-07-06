"""Unified logging module for the KT experiment framework.

This module provides a consistent logging interface for all framework
components. The log level is controlled by the ``LOG_LEVEL`` environment
variable (default: INFO).

Usage:
    from utils.core import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
    logger.warning("Missing data, using defaults")
    logger.error("Model loading failed")

Environment Variables:
    LOG_LEVEL: Controls logging verbosity (DEBUG, INFO, WARNING, ERROR).
               Default: INFO

Log Format:
    [HH:MM:SS][LEVEL][module_name] message
"""

import logging
import os

from rich.logging import RichHandler

# Global logger cache to avoid duplicate handlers
_loggers: dict = {}


def _get_log_level_from_env() -> int:
    """Get the log level from the ``LOG_LEVEL`` environment variable.

    Returns:
        A logging level constant (``logging.DEBUG``, ``logging.INFO``, etc.).
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_mapping.get(log_level_str, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with a unified configuration.

    The returned logger is configured with:
    - A level from the ``LOG_LEVEL`` environment variable (default: INFO).
    - Console-only output (SwanLab handles file logging for training metrics).
    - A uniform format: ``[HH:MM:SS][LEVEL][module_name] message``.

    Args:
        name: Logger name, typically ``__name__`` for module-level logging.

    Returns:
        A configured ``logging.Logger`` instance.

    Example:
        >>> from utils.core import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
        [14:30:25][INFO][my_module] This is an info message
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]

    # Create new logger
    logger = logging.getLogger(name)
    logger.setLevel(_get_log_level_from_env())
    logger.propagate = False

    # Avoid adding handlers if logger already has them (e.g., from root logger)
    if not logger.handlers:
        # Create rich console handler
        console_handler = RichHandler()
        console_handler.setLevel(_get_log_level_from_env())
        # Add handler to logger
        logger.addHandler(console_handler)

    # Cache the logger
    _loggers[name] = logger
    return logger


def set_log_level(level: int) -> None:
    """Set the global log level programmatically.

    This updates all cached loggers and their handlers.

    Args:
        level: A logging level constant (e.g. ``logging.DEBUG``,
               ``logging.INFO``).

    Example:
        >>> from utils.core import get_logger, set_log_level
        >>> import logging
        >>> set_log_level(logging.DEBUG)
        >>> logger = get_logger(__name__)
        >>> logger.debug("This will now be visible")
    """
    os.environ["LOG_LEVEL"] = logging.getLevelName(level)
    for logger in _loggers.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def reset_loggers() -> None:
    """Reset all cached loggers.

    Useful during testing or whenever a full re-initialization of loggers
    is required.

    Example:
        >>> from utils.core import reset_loggers
        >>> reset_loggers()
    """
    global _loggers
    _loggers.clear()
    # Clear root logger handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()


__all__ = ["get_logger", "reset_loggers", "set_log_level"]
