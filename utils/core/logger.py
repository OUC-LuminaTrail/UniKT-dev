"""Unified logging module for the KT experiment framework.

This module provides a consistent logging interface for all framework
components. The log level is controlled by the ``LOG_LEVEL`` environment
variable (default: INFO).

A shared file sink can be registered via :func:`add_file_handler`. Entry
scripts (``train.py``, ``evaluate.py``, ``case_analysis.py``, ...) call it
manually with the run directory, so every framework logger writes to the
run's log file alongside the console output. While the sink is registered,
``warnings.warn`` output (third-party libraries included) is captured into
the same file as WARNING records.

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
    Console: ``[HH:MM:SS][LEVEL][module_name] message`` (RichHandler)
    File:    ``%(asctime)s %(levelname)-8s %(name)s: %(message)s``
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler

# Global logger cache to avoid duplicate handlers
_loggers: dict = {}

# Shared file sink, appended to every framework logger when set. Forked
# DataLoader workers inherit it but stay silent on their hot path
# (Dataset.__getitem__ / collate do not log), so there is no interleaving.
# Revisit if worker-side logging is ever added.
_file_handler: logging.FileHandler | None = None


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


def _all_framework_loggers() -> list[logging.Logger]:
    """Every framework logger the file sink and level updates must reach.

    Module-level loggers captured at import time keep living in the global
    logging registry after :func:`reset_loggers` empties ``_loggers``, so we
    also recover them from there via their RichHandler signature.
    """
    seen: set[int] = set()
    targets: list[logging.Logger] = []
    for lg in _loggers.values():
        if id(lg) not in seen:
            seen.add(id(lg))
            targets.append(lg)
    for lg in logging.Logger.manager.loggerDict.values():
        # Skip PlaceHolder entries; keep only our loggers (RichHandler marker).
        if not isinstance(lg, logging.Logger) or id(lg) in seen:
            continue
        if any(isinstance(h, RichHandler) for h in lg.handlers):
            seen.add(id(lg))
            targets.append(lg)
    # The captureWarnings target carries no RichHandler; match by name.
    py_warnings = logging.Logger.manager.loggerDict.get("py.warnings")
    if isinstance(py_warnings, logging.Logger) and id(py_warnings) not in seen:
        seen.add(id(py_warnings))
        targets.append(py_warnings)
    return targets


def _detach_from_all(handler: logging.Handler) -> None:
    """Remove ``handler`` from every framework logger (idempotent)."""
    for lg in _all_framework_loggers():
        if handler in lg.handlers:
            lg.removeHandler(handler)


def _write_session_header(handler: logging.FileHandler) -> None:
    """Prefix this session with a separator header.

    evaluate/case_analysis reuse the run_dir, so appending without a banner
    would merge two runs' output; the timestamp + pid keeps them apart.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handler.stream.write(
        f"\n{'=' * 70}\n"
        f"# run.log session start  {timestamp}  pid={os.getpid()}\n"
        f"{'=' * 70}\n"
    )
    handler.flush()


def add_file_handler(log_path: str | Path) -> Path:
    """Attach the shared file sink, writing to ``log_path``.

    Any previously attached file handler is detached and closed first, so a
    second call (e.g. a new run directory in a kfold loop) switches the sink.
    The handler is appended to every framework logger; loggers created
    afterwards pick it up via :func:`get_logger`. :func:`warnings.warn`
    output is captured into the file as WARNING records as well.

    Args:
        log_path: Destination file path (parent dirs created as needed).

    Returns:
        The destination path.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    global _file_handler
    old_handler = _file_handler

    # Build + verify the new sink before touching the old one, so a failure
    # (bad path, permissions) leaves the previous handler intact.
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    try:
        handler.setLevel(_get_log_level_from_env())
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        _write_session_header(handler)
    except Exception:
        handler.close()
        raise

    if old_handler is not None:
        _detach_from_all(old_handler)
        old_handler.close()

    # Route warnings.warn() (third-party libraries included) into the sink.
    logging.captureWarnings(True)
    py_warnings = logging.getLogger("py.warnings")
    py_warnings.setLevel(_get_log_level_from_env())

    _file_handler = handler
    for lg in _all_framework_loggers():
        if handler not in lg.handlers:
            lg.addHandler(handler)
    return log_path


def remove_file_handler() -> None:
    """Detach and close the shared file handler, if any.

    Also disables warning capture.
    """
    global _file_handler
    if _file_handler is None:
        return
    _detach_from_all(_file_handler)
    _file_handler.close()
    _file_handler = None
    logging.captureWarnings(False)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with a unified configuration.

    Each logger carries a :class:`rich.logging.RichHandler` for console
    output (``propagate=False`` avoids duplicate messages) and, when a file
    sink is registered via :func:`add_file_handler`, that handler too.

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

    # Console output via rich (SwanLab handles training metrics separately).
    if not logger.handlers:
        console_handler = RichHandler()
        console_handler.setLevel(_get_log_level_from_env())
        logger.addHandler(console_handler)

    # Pick up the shared file sink when it has been registered already.
    if _file_handler is not None and _file_handler not in logger.handlers:
        logger.addHandler(_file_handler)

    # Cache the logger
    _loggers[name] = logger
    return logger


def set_log_level(level: int) -> None:
    """Set the global log level programmatically.

    Updates every framework logger and handler. The shared file sink is
    updated directly so a stale ``_loggers`` cache (or orphan loggers living
    only in the global registry) cannot keep an old level.

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
    if _file_handler is not None:
        _file_handler.setLevel(level)
    for logger in _all_framework_loggers():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


def reset_loggers() -> None:
    """Reset all cached loggers.

    Detaches the shared file handler before clearing the cache, so it is
    removed from every logger while the cache is still populated.

    Example:
        >>> from utils.core import reset_loggers
        >>> reset_loggers()
    """
    remove_file_handler()
    _loggers.clear()
    # Clear root logger handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()


__all__ = [
    "add_file_handler",
    "get_logger",
    "remove_file_handler",
    "reset_loggers",
    "set_log_level",
]
