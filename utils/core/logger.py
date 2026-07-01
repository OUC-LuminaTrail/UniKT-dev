"""KT 实验框架的统一日志模块。

此模块为所有框架组件提供一致的日志接口。
日志级别由 LOG_LEVEL 环境变量控制（默认：INFO）。

Usage:
    from utils.core import get_logger

    logger = get_logger(__name__)
    logger.info("Training started")
    logger.warning("Missing data, using defaults")
    logger.error("Model loading failed")

Environment Variables:
    LOG_LEVEL: 控制日志详细程度（DEBUG、INFO、WARNING、ERROR）。默认：INFO

Log Format:
    [HH:MM:SS][LEVEL][module_name] message
"""

import logging
import os

from rich.logging import RichHandler

# Global logger cache to avoid duplicate handlers
_loggers: dict = {}


def _get_log_level_from_env() -> int:
    """从环境变量 LOG_LEVEL 获取日志级别。

    Returns:
        int: 日志级别常量（logging.DEBUG、logging.INFO 等）
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
    """获取具有统一配置的日志记录器实例。

    日志记录器将配置为：
    - 来自 LOG_LEVEL 环境变量的级别（默认：INFO）
    - 仅控制台输出（SwanLab 处理训练指标的文件日志记录）
    - 统一格式：[HH:MM:SS][LEVEL][module_name] message

    Args:
        name: 日志记录器名称，通常对模块级日志记录使用 __name__

    Returns:
        logging.Logger: 配置的日志记录器实例

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
    """以编程方式设置全局日志级别。

    这将更新所有缓存的日志记录器及其处理程序。

    Args:
        level: 日志级别常量（例如 logging.DEBUG、logging.INFO）

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
    """重置所有缓存的日志记录器。

    这对于测试或需要重新初始化日志记录器时很有用。

    Example:
        >>> from utils.core import reset_loggers
        >>> reset_loggers()
    """
    global _loggers
    _loggers.clear()
    # Clear root logger handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()


__all__ = ["get_logger", "set_log_level", "reset_loggers"]
