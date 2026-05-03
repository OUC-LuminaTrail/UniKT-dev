"""参数配置模块

提供模型参数配置基类和注册表。
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any

from ..core import PARAM_CONFIGS


class BaseParamConfig(ABC):
    """参数配置基类。

    子类必须实现 `define_params()` 方法，返回参数定义。

    Example:
        >>> class MyModelParams(BaseParamConfig):
        ...     def define_params(self) -> tuple[str, dict]:
        ...         return "MyModel Parameters", {
        ...             "hidden_dim": {"type": int, "default": 100, "help": "Hidden dimension"},
        ...         }
    """

    def __init__(self):
        super().__init__()
        # 从实例方法定义初始化
        group, params = self.define_params()
        self.group_name: str = group
        self.params: dict[str, dict[str, Any]] = params

    @abstractmethod
    def define_params(self) -> tuple[str, dict]:
        """定义参数。

        子类必须实现此方法，返回参数组名称和参数字典。

        Returns:
            (group_name, params_dict): 参数组名称和参数定义字典
        """
        raise NotImplementedError("Subclasses must implement define_params method.")

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """添加所有参数到 ArgumentParser。

        Args:
            parser: ArgumentParser 实例
        """
        inst = cls()
        group_name = inst.group_name or f"{cls.__name__} Parameters"
        params = inst.params
        group = parser.add_argument_group(group_name)

        for name, cfg in params.items():
            arg_names = [f"--{name}"]
            if "short" in cfg and cfg["short"]:
                arg_names.insert(0, f"-{cfg['short']}")

            kwargs = {k: v for k, v in cfg.items() if k not in {"short", "type"}}

            kwargs.setdefault("default", None)
            kwargs.setdefault("help", "")
            kwargs.setdefault("required", False)

            if cfg.get("type") is bool:
                kwargs.setdefault(
                    "action",
                    "store_false" if kwargs.get("default") is True else "store_true",
                )
            elif "type" in cfg:
                kwargs["type"] = cfg["type"]

            group.add_argument(*arg_names, **kwargs)


# ============================================================================
# 模型参数注册表（使用统一注册表）
# ============================================================================


def register_model_params(model_name: str):
    """注册模型参数配置。

    Usage:
        >>> @register_model_params("GIKT")
        ... class GIKTModelParams(BaseParamConfig):
        ...     def define_params(self):
        ...         return "GIKT Parameters", {...}

    Args:
        model_name: 模型名称

    Returns:
        装饰器函数
    """

    def decorator(cls: type[BaseParamConfig]) -> type[BaseParamConfig]:
        PARAM_CONFIGS.register(model_name)(cls)
        return cls

    return decorator


def get_model_params(model_name: str) -> type[BaseParamConfig] | None:
    """获取模型参数配置类。

    Args:
        model_name: 模型名称

    Returns:
        模型参数配置类，如果未找到则返回 None
    """
    try:
        return PARAM_CONFIGS.get(model_name)
    except KeyError:
        return None


def list_models() -> list[str]:
    """列出所有已注册的模型参数配置。

    Returns:
        模型名称列表
    """
    return PARAM_CONFIGS.keys()


# ============================================================================
# 预定义的参数配置类
# ============================================================================


class DataParams(BaseParamConfig):
    """数据处理参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "Data Parameters"
        params = {
            "dataset": {
                "type": str,
                "default": None,
                "short": "d",
                "required": True,
                "choices": [
                    "assistments09",
                    "assistments12",
                    "assistments15",
                    "assistments17",
                    "ednet_kt1",
                    "slepemapy",
                    "junyi2015",
                ],
                "help": "Dataset name to use (choices: assistments09, assistments12, assistments15, assistments17, ednet_kt1, slepemapy, junyi2015)",
            },
            "data_base_path": {
                "type": str,
                "default": "./data",
                "short": "dbp",
                "help": "Path to the data files (default: ./data)",
            },
            "fold": {
                "type": int,
                "default": 0,
                "help": "Index of fold for K-Fold cross-validation (default: 0)",
            },
            "kfold": {
                "type": int,
                "default": 5,
                "help": "Number of folds for K-Fold cross-validation (>=2 to enable, default: 5)",
            },
            "test_ratio": {
                "type": float,
                "default": 0.2,
                "help": "Ratio for test set (default: 0.2)",
            },
            "min_seq_len": {
                "type": int,
                "default": 3,
                "help": "Minimum sequence length (default: 3)",
            },
            "max_seq_len": {
                "type": int,
                "default": 200,
                "help": "Maximum sequence length (default: 200)",
            },
        }
        return group_name, params


class EarlyStoppingParams(BaseParamConfig):
    """早停参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "Early Stopping Parameters"
        params = {
            "es_monitor": {
                "type": str,
                "default": "auc",
                "short": "esm",
                "choices": ["auc", "acc", "rmse", "loss"],
                "help": "Metric to monitor for early stopping (choices: auc, acc, rmse, loss, default: auc)",
            },
            "es_mode": {
                "type": str,
                "default": "max",
                "short": "esmo",
                "help": "Optimization mode for monitored metric (default: max)",
            },
            "es_patience": {
                "type": int,
                "default": 10,
                "short": "esp",
                "help": "Number of epochs with no improvement before stopping (default: 10)",
            },
            "es_min_delta": {
                "type": float,
                "default": 0.0,
                "short": "esd",
                "help": "Minimum change to qualify as improvement (default: 0.0)",
            },
            "es_restore_best": {
                "type": bool,
                "default": False,
                "short": "esr",
                "help": "Restore best weights when early stopping triggers",
            },
        }
        return group_name, params


class GeneralParams(BaseParamConfig):
    """通用参数配置（日志、设备、种子等）。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "General Parameters"
        params = {
            "log_dir": {
                "type": str,
                "default": None,
                "short": "ld",
                "help": "Directory to save logs and models (default: runs/<timestamp>)",
            },
            "checkpoint_path": {
                "type": str,
                "default": None,
                "short": "cp",
                "help": "Path to model checkpoint for resuming training",
            },
            "device": {
                "type": str,
                "default": None,
                "short": "dev",
                "help": "Device to use: 'cuda' or 'cpu' (default: auto-detect)",
            },
            "seed": {
                "type": int,
                "default": 42,
                "short": "s",
                "help": "Random seed for reproducibility (default: 42)",
            },
            "deterministic": {
                "type": bool,
                "default": True,
                "short": "det",
                "help": "Disable deterministic algorithms (deterministic is enabled by default)",
            },
            "use_swanlab": {
                "type": bool,
                "default": True,
                "short": "usl",
                "help": "Enable SwanLab experiment tracking (default: True)",
            },
            "skip_test": {
                "type": bool,
                "default": False,
                "short": "st",
                "help": "Skip test set evaluation after training (default: False)",
            },
        }
        return group_name, params


class SamplingParams(BaseParamConfig):
    """数据抽样参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "Sampling Parameters"
        params = {
            "sample_size": {
                "type": int,
                "default": None,
                "help": "Absolute sample count. For random/stratified: number of users. For time: number of interactions (None to disable)",
            },
            "sample_ratio": {
                "type": float,
                "default": None,
                "help": "Sample ratio (0.0-1.0). Overrides sample_size if set. For random/stratified: ratio of users. For time: ratio of interactions",
            },
            "sample_strategy": {
                "type": str,
                "default": "random",
                "choices": ["random", "stratified", "time"],
                "help": "Sampling strategy (default: random, choices: random, stratified, time)",
            },
            "sample_attempts_bins": {
                "type": int,
                "default": [20, 100],
                "nargs": "+",
                "help": "Attempt count bin edges (e.g., 20 100 for low/medium/high)",
            },
            "sample_correct_bins": {
                "type": float,
                "default": [0.4, 0.8],
                "nargs": "+",
                "help": "Correct rate bin edges (e.g., 0.4 0.8 for low/medium/high)",
            },
        }
        return group_name, params


__all__ = [
    "BaseParamConfig",
    "DataParams",
    "EarlyStoppingParams",
    "GeneralParams",
    "SamplingParams",
    "register_model_params",
    "get_model_params",
    "list_models",
]
