"""Parameter configuration module.

Provides model parameter configuration base class and registry.
"""

import argparse
from abc import ABC, abstractmethod
from typing import Any

from ..core import DATA_SOURCES, PARAM_CONFIGS, register_model_params

_PARAM_SOURCES: dict[str, str] = {}


class BaseParamConfig(ABC):
    """Base class for parameter configuration.

    Subclasses must implement `define_params()` to return parameter definitions.

    Example:
        >>> class MyModelParams(BaseParamConfig):
        ...     def define_params(self) -> tuple[str, dict]:
        ...         return "MyModel Parameters", {
        ...             "hidden_dim": {"type": int, "default": 100, "help": "Hidden dimension"},
        ...         }
    """

    def __init__(self):
        """Initialize the parameter configuration from the instance method definition."""
        super().__init__()
        # Initialize from the instance method definition
        group, params = self.define_params()
        self.group_name: str = group
        self.params: dict[str, dict[str, Any]] = params

    @abstractmethod
    def define_params(self) -> tuple[str, dict]:
        """Define parameters.

        Subclasses must implement this method to return the parameter group name and parameter dictionary.

        Returns:
            (group_name, params_dict): Parameter group name and parameter definition dictionary
        """
        raise NotImplementedError("Subclasses must implement define_params method.")

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add all parameters to an ArgumentParser.

        Args:
            parser: ArgumentParser instance
        """
        inst = cls()
        group_name = inst.group_name or f"{cls.__name__} Parameters"
        params = inst.params
        group = parser.add_argument_group(group_name)

        for name, cfg in params.items():
            arg_names = [f"--{name}"]
            if cfg.get("short"):
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
            _PARAM_SOURCES[name] = group_name


def get_param_sources() -> dict[str, str]:
    """Get the mapping of parameter names to their source config group names.

    Returns:
        Dictionary mapping parameter names to source group names
    """
    return dict(_PARAM_SOURCES)


def get_model_params(model_name: str) -> type[BaseParamConfig] | None:
    """Get the model parameter configuration class.

    Args:
        model_name: Name of the model

    Returns:
        Model parameter configuration class, or None if not found
    """
    try:
        return PARAM_CONFIGS.get(model_name)
    except KeyError:
        return None


def list_models() -> list[str]:
    """List all registered model parameter configurations.

    Returns:
        List of model names
    """
    return PARAM_CONFIGS.keys()


# ============================================================================
# Predefined parameter configuration classes
# ============================================================================


class DataParams(BaseParamConfig):
    """Data processing parameter configuration."""

    def define_params(self) -> tuple[str, dict]:
        """Define data processing parameters.

        Returns:
            (group_name, params_dict): Parameter group name and parameter definition dictionary
        """
        group_name = "Data Parameters"
        params = {
            "dataset": {
                "type": str,
                "default": None,
                "short": "d",
                "required": True,
                "help": "Dataset name (required, choices: {})".format(
                    ", ".join(DATA_SOURCES.keys())
                ),
                "choices": list(DATA_SOURCES.keys()),
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
    """Early stopping parameter configuration."""

    def define_params(self) -> tuple[str, dict]:
        """Define early stopping parameters.

        Returns:
            (group_name, params_dict): Parameter group name and parameter definition dictionary
        """
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
    """General parameter configuration (logging, device, seed, etc.)."""

    def define_params(self) -> tuple[str, dict]:
        """Define general parameters.

        Returns:
            (group_name, params_dict): Parameter group name and parameter definition dictionary
        """
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
            "no_swanlab": {
                "type": bool,
                "default": False,
                "short": "nsl",
                "help": "Disable SwanLab experiment tracking (default: SwanLab on)",
            },
            "log_batch_metrics": {
                "type": bool,
                "default": False,
                "short": "lbm",
                "help": "Log per-batch loss to batch_metrics_<phase>.csv (default: False)",
            },
            "skip_test": {
                "type": bool,
                "default": False,
                "short": "st",
                "help": "Skip test set evaluation after training (default: False)",
            },
            "cache": {
                "type": bool,
                "default": False,
                "short": "ca",
                "help": "Enable disk cache for model data preparation (default: False)",
            },
        }
        return group_name, params


class CompileParams(BaseParamConfig):
    """torch.compile compilation parameter configuration."""

    def define_params(self) -> tuple[str, dict]:
        """Define compilation parameters.

        Returns:
            (group_name, params_dict): Parameter group name and parameter definition dictionary
        """
        group_name = "Compile Parameters"
        params = {
            "compile": {
                "type": bool,
                "default": False,
                "help": "Enable torch.compile for model optimization (default: False)",
            },
            "compile_mode": {
                "type": str,
                "default": "default",
                "choices": [
                    "default",
                    "reduce-overhead",
                    "max-autotune",
                    "max-autotune-no-cudagraphs",
                ],
                "help": "Compilation mode (choices: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs, default: default)",
            },
            "compile_fullgraph": {
                "type": bool,
                "default": False,
                "help": "Require the entire function be capturable into a single graph, raises error on graph breaks (default: False)",
            },
            "compile_dynamic": {
                "type": bool,
                "default": None,
                "help": "Use dynamic shape tracing to avoid recompilations when sizes change (default: None, PyTorch auto-detects dynamism)",
            },
            "compile_backend": {
                "type": str,
                "default": "inductor",
                "help": "Compilation backend (default: inductor)",
            },
        }
        return group_name, params


class SamplingParams(BaseParamConfig):
    """Data sampling parameter configuration."""

    def define_params(self) -> tuple[str, dict]:
        """Define sampling parameters.

        Returns:
            (group_name, params_dict): Parameter group name and parameter definition dictionary
        """
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
    "CompileParams",
    "DataParams",
    "EarlyStoppingParams",
    "GeneralParams",
    "SamplingParams",
    "get_model_params",
    "get_param_sources",
    "list_models",
    "register_model_params",
]
