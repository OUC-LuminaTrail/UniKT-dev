"""General hyperparameter management module.

Supports saving, loading, and validating hyperparameters for experiment
configuration and tracking.
"""

import json
import os
from argparse import Namespace
from datetime import datetime
from typing import Any

import torch
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.core import get_logger

logger = get_logger(__name__)


class HyperparameterManager:
    """Hyperparameter manager.

    Features:
    1. Save hyperparameters to JSON/YAML files
    2. Load hyperparameters from files
    3. Validate hyperparameter completeness
    4. Generate hyperparameter summaries
    5. Support version control and experiment tracking
    """

    def __init__(self, save_dir: str | None = None):
        """Initialise the hyperparameter manager.

        Args:
            save_dir: Save directory. If None, uses the default runs directory.
        """
        self.save_dir = save_dir
        self.hyperparams: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
        }
        # Dedicated console for summary rendering, sharing width/colour detection
        # with the RichHandler logger.
        self._console = Console()

    def get_hyperparameters_dict(self) -> dict[str, Any]:
        """Return the flattened hyperparameter dictionary."""
        result: dict[str, Any] = {}
        for group_params in self.hyperparams.values():
            if isinstance(group_params, dict):
                result.update(group_params)
        return result

    def add_hyperparams(self, params: dict | Namespace, group: str | None = None):
        """Add hyperparameters, automatically serialising all passed parameters.

        Args:
            params: Hyperparameter dictionary or argparse.Namespace.
            group: Parameter group name (e.g. 'model', 'training', 'data').
                   If None, parameters are added at the root level.
        """
        if isinstance(params, Namespace):
            params = vars(params)

        serialized_params = self._serialize_params(params)

        if group:
            if group not in self.hyperparams:
                self.hyperparams[group] = {}
            self.hyperparams[group].update(serialized_params)
        else:
            self.hyperparams.update(serialized_params)

    def add_metadata(self, key: str, value: Any):
        """Add metadata information.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = self._serialize_value(value)

    def _serialize_value(self, value: Any) -> Any:
        """Serialise a single value to a JSON-compatible representation.

        Args:
            value: Value to serialise.

        Returns:
            Serialised value.
        """
        if value is None:
            return None
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, torch.Tensor):
            return value.tolist()
        elif isinstance(value, torch.device):
            return str(value)
        elif isinstance(value, Namespace):
            return vars(value)
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, set):
            return list(value)
        elif hasattr(value, "__fspath__"):
            return str(value)
        elif callable(value):
            if hasattr(value, "__name__"):
                return f"<callable: {value.__name__}>"
            else:
                return f"<callable: {type(value).__name__}>"
        else:
            try:
                return str(value)
            except Exception:
                return f"<{type(value).__name__} object>"

    def _serialize_params(self, params: dict) -> dict:
        """Serialise a parameter dictionary.

        Args:
            params: Parameter dictionary.

        Returns:
            Serialised parameter dictionary.
        """
        return {k: self._serialize_value(v) for k, v in params.items()}

    def save(self, filename: str = "hyperparameters.json", format: str = "json"):
        """Save hyperparameters to a file.

        Args:
            filename: File name.
            format: Save format ('json' or 'yaml').
        """
        if self.save_dir is None:
            raise ValueError("Save directory not set. Please set save_dir first.")

        os.makedirs(self.save_dir, exist_ok=True)

        filepath = os.path.join(self.save_dir, filename)

        data = {"metadata": self.metadata, "hyperparameters": self.hyperparams}

        if format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        elif format == "yaml":
            try:
                import yaml

                with open(filepath, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            except ImportError:
                logger.warning("PyYAML not installed. Saving as JSON instead.")
                with open(
                    filepath.replace(".yaml", ".json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")

        logger.info(f"Hyperparameters saved to: {filepath}")

    def load(self, filepath: str) -> dict:
        """Load hyperparameters from a file.

        Args:
            filepath: File path.

        Returns:
            Loaded hyperparameter dictionary.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Hyperparameter file not found: {filepath}")

        _, ext = os.path.splitext(filepath)

        if ext == ".json":
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        elif ext in [".yaml", ".yml"]:
            try:
                import yaml

                with open(filepath, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML not installed. Cannot load YAML file.")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        self.metadata = data.get("metadata", {})
        self.hyperparams = data.get("hyperparameters", {})

        logger.info(f"Hyperparameters loaded from: {filepath}")
        return self.hyperparams

    def render_summary(self) -> Panel:
        """Render a hyperparameter summary as a rich Panel with groups and key/value pairs."""
        sections: list = []

        # Compact metadata row
        meta_pairs = [
            f"{k}={v}"
            for k, v in self.metadata.items()
            if k not in ("model_name", "dataset_name", "created_at")
        ]
        if meta_pairs:
            sections.append(Text("   ".join(meta_pairs), style="dim"))
            sections.append(Text(""))

        for group, params in self.hyperparams.items():
            if not params:
                continue
            sections.append(Text(f"\u25a0 {group}", style="bold magenta"))
            sections.append(self._param_grid(params))
            sections.append(Text(""))

        return Panel(
            Group(*sections),
            title=self._build_title(),
            title_align="left",
            border_style="blue",
            padding=(0, 1),
            expand=False,
        )

    def print_summary(self, console: Console | None = None) -> None:
        """Print a rich hyperparameter summary card."""
        (console or self._console).print(self.render_summary())

    def _param_grid(self, params: dict) -> Table:
        """Arrange group parameters into a two-column key/value grid, halving row count."""
        items = list(params.items())
        half = (len(items) + 1) // 2
        left, right = items[:half], items[half:]

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="cyan", no_wrap=True)
        grid.add_column()
        grid.add_column(style="cyan", no_wrap=True)
        grid.add_column()

        for i in range(half):
            lk, lv = left[i]
            if i < len(right):
                rk, rv = right[i]
                grid.add_row(str(lk), str(lv), str(rk), str(rv))
            else:
                grid.add_row(str(lk), str(lv), "", "")
        return grid

    def _build_title(self) -> str:
        """Build the card title, appending available model/dataset metadata."""
        model = self.metadata.get("model_name")
        dataset = self.metadata.get("dataset_name")
        if model and dataset:
            return f"Hyperparameters \u00b7 {model} @ {dataset}"
        if model:
            return f"Hyperparameters \u00b7 {model}"
        if dataset:
            return f"Hyperparameters \u00b7 @{dataset}"
        return "Hyperparameters"

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten a nested dictionary.

        Args:
            d: Dictionary to flatten.
            parent_key: Parent key name.
            sep: Separator.

        Returns:
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def to_namespace(self, group: str | None = None) -> Namespace:
        """Convert hyperparameters to a Namespace object.

        Args:
            group: If specified, only convert parameters from the given group.

        Returns:
            Namespace object.
        """
        params = (
            self.hyperparams.get(group, {})
            if group
            else self._flatten_dict(self.hyperparams)
        )

        return Namespace(**params)

    def validate_required(self, required_params: list) -> bool:
        """Validate that all required parameters exist.

        Args:
            required_params: List of required parameter names (dot notation
                             supported, e.g. 'model.hidden_dim').

        Returns:
            True if all required parameters exist, False otherwise.
        """
        flat_params = self._flatten_dict(self.hyperparams)
        missing = []

        for param in required_params:
            if param not in flat_params:
                missing.append(param)

        if missing:
            logger.warning(f"Missing required parameters: {missing}")
            return False

        return True


def create_hyperparameter_manager(
    args: dict | Namespace,
    save_dir: str,
    model_name: str | None = None,
    dataset_name: str | None = None,
    auto_group: bool = True,
) -> HyperparameterManager:
    """Create and configure a HyperparameterManager.

    Args:
        args: Hyperparameters (dictionary or Namespace).
        save_dir: Save directory.
        model_name: Model name.
        dataset_name: Dataset name.
        auto_group: Whether to automatically group parameters based on their
                    source (BaseParamConfig subclasses). Defaults to True.

    Returns:
        A configured HyperparameterManager instance.
    """
    from utils.config.param_config import get_param_sources

    manager = HyperparameterManager(save_dir=save_dir)

    if model_name:
        manager.add_metadata("model_name", model_name)
    if dataset_name:
        manager.add_metadata("dataset_name", dataset_name)

    args_dict = vars(args) if isinstance(args, Namespace) else args

    if auto_group:
        param_sources = get_param_sources()
        grouped: dict[str, dict] = {}

        for key, value in args_dict.items():
            group = param_sources.get(key, "General Parameters")
            grouped.setdefault(group, {})[key] = value

        for group, params in grouped.items():
            manager.add_hyperparams(params, group=group)
    else:
        manager.add_hyperparams(args_dict)

    return manager


__all__ = ["HyperparameterManager", "create_hyperparameter_manager"]
