"""Data processing package.

Importing this package triggers static registration discovery (scans source files
without importing them), writing all ``@register_data_source`` entries into the
``DATA_SOURCES`` lazy index. Data source code is imported on demand only when
``get_data_source(...)`` is called.
"""

from pathlib import Path
from typing import Any

from utils.core import DATA_SOURCES, discover_registrations

from .data_source import DataSource

discover_registrations(Path(__file__).parent, "utils.data_process")


def get_data_source(dataset_name: str, args: Any) -> DataSource:
    """Get a data source instance by name with on-demand lazy import.

    Args:
        dataset_name: Dataset name (e.g. ``"assistments09"``, ``"ednet_kt1"``).
        args: Data source configuration parameters.

    Returns:
        A configured ``DataSource`` instance.

    Raises:
        ValueError: If the dataset name is not registered.
    """
    if dataset_name not in DATA_SOURCES:
        available = ", ".join(DATA_SOURCES.keys())
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {available}")
    dataset_cls = DATA_SOURCES.get(dataset_name)
    return dataset_cls(args=args)


__all__ = ["DataSource", "get_data_source"]
