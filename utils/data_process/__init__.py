"""Data processing package.

Importing this package triggers static registration discovery (scans source files
without importing them), writing all ``@register_data_source`` entries into the
``DATA_SOURCES`` lazy index. Data source code is imported on demand only when
``get_data_source(...)`` is called.
"""

from pathlib import Path

from utils.core import DATA_SOURCES, discover_registrations, get_supported_datasets

from .data_source import DataSource

discover_registrations(Path(__file__).parent, "utils.data_process")


def _rc_to_args_namespace(rc):
    """Flatten a RunConfig's data + general nodes into the flat namespace DataSource consumes.

    Yields ``args.dataset`` / ``args.seed`` / ``args.min_seq_len`` / ... for
    DataSource internals.
    """
    import argparse

    from omegaconf import OmegaConf

    flat = OmegaConf.to_container(rc.data, resolve=True)
    flat["seed"] = rc.general.seed
    flat["device"] = rc.general.device
    return argparse.Namespace(**flat)


def get_data_source(rc) -> DataSource:
    """Get a data source instance from a RunConfig, with on-demand lazy import.

    Args:
        rc: RunConfig as an OmegaConf ``DictConfig``; the dataset is read from
            ``rc.data.dataset``.

    Returns:
        A configured ``DataSource`` instance.

    Raises:
        ValueError: If the dataset name is not registered.
    """
    dataset_name = rc.data.dataset
    if dataset_name not in DATA_SOURCES:
        available = ", ".join(get_supported_datasets())
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {available}")
    dataset_cls = DATA_SOURCES.get(dataset_name)
    return dataset_cls(args=_rc_to_args_namespace(rc))


__all__ = ["DataSource", "get_data_source"]
