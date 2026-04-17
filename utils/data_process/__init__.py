from typing import Any

from utils.core import DATA_SOURCES

from .assist09 import Assistments2009Data
from .assist12 import Assistments2012Data
from .assist17 import Assistments2017Data
from .data_source import DataSource
from .ednet_kt1 import EdNetKT1Data
from .slepemapy import SlepemapyData
from .junyi2015 import Junyi2015Data


def get_data_source(dataset_name: str, args: Any) -> DataSource:
    """Get the appropriate DataSource instance for a given dataset name.

    Args:
        dataset_name: Name of the dataset (e.g., 'assistments09', 'ednet_kt1')
        args: Configuration arguments for the data source

    Returns:
        DataSource instance configured with the provided args

    Raises:
        ValueError: If dataset_name is not registered in DATA_SOURCES
    """
    if dataset_name not in DATA_SOURCES:
        available = ", ".join(DATA_SOURCES.keys())
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {available}")

    dataset_cls = DATA_SOURCES.get(dataset_name)
    return dataset_cls(args=args)


__all__ = [
    "Assistments2009Data",
    "Assistments2012Data",
    "Assistments2017Data",
    "EdNetKT1Data",
    "SlepemapyData",
    "Junyi2015Data",
    "DataSource",
    "get_data_source",
]
