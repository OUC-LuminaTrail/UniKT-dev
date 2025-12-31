from .assist09 import Assistments2009Data
from .assist12 import Assistments2012Data
from .assist17 import Assistments2017Data
from .ednet_kt1 import EdNetKT1Data
from .data_source import DataSource
from typing import Any


def get_data_source(dataset_name: str, args: Any) -> DataSource:
    """根据数据集名称获取对应的数据源类实例"""
    from utils.core import DATA_SOURCES

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
    "DataSource",
    "get_data_source",
]
