"""数据处理包。

导入本包触发静态注册发现(扫描源码,不导入数据源代码),把所有 ``@register_data_source``
注册项写入 ``DATA_SOURCES`` 懒索引。数据源代码在 ``get_data_source(...)`` 时才按需导入。
"""

from pathlib import Path
from typing import Any

from utils.core import DATA_SOURCES, discover_registrations

from .data_source import DataSource

discover_registrations(Path(__file__).parent, "utils.data_process")


def get_data_source(dataset_name: str, args: Any) -> DataSource:
    """按名取数据源实例(按需懒导入对应模块)。

    Args:
        dataset_name: 数据集名(如 ``"assistments09"``、``"ednet_kt1"``)。
        args: 数据源配置参数。

    Returns:
        配置好的 ``DataSource`` 实例。

    Raises:
        ValueError: 数据集名未注册。
    """
    if dataset_name not in DATA_SOURCES:
        available = ", ".join(DATA_SOURCES.keys())
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {available}")
    dataset_cls = DATA_SOURCES.get(dataset_name)
    return dataset_cls(args=args)


__all__ = ["DataSource", "get_data_source"]
