from .assist09 import Assistments2009Data
from .assist12 import Assistments2012Data
from .assist17 import Assistments2017Data
from .ednet_kt1 import EdNetKT1Data
from .data_source import DataSource


def get_data_source(dataset_name: str, args):
    """根据数据集名称获取对应的数据源类实例"""
    if dataset_name == "assistments09":
        return Assistments2009Data(args=args)
    elif dataset_name == "assistments12":
        return Assistments2012Data(args=args)
    elif dataset_name == "assistments17":
        return Assistments2017Data(args=args)
    elif dataset_name == "ednet_kt1":
        return EdNetKT1Data(args=args)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


__all__ = [
    "Assistments2009Data",
    "Assistments2012Data",
    "Assistments2017Data",
    "EdNetKT1Data",
    "DataSource",
]
