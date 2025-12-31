"""配置基类模块

提供配置类的基定义和通用工具。
"""

import argparse
from dataclasses import dataclass


@dataclass
class BaseConfig:
    """配置基类。

    所有配置类的基类，提供从命令行参数创建配置的通用接口。
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """添加配置参数到参数解析器。

        子类可以重写此方法以添加模型特定的参数。

        Args:
            parser: ArgumentParser 实例
        """
        pass

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BaseConfig":
        """从命令行参数创建配置实例。

        Args:
            args: 命令行参数命名空间

        Returns:
            配置实例
        """
        # 获取 dataclass 的字段
        if hasattr(cls, "__dataclass_fields__"):
            field_names = cls.__dataclass_fields__.keys()
            kwargs = {k: v for k, v in vars(args).items() if k in field_names}
            return cls(**kwargs)
        else:
            return cls()


__all__ = ["BaseConfig"]
