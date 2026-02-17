"""统一注册表系统

提供类型安全的组件注册和管理，支持命名空间隔离。
合并了原有的多个注册表（全局、params、ablation）。
"""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class UniversalRegistry:
    """通用注册表，支持命名空间和类型约束。

    Args:
        name: 注册表名称
        namespace: 命名空间，用于隔离不同模块的注册表

    Example:
        >>> models = UniversalRegistry("models", namespace="kt")
        >>> @models.register("MyModel")
        ... class MyModel:
        ...     pass
        >>> models.get("MyModel")
        <class '__main__.MyModel'>
    """

    def __init__(self, name: str, namespace: str = "kt"):
        self._name = name
        self._namespace = namespace
        self._registry: dict[str, type[T]] = {}

    @property
    def full_name(self) -> str:
        """获取完整的注册表名称（包含命名空间）。"""
        return f"{self._namespace}.{self._name}"

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        """注册组件。

        Args:
            name: 用于注册组件的名称。如果为 None，则使用类/函数名称。

        Returns:
            装饰器函数

        Raises:
            KeyError: 如果名称已存在
        """

        def _register(cls: type[T]) -> type[T]:
            register_name = name if name is not None else cls.__name__
            if register_name in self._registry:
                raise KeyError(
                    f"'{register_name}' already registered in '{self.full_name}'"
                )
            self._registry[register_name] = cls
            return cls

        return _register

    def get(self, name: str) -> type[T]:
        """获取已注册的组件。

        Args:
            name: 组件名称

        Returns:
            已注册的组件类

        Raises:
            KeyError: 如果组件未找到
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in '{self.full_name}'. Available: {available}"
            )
        return self._registry[name]

    def keys(self) -> list:
        """获取所有已注册的名称。"""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """检查组件是否已注册。"""
        return name in self._registry

    def __repr__(self) -> str:
        return f"UniversalRegistry('{self.full_name}', items={self.keys()})"


# ============================================================================
# 全局注册表（使用命名空间隔离）
# ============================================================================

# 知识追踪模型注册表
MODELS = UniversalRegistry("models", namespace="kt")

# 训练器注册表
TRAINERS = UniversalRegistry("trainers", namespace="kt")

# 数据源注册表
DATA_SOURCES = UniversalRegistry("data_sources", namespace="kt")

# 通用组件注册表
COMPONENTS = UniversalRegistry("components", namespace="kt")

# 参数配置注册表
PARAM_CONFIGS = UniversalRegistry("param_configs", namespace="kt")

# 消融策略注册表
ABLATION_STRATEGIES = UniversalRegistry("strategies", namespace="ablation")

# 案例分析器注册表
ANALYZERS = UniversalRegistry("analyzers", namespace="kt")


# ============================================================================
# 向后兼容的便捷函数
# ============================================================================


def register_model(name: str | None = None):
    """注册模型的便捷函数。

    Args:
        name: 模型名称，如果为 None 则使用类名

    Example:
        >>> @register_model("MyModel")
        ... class MyModel:
        ...     pass
    """
    return MODELS.register(name)


def register_trainer(name: str | None = None):
    """注册训练器的便捷函数。

    Args:
        name: 训练器名称，如果为 None 则使用类名

    Example:
        >>> @register_trainer("MyTrainer")
        ... class MyTrainer:
        ...     pass
    """
    return TRAINERS.register(name)


def register_data_source(name: str | None = None):
    """注册数据源的便捷函数。

    Args:
        name: 数据源名称，如果为 None 则使用类名

    Example:
        >>> @register_data_source("MyDataset")
        ... class MyDataset:
        ...     pass
    """
    return DATA_SOURCES.register(name)


def register_analyzer(name: str | None = None):
    """注册案例分析器的便捷函数。

    Args:
        name: 分析器名称，如果为 None 则使用类名

    Example:
        >>> @register_analyzer("GIKT")
        ... class GIKTAnalyzer:
        ...     pass
    """
    return ANALYZERS.register(name)
