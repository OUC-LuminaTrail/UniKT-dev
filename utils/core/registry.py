"""统一注册表系统:装饰器注册 + 静态发现懒加载。

每个注册表维护两张表:

- ``_registry``: 名字 -> 类,由 ``@register_<role>(...)`` 装饰器在模块导入时填充。
- ``_index``: 名字 -> 模块路径,由 :mod:`utils.core.discovery` 在**不导入模块**的前提下
  扫描源码填充。
"""

from collections.abc import Callable


class UniversalRegistry:
    """通用注册表:支持装饰器注册与静态发现懒加载。

    Example:
        >>> TRAINERS = UniversalRegistry("trainers")
        >>> @register_trainer("GIKT")
        ... class GIKTTrainer: ...
        >>> TRAINERS.get("GIKT")
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: dict[str, type] = {}
        self._index: dict[str, str] = {}

    def register(self, name: str | None = None) -> Callable[[type], type]:
        """装饰器:把类绑定到 ``name``,在模块导入时触发。

        Args:
            name: 注册名。为 ``None`` 时取类名。

        Raises:
            KeyError: 该名字已绑定到**另一个**类。
        """

        def decorator(cls: type) -> type:
            n = name if name is not None else cls.__name__
            prev = self._registry.get(n)
            if prev is not None and prev is not cls:
                raise KeyError(f"'{n}' already registered in '{self._name}'")
            self._registry[n] = cls
            self._index.pop(n, None)  # 已实例化,懒索引不再需要
            return cls

        return decorator

    def index(self, name: str, module_path: str) -> None:
        """静态发现入口:记录 ``name`` 所在模块路径,**不导入模块**。

        Args:
            name: 注册名。
            module_path: 该名字所在模块的点分路径(如 ``model.GIKT.GIKT_trainer``)。

        Raises:
            KeyError: 同一名字被索引到两个不同模块。
        """
        prev = self._index.get(name)
        if prev is not None and prev != module_path:
            raise KeyError(
                f"'{name}' indexed twice in '{self._name}': {prev} vs {module_path}"
            )
        self._index.setdefault(name, module_path)

    def get(self, name: str) -> type:
        """按名取类;必要时按 ``_index`` 懒导入对应模块。

        Raises:
            KeyError: 名字未注册。
        """
        cls = self._registry.get(name)
        if cls is not None:
            return cls
        module_path = self._index.get(name)
        if module_path is not None:
            import importlib

            importlib.import_module(module_path)  # 触发装饰器 -> 填充 _registry
            cls = self._registry.get(name)
            if cls is not None:
                return cls
        raise KeyError(
            f"'{name}' not found in '{self._name}'. Available: {', '.join(self.keys())}"
        )

    def keys(self) -> list[str]:
        """全部已注册名字(已加载与懒索引取并集,去重)。"""
        seen = list(self._registry.keys())
        seen += [k for k in self._index if k not in self._registry]
        return seen

    def __contains__(self, name: object) -> bool:
        return name in self._registry or name in self._index

    def __repr__(self) -> str:
        return f"UniversalRegistry('{self._name}', items={self.keys()})"


# ============================================================================
# 全局注册表
# ============================================================================

TRAINERS = UniversalRegistry("trainers")
PARAM_CONFIGS = UniversalRegistry("param_configs")
DATA_SOURCES = UniversalRegistry("data_sources")
ANALYZERS = UniversalRegistry("analyzers")
METRIC_LOGGERS = UniversalRegistry("metric_loggers")


# ============================================================================
# 便捷装饰器:统一为 @register_<role>("name") 词汇
# ============================================================================


def register_trainer(name: str | None = None):
    """注册训练器到 ``TRAINERS``。"""
    return TRAINERS.register(name)


def register_model_params(name: str | None = None):
    """注册模型参数配置到 ``PARAM_CONFIGS``。"""
    return PARAM_CONFIGS.register(name)


def register_data_source(name: str | None = None):
    """注册数据源到 ``DATA_SOURCES``。"""
    return DATA_SOURCES.register(name)


def register_analyzer(name: str | None = None):
    """注册案例分析器到 ``ANALYZERS``。"""
    return ANALYZERS.register(name)


def register_metric_logger(name: str | None = None):
    """注册指标记录后端到 ``METRIC_LOGGERS``。"""
    return METRIC_LOGGERS.register(name)
