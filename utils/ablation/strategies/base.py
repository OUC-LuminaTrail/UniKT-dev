"""
Base strategy class for ablation studies.

Defines the interface and common functionality for all ablation strategies.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch.nn as nn

from utils.core import COMPONENTS


class BaseAblationStrategy(ABC):
    """消融策略基类。

    所有消融策略必须继承此类并实现 apply 和 cleanup 方法。
    """

    def __init__(self, model: nn.Module, target: str, params: dict | None = None):
        """初始化消融策略。

        Args:
            model: 要应用消融的PyTorch模型
            target: 目标模块名称（例如 "conv"、"history_review"）
            params: 策略的额外参数
        """
        self.model = model
        self.target = target
        self.params = params or {}
        self._original_state: Any | None = None
        self._handles: list = []

    @abstractmethod
    def apply(self) -> None:
        """将消融策略应用到模型上。

        此方法应该就地修改模型以达到所需的消融效果。
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """将模型恢复到原始状态。

        此方法应该撤销 apply() 所做的所有更改。
        """
        pass

    def _get_target_module(self) -> nn.Module:
        """从模型中获取目标模块。

        Returns:
            目标模块

        Raises:
            AttributeError: 如果找不到目标模块
        """
        if not hasattr(self.model, self.target):
            # Get only module names for cleaner error message
            model_modules = [name for name, _ in self.model.named_modules()]
            # Filter out empty string (root module) and sort
            model_modules = sorted([m for m in model_modules if m])

            raise AttributeError(
                f"Model does not have attribute '{self.target}'. Available modules: {model_modules}"
            )
        return getattr(self.model, self.target)

    def __enter__(self):
        """上下文管理器入口。"""
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口。"""
        self.cleanup()
        return False


@contextmanager
def apply_ablation(
    model: nn.Module, strategy_type: str, target: str, params: dict | None = None
) -> Generator[BaseAblationStrategy, None, None]:
    """应用消融策略的上下文管理器。

    Args:
        model: PyTorch模型
        strategy_type: 策略类型（例如 "module_disable"）
        target: 目标模块名称
        params: 策略参数

    Yields:
        策略实例

    Example:
        with apply_ablation(model, "module_disable", "conv"):
            # 禁用conv的模型
            output = model(input)
    """
    from utils.ablation import get_strategy

    strategy_class = get_strategy(strategy_type)
    strategy = strategy_class(model, target, params)

    try:
        strategy.apply()
        yield strategy
    finally:
        strategy.cleanup()


@COMPONENTS.register("IdentityModule")
class IdentityModule(nn.Module):
    """返回输入不变的恒等模块。

    用于通过替换模块来禁用模块。
    """

    def forward(self, *args, **kwargs):
        """返回第一个参数不变。"""
        if len(args) == 0:
            raise ValueError("IdentityModule requires at least one argument")
        return args[0]


@COMPONENTS.register("PassThroughModule")
class PassThroughModule(nn.Module):
    """返回指定索引参数的透传模块。

    Args:
        index: 要返回的参数索引（从0开始）

    Example:
        # 返回第一个参数
        pass_first = PassThroughModule(index=0)
        result = pass_first(view1, view2)  # = view1

        # 返回第二个参数
        pass_second = PassThroughModule(index=1)
        result = pass_second(view1, view2)  # = view2
    """

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index

    def forward(self, *args, **kwargs):
        """返回指定索引的参数。"""
        if len(args) == 0:
            raise ValueError("PassThroughModule requires at least one argument")
        if self.index >= len(args):
            raise IndexError(
                f"Index {self.index} out of range. Only {len(args)} arguments provided."
            )
        return args[self.index]


__all__ = [
    "BaseAblationStrategy",
    "apply_ablation",
    "IdentityModule",
    "PassThroughModule",
]
