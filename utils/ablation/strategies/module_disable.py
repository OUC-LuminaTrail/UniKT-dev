"""
Module disable strategy.

Replaces a target module with an identity module, effectively disabling it.
"""

from utils.core import ABLATION_STRATEGIES
from .base import BaseAblationStrategy, IdentityModule


@ABLATION_STRATEGIES.register("module_disable")
class ModuleDisableStrategy(BaseAblationStrategy):
    """通过将模块替换为恒等模块来禁用模块的策略。

    这对于测试特定组件对整体模型性能的贡献很有用。

    Example:
        with ModuleDisableStrategy(model, "conv"):
            # model.conv 现在是一个恒等模块
            output = model(input)
    """

    def apply(self) -> None:
        """将目标模块替换为恒等模块。"""
        target_module = self._get_target_module()
        self._original_state = target_module
        setattr(self.model, self.target, IdentityModule())

    def cleanup(self) -> None:
        """恢复原始模块。"""
        if self._original_state is not None:
            setattr(self.model, self.target, self._original_state)
            self._original_state = None


__all__ = ["ModuleDisableStrategy"]
