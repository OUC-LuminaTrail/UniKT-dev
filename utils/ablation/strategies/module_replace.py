"""
Module replacement strategy.

Replaces a target module with a custom implementation.
"""

import torch.nn as nn

from utils.core import ABLATION_STRATEGIES, COMPONENTS

from .base import BaseAblationStrategy


@ABLATION_STRATEGIES.register("module_replace")
class ModuleReplaceStrategy(BaseAblationStrategy):
    """用自定义实现替换目标模块的策略。

    这对于测试复杂模块的简化版本或替代实现很有用。

    Parameters:
        replacement_module: 要替换的模块类或实例
        replacement_params: 传递给替换模块构造函数的参数

    Example:
        class SimpleInteraction(nn.Module):
            def forward(self, student_status, knowledge_status, user_mask):
                return (student_status * knowledge_status).sum(dim=-1)

        with ModuleReplaceStrategy(
            model,
            "general_interaction",
            replacement_module=SimpleInteraction
        ):
            output = model(input)
    """

    def __init__(
        self,
        model: nn.Module,
        target: str,
        params: dict | None = None,
    ):
        super().__init__(model, target, params)

        # Parse parameters
        replacement_module = self.params.get("replacement_module")
        replacement_params = self.params.get("replacement_params", {})

        if replacement_module is None:
            raise ValueError("replacement_module must be specified in params")

        # Resolve replacement module if it's a string
        if isinstance(replacement_module, str):
            if replacement_module in COMPONENTS:
                replacement_module = COMPONENTS.get(replacement_module)
            elif hasattr(nn, replacement_module):
                replacement_module = getattr(nn, replacement_module)
            else:
                raise ValueError(
                    f"Could not resolve replacement_module '{replacement_module}'. "
                    "It must be a class, an instance, or a name in COMPONENTS or torch.nn"
                )

        # Create replacement module
        if isinstance(replacement_module, type):
            # It's a class, instantiate it
            self.replacement = replacement_module(**replacement_params)
        else:
            # It's already an instance
            self.replacement = replacement_module

    def apply(self) -> None:
        """将目标模块替换为自定义实现。"""
        target_module = self._get_target_module()
        self._original_state = target_module
        setattr(self.model, self.target, self.replacement)

    def cleanup(self) -> None:
        """恢复原始模块。"""
        if self._original_state is not None:
            setattr(self.model, self.target, self._original_state)
            self._original_state = None


__all__ = ["ModuleReplaceStrategy"]
