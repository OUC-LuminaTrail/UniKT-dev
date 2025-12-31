"""
Module zeroing strategy.

Uses forward hooks to zero out the entire output of a module.
This is a safer alternative to ModuleDisableStrategy when the output shape
does not match the input shape.
"""

from typing import Any

import torch
import torch.nn as nn
from utils.core import ABLATION_STRATEGIES
from .base import BaseAblationStrategy


@ABLATION_STRATEGIES.register("module_zero")
class ModuleZeroStrategy(BaseAblationStrategy):
    """将整个模块输出置零的策略。

    这对于在保持预期输出形状的同时禁用模块的贡献很有用。

    Example:
        with ModuleZeroStrategy(model, "history_review"):
            # history_review 输出现在全为零
            output = model(input)
    """

    def apply(self) -> None:
        """注册前向钩子以置零输出。"""
        target_module = self._get_target_module()

        def zero_hook(module: nn.Module, input: Any, output: Any) -> Any:
            """Hook function to zero out output."""

            def process_output(obj: Any) -> Any:
                if isinstance(obj, torch.Tensor):
                    return torch.zeros_like(obj)
                elif isinstance(obj, dict):
                    return {k: process_output(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)([process_output(item) for item in obj])
                return obj

            return process_output(output)

        # 如果是 ModuleList 或 ModuleDict，为所有子模块注册钩子
        if isinstance(target_module, (nn.ModuleList, nn.ModuleDict)):
            for m in (
                target_module.values()
                if isinstance(target_module, nn.ModuleDict)
                else target_module
            ):
                handle = m.register_forward_hook(zero_hook)
                self._handles.append(handle)
        else:
            handle = target_module.register_forward_hook(zero_hook)
            self._handles.append(handle)

    def cleanup(self) -> None:
        """移除所有注册的钩子。"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


__all__ = ["ModuleZeroStrategy"]
