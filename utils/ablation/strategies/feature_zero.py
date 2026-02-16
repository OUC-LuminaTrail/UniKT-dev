"""
Feature zeroing strategy.

Uses forward hooks to zero out specific dimensions of module outputs.
"""

from typing import Any

import torch
import torch.nn as nn

from utils.core import ABLATION_STRATEGIES

from .base import BaseAblationStrategy


@ABLATION_STRATEGIES.register("feature_zero")
class FeatureZeroStrategy(BaseAblationStrategy):
    """将模块输出的特定维度置零的策略。

    这对于测试特定特征或嵌入维度的重要性很有用。

    Parameters:
        indices: 要置零的维度索引列表
        dim: 要置零的维度（默认为-1，即最后一个维度）

    Example:
        with FeatureZeroStrategy(model, "question_embedding", indices=[0, 1, 2]):
            # question_embedding 输出的前3个维度被置零
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
        self.indices = self.params.get("indices", [])
        self.dim = self.params.get("dim", -1)

        if not isinstance(self.indices, list):
            self.indices = list(self.indices)

    def apply(self) -> None:
        """注册前向钩子以置零特征。"""
        target_module = self._get_target_module()

        def zero_hook(module: nn.Module, input: Any, output: Any) -> Any:
            """用于置零指定维度的钩子函数。"""

            def process_output(obj: Any) -> Any:
                if isinstance(obj, torch.Tensor):
                    # Check indices bounds
                    dim_size = obj.size(self.dim)
                    valid_indices = [i for i in self.indices if 0 <= i < dim_size]
                    out_of_bounds = [i for i in self.indices if i < 0 or i >= dim_size]

                    if out_of_bounds:
                        raise ValueError(
                            f"Indices {out_of_bounds} out of bounds for dimension "
                            f"{self.dim} (size: {dim_size})"
                        )

                    # Zero out specified indices
                    obj.index_fill_(
                        self.dim, torch.tensor(valid_indices, device=obj.device), 0
                    )
                    return obj
                elif isinstance(obj, dict):
                    # Handle dictionary outputs
                    for key, value in obj.items():
                        obj[key] = process_output(value)
                    return obj
                elif isinstance(obj, (list, tuple)):
                    # Handle list or tuple outputs (e.g., from LSTM)
                    processed = [process_output(item) for item in obj]
                    return type(obj)(processed)
                return obj

            return process_output(output)

        handle = target_module.register_forward_hook(zero_hook)
        self._handles.append(handle)

    def cleanup(self) -> None:
        """移除所有注册的钩子。"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


__all__ = ["FeatureZeroStrategy"]
