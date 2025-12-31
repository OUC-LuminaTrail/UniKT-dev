"""
Parameter freeze strategy.

Freezes parameters of a target module so they don't participate in gradient updates.
"""

from typing import Dict, Optional

import torch.nn as nn
from utils.core import ABLATION_STRATEGIES
from .base import BaseAblationStrategy


@ABLATION_STRATEGIES.register("parameter_freeze")
class ParameterFreezeStrategy(BaseAblationStrategy):
    """冻结目标模块参数的策略。

    这对于测试可学习参数的贡献或模拟预训练的冻结组件很有用。

    Example:
        with ParameterFreezeStrategy(model, "lstm"):
            # LSTM 参数被冻结 (requires_grad=False)
            loss.backward()
            optimizer.step()  # LSTM 参数不会更新
    """

    def __init__(
        self,
        model: nn.Module,
        target: str,
        params: Optional[Dict] = None,
    ):
        super().__init__(model, target, params)

        # Parse parameters
        self.freeze_bias = self.params.get("freeze_bias", True)
        self.freeze_weight = self.params.get("freeze_weight", True)

    def apply(self) -> None:
        """将目标模块参数的 requires_grad 设置为 False。"""
        target_module = self._get_target_module()

        # 存储原始的 requires_grad 状态
        self._original_state = {}

        for name, param in target_module.named_parameters():
            self._original_state[name] = param.requires_grad

            # Apply freezing based on parameter type
            should_freeze = True
            if "bias" in name and not self.freeze_bias:
                should_freeze = False
            if "weight" in name and not self.freeze_weight:
                should_freeze = False

            if should_freeze:
                param.requires_grad = False

    def cleanup(self) -> None:
        """恢复原始的 requires_grad 状态。"""
        if self._original_state is None:
            return

        target_module = self._get_target_module()

        for name, param in target_module.named_parameters():
            if name in self._original_state:
                param.requires_grad = self._original_state[name]

        self._original_state = None


__all__ = ["ParameterFreezeStrategy"]
