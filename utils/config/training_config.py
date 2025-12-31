"""训练配置模块

提供训练相关配置，包括早停配置。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStoppingConfig:
    """早停配置。

    Attributes:
        monitor: 监控的指标，可选 'auc', 'acc', 'rmse', 'loss'
        mode: 优化模式，'max' 用于 auc/acc，'min' 用于 rmse/loss
        patience: 容忍的 epoch 数
        min_delta: 最小改善阈值
    """

    monitor: str = "auc"
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.0


class EarlyStopping:
    """通用早停工具。

    用法：
        >>> es = EarlyStopping(EarlyStoppingConfig(patience=5, monitor='auc', mode='max'))
        >>> should_stop = es.step(current_val_metric)

    特性：
    - 支持 min/max 模式
    - 支持 min_delta 容忍区间
    - 记录最佳指标值与对应 epoch
    """

    def __init__(
        self,
        config: Optional[EarlyStoppingConfig] = None,
        *,
        monitor: Optional[str] = None,
        mode: Optional[str] = None,
        patience: Optional[int] = None,
        min_delta: Optional[float] = None,
    ):
        if config is None:
            config = EarlyStoppingConfig()
        # 允许通过关键字参数覆盖
        if monitor is not None:
            config.monitor = monitor
        if mode is not None:
            config.mode = mode
        if patience is not None:
            config.patience = patience
        if min_delta is not None:
            config.min_delta = min_delta

        self.cfg = config
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.num_bad_epochs: int = 0

        mode = self.cfg.mode.lower()
        if mode not in ("min", "max"):
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")
        self._cmp_sign = -1.0 if mode == "min" else 1.0

    def _is_improved(self, current: float, best: float) -> bool:
        # 通过乘以 sign 统一比较方向
        return (current - best) * self._cmp_sign > self.cfg.min_delta

    def step(self, current: float, epoch: Optional[int] = None) -> bool:
        """输入本轮验证指标，返回是否需要早停。

        Args:
            current: 当前 epoch 的指标值
            epoch: 当前 epoch 编号（可选）

        Returns:
            是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return False

        if self._is_improved(current, self.best_score):
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.cfg.patience


__all__ = ["EarlyStopping", "EarlyStoppingConfig"]
