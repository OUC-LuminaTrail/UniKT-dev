from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStoppingConfig:
    monitor: str = "auc"  # one of: 'auc', 'acc', 'rmse', 'loss'
    mode: str = "max"  # 'max' for auc/acc, 'min' for rmse/loss
    patience: int = 10
    min_delta: float = 0.0
    restore_best: bool = True


class EarlyStopping:
    """
    通用早停工具。

    用法：
        es = EarlyStopping(EarlyStoppingConfig(patience=5, monitor='auc', mode='max'))
        should_stop = es.step(current_val_metric)

    特性：
    - 支持 min/max 模式
    - 支持 min_delta 容忍区间
    - 记录最佳指标值与对应 epoch
    """

    def __init__(self, config: Optional[EarlyStoppingConfig] = None, *,
                 monitor: Optional[str] = None, mode: Optional[str] = None,
                 patience: Optional[int] = None, min_delta: Optional[float] = None,
                 restore_best: Optional[bool] = None):
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
        if restore_best is not None:
            config.restore_best = restore_best

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
        """
        输入本轮验证指标，返回是否需要早停。
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
