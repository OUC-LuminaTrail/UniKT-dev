"""Optuna 与训练循环集成的回调。"""

import math

from optuna.trial import Trial

from utils.core import get_logger
from utils.training import Callback

logger = get_logger(__name__)


class OptunaTrialCallback(Callback):
    """将验证指标上报给 Optuna trial 并追踪其最佳值，支持剪枝。

    剪枝时仅置位 pruned 并通过 should_stop 正常结束训练循环（保留 on_train_end /
    _finish 的资源清理），由调用方在 run() 返回后抛出 TrialPruned 标记该 trial。
    """

    def __init__(self, trial: Trial, metric_name: str, maximize: bool):
        self.trial = trial
        self.metric_name = metric_name.lower()
        self.maximize = maximize
        self.best_value: float | None = None
        self.pruned = False

    def _extract_value(self, metrics: dict, loss: float | None) -> float | None:
        """从验证指标字典中提取优化指标的原始值，缺失或非有限值返回 None。"""
        value = loss if self.metric_name == "loss" else metrics.get(self.metric_name)
        if value is None or not math.isfinite(value):
            return None
        return float(value)

    def _is_better(self, current: float) -> bool:
        if self.best_value is None:
            return True
        return current > self.best_value if self.maximize else current < self.best_value

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        if phase != "val":
            return

        value = self._extract_value(metrics, loss)
        if value is None:
            return

        if self._is_better(value):
            self.best_value = value

        self.trial.report(value, epoch)
        if self.trial.should_prune():
            self.pruned = True

    def should_stop(self, **kwargs) -> bool:
        return self.pruned


__all__ = ["OptunaTrialCallback"]
