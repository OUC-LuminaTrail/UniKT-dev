"""指标计算与聚合模块

处理训练/验证指标的累积、计算和记录。
"""

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error

from ..core import get_logger

logger = get_logger(__name__)


class MetricsAccumulator:
    """指标累积器。

    职责：
    1. 收集 batch 级别的预测和标签
    2. 计算 epoch 级别的聚合指标
    3. 记录指标到实验追踪系统

    Example:
        >>> accum = MetricsAccumulator(use_swanlab=True)
        >>> accum.reset("train")
        >>> for batch in dataloader:
        ...     outputs = model(batch)
        ...     accum.update("train", outputs)
        >>> metrics = accum.compute("train")
        >>> accum.log("train", metrics, epoch=0)
    """

    def __init__(self, use_swanlab: bool = True):
        """初始化指标累积器。

        Args:
            use_swanlab: 是否使用 SwanLab 记录指标
        """
        self.use_swanlab = use_swanlab
        self._accumulators: dict[str, dict[str, list]] = {}

    def reset(self, phase: str):
        """重置指定 phase 的累积器。

        Args:
            phase: "train" 或 "val"
        """
        if phase not in self._accumulators:
            self._accumulators[phase] = {
                "y_hat": [],
                "y_label": [],
                "y_pred": [],
            }
        else:
            for key in self._accumulators[phase]:
                self._accumulators[phase][key] = []

    def update(self, phase: str, outputs: dict[str, torch.Tensor]):
        """更新累积器。

        Args:
            phase: "train" 或 "val"
            outputs: 包含 "y_hat", "y_label", "y_predict" 的字典
        """
        if phase not in self._accumulators:
            self.reset(phase)

        accum = self._accumulators[phase]
        accum["y_hat"].append(outputs["y_hat"].detach().cpu())
        accum["y_label"].append(outputs["y_label"].detach().cpu())
        accum["y_pred"].append(outputs["y_predict"].detach().cpu())

    def compute(self, phase: str) -> dict[str, float]:
        """计算 epoch 级别指标。

        Args:
            phase: "train" 或 "val"

        Returns:
            包含 "acc", "auc", "rmse" 的字典
        """
        if phase not in self._accumulators:
            return {}

        accum = self._accumulators[phase]
        if not accum["y_label"]:
            return {}

        # 拼接所有 batch 并转换为 numpy 数组
        y_label: np.ndarray = torch.cat(accum["y_label"]).numpy()
        y_pred: np.ndarray = torch.cat(accum["y_pred"]).numpy()
        y_hat: np.ndarray = torch.cat(accum["y_hat"]).numpy()

        # 检查预测值是否为概率值，如果不是则自动应用 sigmoid
        if not ((y_hat >= 0).all() and (y_hat <= 1).all()):
            y_hat = expit(y_hat)  # 数值稳定的 sigmoid 实现

        # 计算指标
        metrics = {
            "acc": float(accuracy_score(y_label, y_pred)),
        }

        # AUC 可能失败
        try:
            metrics["auc"] = float(roc_auc_score(y_label, y_hat))
        except ValueError:
            metrics["auc"] = 0.0

        metrics["rmse"] = float(root_mean_squared_error(y_label, y_hat))

        return metrics

    def log(self, phase: str, metrics: dict[str, float], epoch: int):
        """记录指标到 SwanLab。

        Args:
            phase: "train" 或 "val"
            metrics: 指标字典
            epoch: 当前 epoch
        """
        if not self.use_swanlab:
            return

        try:
            import swanlab

            prefix = "Train/" if phase == "train" else "Val/"

            for name, value in metrics.items():
                swanlab.log({f"{prefix}{name.upper()}-epoch": value}, step=epoch)
        except ImportError:
            logger.warning("SwanLab not available, skipping metric logging")


__all__ = ["MetricsAccumulator"]
