"""指标计算与聚合模块

处理训练/验证指标的累积、计算和记录。

各模型的 forward_pass 输出须包含以下字段：
    y_label   : 真实标签（0/1）
    y_predict : 二元预测（0/1），用于 ACC
    y_score   : 排序分数（任意实数均可），用于 AUC
    y_prob    : 预测概率（[0,1]），用于 RMSE
"""

import numpy as np
import torch
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
        self.use_swanlab = use_swanlab
        self._accumulators: dict[str, dict[str, list]] = {}

    def reset(self, phase: str):
        """重置指定 phase 的累积器。"""
        self._accumulators[phase] = {
            "y_label": [],
            "y_pred": [],
            "y_score": [],
            "y_prob": [],
            "group_id": [],
        }

    def update(self, phase: str, outputs: dict[str, torch.Tensor]):
        """更新累积器。

        Args:
            phase: "train" 或 "val"
            outputs: 须包含 "y_label", "y_predict", "y_score", "y_prob"
        """
        if phase not in self._accumulators:
            self.reset(phase)

        accum = self._accumulators[phase]
        accum["y_label"].append(outputs["y_label"].detach().cpu())
        accum["y_pred"].append(outputs["y_predict"].detach().cpu())
        accum["y_score"].append(outputs["y_score"].detach().cpu())
        accum["y_prob"].append(outputs["y_prob"].detach().cpu())
        group_id = outputs.get("group_id")
        if group_id is not None:
            accum["group_id"].append(group_id.detach().cpu())

    def compute(self, phase: str) -> dict[str, float]:
        """计算 epoch 级别指标。

        Returns:
            包含 "acc", "auc", "rmse" 的字典
        """
        if phase not in self._accumulators:
            return {}

        accum = self._accumulators[phase]
        if not accum["y_label"]:
            return {}

        # test 阶段且提供 group_id 时，执行全局 group-level late-mean 聚合
        if phase == "test" and accum["group_id"]:
            group_id: np.ndarray = torch.cat(accum["group_id"]).numpy()
            y_label_raw: np.ndarray = torch.cat(accum["y_label"]).numpy()
            y_score_raw: np.ndarray = torch.cat(accum["y_score"]).numpy()

            uniq_groups, inverse = np.unique(group_id, return_inverse=True)
            group_count = np.bincount(inverse).astype(np.float64)
            group_sum = np.bincount(inverse, weights=y_score_raw).astype(np.float64)
            group_score = group_sum / np.maximum(group_count, 1.0)

            # 标签取每个 group 首个值；同时校验一致性
            first_idx = np.full(uniq_groups.shape[0], -1, dtype=np.int64)
            for idx, gidx in enumerate(inverse):
                if first_idx[gidx] == -1:
                    first_idx[gidx] = idx
            group_label = y_label_raw[first_idx].astype(np.float64)

            if np.any(y_label_raw != group_label[inverse]):
                raise ValueError(
                    "Inconsistent labels within the same group_id in test evaluation."
                )

            group_pred = (group_score >= 0.5).astype(np.float64)
            metrics = {
                "acc": float(accuracy_score(group_label, group_pred)),
                "rmse": float(root_mean_squared_error(group_label, group_score)),
            }
            try:
                metrics["auc"] = float(roc_auc_score(group_label, group_score))
            except ValueError:
                metrics["auc"] = 0.0
            return metrics

        y_label: np.ndarray = torch.cat(accum["y_label"]).numpy()
        y_pred: np.ndarray = torch.cat(accum["y_pred"]).numpy()
        y_score: np.ndarray = torch.cat(accum["y_score"]).numpy()
        y_prob: np.ndarray = torch.cat(accum["y_prob"]).numpy()

        metrics = {
            "acc": float(accuracy_score(y_label, y_pred)),
        }

        try:
            metrics["auc"] = float(roc_auc_score(y_label, y_score))
        except ValueError:
            metrics["auc"] = 0.0

        metrics["rmse"] = float(root_mean_squared_error(y_label, y_prob))

        return metrics

    def log(self, phase: str, metrics: dict[str, float], epoch: int):
        """记录指标到 SwanLab。"""
        if not self.use_swanlab:
            return

        try:
            import swanlab

            if phase == "train":
                prefix = "Train/"
            elif phase == "val":
                prefix = "Val/"
            else:
                prefix = f"{phase.capitalize()}/"
            for name, value in metrics.items():
                swanlab.log({f"{prefix}{name.upper()}-epoch": value}, step=epoch)
        except ImportError:
            logger.warning("SwanLab not available, skipping metric logging")


__all__ = ["MetricsAccumulator"]
