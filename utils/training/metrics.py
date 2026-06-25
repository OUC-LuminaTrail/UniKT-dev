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


def _group_scores(y_score, inverse, num_groups, fusion_type, threshold):
    """按 fusion_type 计算每个 group 的聚合分数。

    mean: 组内均值；vote: 按组内多数对/错方向取相应子集均值，子集为空时回退整组；
    all: 全对/全错组取整组均值，其余组按多数方向取子集均值。
    """
    group_count = np.bincount(inverse, minlength=num_groups).astype(np.float64)
    if fusion_type == "mean":
        group_sum = np.bincount(inverse, weights=y_score, minlength=num_groups).astype(
            np.float64
        )
        return group_sum / np.maximum(group_count, 1.0)

    correct_sum = np.bincount(
        inverse, weights=(y_score >= threshold), minlength=num_groups
    ).astype(np.float64)
    majority = (correct_sum / np.maximum(group_count, 1.0)) >= 0.5

    if fusion_type == "vote":
        selected = np.where(
            majority[inverse], y_score >= threshold, y_score < threshold
        )
        selected_count = np.bincount(
            inverse, weights=selected, minlength=num_groups
        ).astype(np.float64)
        mask = selected | (selected_count == 0)[inverse]
    elif fusion_type == "all":
        uniform = (correct_sum == group_count) | (correct_sum == 0)
        base = np.where(majority[inverse], y_score >= threshold, y_score < threshold)
        mask = base | uniform[inverse]
    else:
        raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    weights = mask.astype(np.float64)
    numerator = np.bincount(
        inverse, weights=y_score * weights, minlength=num_groups
    ).astype(np.float64)
    denominator = np.bincount(inverse, weights=weights, minlength=num_groups).astype(
        np.float64
    )
    return numerator / np.maximum(denominator, 1.0)


class MetricsAccumulator:
    """指标累积器。

    职责：
    1. 收集 batch 级别的预测和标签
    2. 计算 epoch 级别的聚合指标

    指标的持久化记录由 MetricLogger 负责，本类只负责计算。

    Example:
        >>> accum = MetricsAccumulator()
        >>> accum.reset("train")
        >>> for batch in dataloader:
        ...     outputs = model(batch)
        ...     accum.update("train", outputs)
        >>> metrics = accum.compute("train")
    """

    def __init__(self):
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

        train/val 返回 acc/auc/rmse；test 且提供 group_id 时，对 mean/vote/all
        三种 group 聚合分别返回 {fusion}_acc/{fusion}_auc/{fusion}_rmse。
        """
        if phase not in self._accumulators:
            return {}

        accum = self._accumulators[phase]
        if not accum["y_label"]:
            return {}

        # test 阶段提供 group_id 时，对 mean/vote/all 三种 group 聚合分别计算指标
        if phase == "test" and accum["group_id"]:
            group_id: np.ndarray = torch.cat(accum["group_id"]).numpy()
            y_label_raw: np.ndarray = torch.cat(accum["y_label"]).numpy()
            y_score_raw: np.ndarray = torch.cat(accum["y_score"]).numpy()

            uniq_groups, inverse = np.unique(group_id, return_inverse=True)
            num_groups = uniq_groups.shape[0]

            # 标签取每个 group 首个值；同时校验一致性
            first_idx = np.full(num_groups, inverse.shape[0], dtype=np.int64)
            np.minimum.at(first_idx, inverse, np.arange(inverse.shape[0]))
            group_label = y_label_raw[first_idx].astype(np.float64)

            if np.any(y_label_raw != group_label[inverse]):
                raise ValueError(
                    "Inconsistent labels within the same group_id in test evaluation."
                )

            metrics = {}
            for fusion in ("mean", "vote", "all"):
                group_score = _group_scores(
                    y_score_raw, inverse, num_groups, fusion, 0.5
                )
                group_pred = (group_score >= 0.5).astype(np.float64)
                metrics[f"{fusion}_acc"] = float(
                    accuracy_score(group_label, group_pred)
                )
                metrics[f"{fusion}_rmse"] = float(
                    root_mean_squared_error(group_label, group_score)
                )
                try:
                    metrics[f"{fusion}_auc"] = float(
                        roc_auc_score(group_label, group_score)
                    )
                except ValueError:
                    metrics[f"{fusion}_auc"] = 0.0
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


__all__ = ["MetricsAccumulator"]
