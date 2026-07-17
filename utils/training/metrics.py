"""Metric computation and accumulation module.

Handles aggregation, computation, and logging of training/validation metrics.

Each model's forward_pass output must contain the following fields:
    y_label   : ground-truth labels (0/1)
    y_predict : binary predictions (0/1), used for ACC
    y_score   : ranking scores (any real number), used for AUC
    y_prob    : predicted probabilities ([0,1]), used for RMSE
    y_score   : ranking scores (any real number), also used for AUPRC
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    root_mean_squared_error,
)


def _group_scores(y_score, inverse, num_groups, fusion_type, threshold):
    """Compute aggregated scores per group according to fusion_type.

    Args:
        y_score: Per-instance scores.
        inverse: Inverse mapping from instance to group index.
        num_groups: Total number of groups.
        fusion_type: Aggregation strategy — ``"mean"`` (group mean),
            ``"vote"`` (majority direction subset mean), or ``"all"``
            (whole-group mean for unanimous groups, otherwise majority
            subset mean).
        threshold: Classification threshold.

    Returns:
        Array of aggregated scores, one per group.

    Raises:
        ValueError: If fusion_type is unsupported.
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
    """Accumulator for batch-level predictions and epoch-level metrics.

    Responsibilities:
    1. Collect batch-level predictions and labels.
    2. Compute epoch-level aggregated metrics.

    Metric persistence is handled by MetricLogger; this class only
    performs computation.

    Example:
        >>> accum = MetricsAccumulator()
        >>> accum.reset("train")
        >>> for batch in dataloader:
        ...     outputs = model(batch)
        ...     accum.update("train", outputs)
        >>> metrics = accum.compute("train")
    """

    def __init__(self):
        """Initialize the accumulator with an empty internal store."""
        self._accumulators: dict[str, dict[str, list]] = {}

    def reset(self, phase: str):
        """Reset accumulators for a given phase.

        Args:
            phase: Phase name, e.g. ``"train"`` or ``"val"``.
        """
        self._accumulators[phase] = {
            "y_label": [],
            "y_pred": [],
            "y_score": [],
            "y_prob": [],
            "group_id": [],
        }

    def update(self, phase: str, outputs: dict[str, torch.Tensor]):
        """Update accumulators with a batch of model outputs.

        Args:
            phase: Phase name, e.g. ``"train"`` or ``"val"``.
            outputs: Dict containing ``"y_label"``, ``"y_predict"``,
                ``"y_score"``, and ``"y_prob"`` tensors.
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
        """Compute epoch-level metrics for the given phase.

        For train/val phases, returns acc/auc/auprc/rmse.
        For test phase with group_id provided, returns per-fusion
        metrics (``{fusion}_acc``, ``{fusion}_auc``, ``{fusion}_auprc``, ``{fusion}_rmse``).

        Args:
            phase: Phase name, e.g. ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            Dictionary mapping metric names to values. Empty dict if
            no data has been accumulated for the given phase.
        """
        if phase not in self._accumulators:
            return {}

        accum = self._accumulators[phase]
        if not accum["y_label"]:
            return {}

        # Test phase with group_id: compute per-fusion metrics
        if phase == "test" and accum["group_id"]:
            group_id: np.ndarray = torch.cat(accum["group_id"]).numpy()
            y_label_raw: np.ndarray = torch.cat(accum["y_label"]).numpy()
            y_score_raw: np.ndarray = torch.cat(accum["y_score"]).numpy()

            uniq_groups, inverse = np.unique(group_id, return_inverse=True)
            num_groups = uniq_groups.shape[0]

            # Label: take the first value per group; verify consistency
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

                try:
                    metrics[f"{fusion}_auprc"] = float(
                        average_precision_score(group_label, group_score)
                    )
                except ValueError:
                    metrics[f"{fusion}_auprc"] = 0.0
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

        try:
            metrics["auprc"] = float(average_precision_score(y_label, y_score))
        except ValueError:
            metrics["auprc"] = 0.0

        metrics["rmse"] = float(root_mean_squared_error(y_label, y_prob))

        return metrics


__all__ = ["MetricsAccumulator"]
