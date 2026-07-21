"""Accuracy metric."""

import numpy as np
from sklearn.metrics import accuracy_score

from utils.core import register_metric

from .base import Metric, MetricContext


@register_metric("acc")
class AccuracyMetric(Metric):
    """Classification accuracy.

    Train/val: accuracy of the model's binary predictions (``y_pred``).
    Test/group: accuracy of thresholding the fused group score at 0.5.
    """

    def compute(self, ctx: MetricContext) -> dict[str, float]:
        """Accuracy; per-fusion in test/group mode."""
        if ctx.groups:
            return {
                f"{fusion}_acc": float(
                    accuracy_score(label, (score >= 0.5).astype(np.float64))
                )
                for fusion, (label, score) in ctx.groups.items()
            }
        return {"acc": float(accuracy_score(ctx.y_label, ctx.y_pred))}
