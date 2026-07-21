"""Cohen's Kappa metric."""

import numpy as np
from sklearn.metrics import cohen_kappa_score

from utils.core import register_metric

from .base import Metric, MetricContext


@register_metric("kappa")
class KappaMetric(Metric):
    """Cohen's Kappa (agreement corrected for chance).

    Train/val: computed on the model's binary predictions (``y_pred``).
    Test/group: computed on thresholding the fused group score at 0.5.
    """

    def compute(self, ctx: MetricContext) -> dict[str, float]:
        """Cohen's Kappa; per-fusion in test/group mode."""
        if ctx.groups:
            return {
                f"{fusion}_kappa": float(
                    cohen_kappa_score(label, (score >= 0.5).astype(np.float64))
                )
                for fusion, (label, score) in ctx.groups.items()
            }
        return {"kappa": float(cohen_kappa_score(ctx.y_label, ctx.y_pred))}
