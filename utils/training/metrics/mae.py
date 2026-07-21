"""MAE (mean absolute error) metric."""

from sklearn.metrics import mean_absolute_error

from utils.core import register_metric

from .base import Metric, MetricContext


@register_metric("mae")
class MAEMetric(Metric):
    """Mean absolute error.

    Train/val: computed on predicted probabilities (``y_prob``).
    Test/group: computed on fused group scores.
    """

    def compute(self, ctx: MetricContext) -> dict[str, float]:
        """MAE; per-fusion in test/group mode."""
        if ctx.groups:
            return {
                f"{fusion}_mae": float(mean_absolute_error(label, score))
                for fusion, (label, score) in ctx.groups.items()
            }
        return {"mae": float(mean_absolute_error(ctx.y_label, ctx.y_prob))}
