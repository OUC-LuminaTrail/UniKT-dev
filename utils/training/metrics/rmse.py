"""RMSE metric."""

from sklearn.metrics import root_mean_squared_error

from utils.core import register_metric

from .base import Metric, MetricContext


@register_metric("rmse")
class RMSEMetric(Metric):
    """Root mean squared error.

    Train/val: computed on predicted probabilities (``y_prob``).
    Test/group: computed on fused group scores.
    """

    def compute(self, ctx: MetricContext) -> dict[str, float]:
        """RMSE; per-fusion in test/group mode."""
        if ctx.groups:
            return {
                f"{fusion}_rmse": float(root_mean_squared_error(label, score))
                for fusion, (label, score) in ctx.groups.items()
            }
        return {"rmse": float(root_mean_squared_error(ctx.y_label, ctx.y_prob))}
