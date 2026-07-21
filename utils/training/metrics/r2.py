"""R² (squared Pearson correlation) metric."""

from utils.core import register_metric

from .base import Metric, MetricContext
from .grouping import _pearson_r2


@register_metric("r2")
class R2Metric(Metric):
    """Squared Pearson correlation between truth and prediction.

    Train/val: computed on predicted probabilities (``y_prob``). Test/group:
    computed on fused group scores. Returns 0.0 when either input has zero
    variance (correlation undefined).
    """

    def compute(self, ctx: MetricContext) -> dict[str, float]:
        """Squared Pearson r²; per-fusion in test/group mode."""
        if ctx.groups:
            return {
                f"{fusion}_r2": _pearson_r2(label, score)
                for fusion, (label, score) in ctx.groups.items()
            }
        return {"r2": _pearson_r2(ctx.y_label, ctx.y_prob)}
