"""AUPRC (average precision) metric."""

from sklearn.metrics import average_precision_score

from utils.core import register_metric

from .base import Metric, MetricContext


def _safe_avg_precision(label, score) -> float:
    try:
        return float(average_precision_score(label, score))
    except ValueError:
        return 0.0


@register_metric("auprc")
class AUPRCMetric(Metric):
    """Area under the precision-recall curve (average precision).

    Train/val: computed on ranking scores (``y_score``). Test/group: computed
    on fused group scores. Returns 0.0 when undefined.
    """

    def compute(self, ctx: MetricContext) -> dict[str, float]:
        """Average precision; per-fusion in test/group mode, 0.0 when undefined."""
        if ctx.groups:
            return {
                f"{fusion}_auprc": _safe_avg_precision(label, score)
                for fusion, (label, score) in ctx.groups.items()
            }
        return {"auprc": _safe_avg_precision(ctx.y_label, ctx.y_score)}
