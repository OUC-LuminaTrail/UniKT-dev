"""ROC AUC metric."""

from sklearn.metrics import roc_auc_score

from utils.core import register_metric

from .base import Metric, MetricContext


def _safe_roc_auc(label, score) -> float:
    try:
        return float(roc_auc_score(label, score))
    except ValueError:
        return 0.0


@register_metric("auc")
class AUCMetric(Metric):
    """Area under the ROC curve.

    Train/val: computed on ranking scores (``y_score``). Test/group: computed
    on fused group scores. Returns 0.0 when undefined (single-class labels).
    """

    def compute(self, ctx: MetricContext) -> dict[str, float]:
        """ROC AUC; per-fusion in test/group mode, 0.0 when undefined."""
        if ctx.groups:
            return {
                f"{fusion}_auc": _safe_roc_auc(label, score)
                for fusion, (label, score) in ctx.groups.items()
            }
        return {"auc": _safe_roc_auc(ctx.y_label, ctx.y_score)}
