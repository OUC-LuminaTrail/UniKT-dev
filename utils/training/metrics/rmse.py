"""RMSE metric."""

from sklearn.metrics import root_mean_squared_error

from utils.core import register_metric

from .base import Metric


@register_metric("rmse")
class RMSEMetric(Metric):
    """Root mean squared error.

    Train/val: predicted probabilities (``y_prob``). Test/group: fused group
    scores.
    """

    name = "rmse"
    source = "y_prob"

    def score(self, y_true, y_value):
        """Return root mean squared error."""
        return float(root_mean_squared_error(y_true, y_value))
