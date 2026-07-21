"""Metrics package: pluggable, auto-discovered evaluation metrics.

Each metric lives in its own module under this package and registers via
``@register_metric("name")``. Importing this package triggers static discovery
(component modules are scanned without being imported; a metric module is
imported lazily only when its value is first requested), mirroring the
efficiency-stages pattern. Add a metric by dropping a decorated class into
this directory — no manual registration elsewhere.

Example::

    # utils/training/metrics/ndcg.py
    @register_metric("ndcg")
    class NDCGMetric(Metric):
        def compute(self, ctx):
            ...
"""

from pathlib import Path

from utils.core import discover_registrations

from .accumulator import MetricsAccumulator
from .base import Metric, MetricContext

# Populate the METRICS lazy index by scanning this package (no sub-module
# imports yet). Must run before any ``compute()`` call reads ``METRICS.keys()``;
# placing it after the class imports above keeps all imports at the top for the
# formatter while still preceding runtime metric resolution.
discover_registrations(Path(__file__).parent, "utils.training.metrics")

__all__ = ["Metric", "MetricContext", "MetricsAccumulator"]
