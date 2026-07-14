"""Environment metadata + background resource sampling.

- :mod:`.env_info`: one-shot hardware/software snapshot (:func:`collect_environment`).
- :mod:`.sampling`: descriptor-driven, stage-scoped resource sampling.
"""

from .env_info import EnvironmentInfo, collect_environment
from .sampling import (
    RESOURCE_METRICS,
    MetricSampler,
    ResourceMetric,
    ResourceSampler,
    ResourceStats,
    ResourceSummary,
    StageScopedSampler,
)

__all__ = [
    "RESOURCE_METRICS",
    "EnvironmentInfo",
    "MetricSampler",
    "ResourceMetric",
    "ResourceSampler",
    "ResourceStats",
    "ResourceSummary",
    "StageScopedSampler",
    "collect_environment",
]
