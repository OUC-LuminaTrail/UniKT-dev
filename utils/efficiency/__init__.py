"""Model efficiency benchmark module.

Standalone efficiency benchmark: build model + data, then measure model scale,
compute cost, inference efficiency, training efficiency (peak memory +
throughput), and system/environment metadata under controlled conditions for a
fair, reproducible efficiency comparison.

Stages are pluggable: each lives under :mod:`utils.efficiency.stages`, registers
via ``@register_efficiency_stage("name")``, and is auto-discovered. Entry:
``python efficiency.py -m MODEL -d DATASET``.
"""

from . import stages  # noqa: F401  — triggers static stage discovery
from .config import EfficiencyConfig
from .environment import (
    EnvironmentInfo,
    ResourceSampler,
    ResourceStats,
    ResourceSummary,
    collect_environment,
)
from .inference import InferenceMetrics, benchmark_inference
from .model_profile import ModelProfile, profile_model
from .report import EfficiencyReport
from .session import EfficiencySession
from .stages.base import EfficiencyStage, StageContext
from .trace import TraceProfile, benchmark_trace
from .training import TrainingMetrics, benchmark_training

__all__ = [
    "EfficiencyConfig",
    "EfficiencyReport",
    "EfficiencySession",
    "EfficiencyStage",
    "EnvironmentInfo",
    "InferenceMetrics",
    "ModelProfile",
    "ResourceSampler",
    "ResourceStats",
    "ResourceSummary",
    "StageContext",
    "TraceProfile",
    "TrainingMetrics",
    "benchmark_inference",
    "benchmark_trace",
    "benchmark_training",
    "collect_environment",
    "profile_model",
]
