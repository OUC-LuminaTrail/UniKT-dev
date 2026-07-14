"""Efficiency stage contract: shared context + pluggable stage ABC.

A stage is one measurement dimension (params/FLOPs, inference latency, training
throughput, ...). Stages register via ``@register_efficiency_stage("name")`` and
are auto-discovered from this package — drop a file here, it shows up. The
session runs the enabled stages in ``priority`` order, each contributing a
serializable result to the report.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from rich.table import Table

from ..config import EfficiencyConfig
from ..environment import EnvironmentInfo


@dataclass
class StageContext:
    """Shared runtime context injected into every stage by the session."""

    trainer: Any
    device: torch.device
    sample_batch: Any
    batch_size: int
    valid_tokens: int
    seq_len: int | None
    cfg: EfficiencyConfig
    environment: EnvironmentInfo
    output_dir: Path | None = None


class EfficiencyStage(ABC):
    """One pluggable efficiency measurement.

    Subclasses set ``name`` (registry key + report result key) and ``priority``
    (smaller runs earlier), then implement ``run`` and ``format_table``. Stages
    are stateless: all per-run state arrives via :class:`StageContext`.
    """

    name: ClassVar[str] = ""
    priority: ClassVar[int] = 100

    @abstractmethod
    def run(self, ctx: StageContext) -> Any:
        """Execute the measurement and return a serializable result."""

    @abstractmethod
    def format_table(self, result: Any) -> Table | None:
        """Render the result as a Rich table, or ``None`` to print nothing."""


def format_flops(flops: int) -> str:
    """Human-readable FLOPs (K/M/G)."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} G"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} M"
    if flops >= 1e3:
        return f"{flops / 1e3:.2f} K"
    return f"{flops}"


def format_duration(seconds: float) -> str:
    """Human-readable wall-clock duration."""
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"
