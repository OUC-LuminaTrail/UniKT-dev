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

from ..environment import EnvironmentInfo
from ..measures.formatting import (  # noqa: F401  — re-exported for stages
    format_duration,
    format_flops,
)
from ..target import BenchmarkTarget

# Title style for efficiency result table.
TITLE_STYLE = "bold cyan"


@dataclass
class StageContext:
    """Shared runtime context injected into every stage by the session.

    ``cfg`` is the composed EfficiencyConfig (general + per-stage sub-nodes),
    typed ``Any`` to avoid a config↔stages import cycle. Stages read it via
    :attr:`general` and :meth:`stage_cfg`.
    """

    target: BenchmarkTarget
    device: torch.device
    sample_batch: Any
    batch_size: int
    valid_tokens: int
    seq_len: int | None
    cfg: Any
    environment: EnvironmentInfo
    output_dir: Path | None = None
    # Populated only when a stage requiring the test split is enabled; the test
    # loader is skipped otherwise so profile-only runs stay cheap.
    test_batch: Any = None
    test_batch_size: int = 0
    test_valid_tokens: int = 0

    @property
    def general(self) -> Any:
        """Cross-stage config node (warmup, modes, routing, ...)."""
        return self.cfg.general

    def stage_cfg(self, name: str) -> Any:
        """This stage's own config sub-node (its registered name)."""
        return getattr(self.cfg, name)


class EfficiencyStage(ABC):
    """One pluggable efficiency measurement.

    Subclasses set ``name`` (registry key + report result key) and ``priority``
    (smaller runs earlier), then implement ``run`` and ``format_table``. Stages
    are stateless: all per-run state arrives via :class:`StageContext`.
    """

    name: ClassVar[str] = ""
    priority: ClassVar[int] = 100
    # Opt in to have the session prefetch a test batch into the context; loading
    # the test split is not free, so it stays off by default.
    requires_test_data: ClassVar[bool] = False

    @abstractmethod
    def run(self, ctx: StageContext) -> Any:
        """Execute the measurement and return a serializable result."""

    @classmethod
    @abstractmethod
    def format_table(cls, result: Any) -> Table | None:
        """Render the result as a Rich table, or ``None`` to print nothing."""

    @classmethod
    def make_kv_table(cls, title: str) -> Table:
        """Standard Key/Value result table (no header/frame) with uniform title."""
        table = Table(title=title, title_style=TITLE_STYLE, show_header=False, box=None)
        table.add_column("Key", style="yellow", no_wrap=True)
        table.add_column("Value", style="white")
        return table
