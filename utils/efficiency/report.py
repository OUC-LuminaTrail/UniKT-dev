"""Efficiency report: stage-agnostic container + JSON/Rich console output.

The report holds each stage's result under its registry name in ``results``;
console rendering looks the stage up in ``EFFICIENCY_STAGES`` and asks it to
format its own table. Adding a stage never touches this file — its result and
table appear automatically.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from utils.core import EFFICIENCY_STAGES, get_logger

from .environment import EnvironmentInfo, ResourceStats

logger = get_logger(__name__)


@dataclass
class EfficiencyReport:
    """Stage-agnostic efficiency report."""

    model_name: str
    dataset_name: str
    timestamp: str
    batch_size: int
    seq_len: int | None
    modes: list[str]
    config: dict[str, Any]
    determinism: dict[str, Any]
    environment: EnvironmentInfo
    resource: ResourceStats | None
    results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the report (including nested stage results) to a dict."""
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        """Write the report as JSON, creating parent dirs as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"[Report] saved to {path}")

    def print_console(self) -> None:
        """Print the report: one table per stage result, then resource + environment."""
        console = Console()
        console.print()
        console.print(
            f"[bold cyan]Efficiency Report[/]  "
            f"[white]{self.model_name}[/] on [white]{self.dataset_name}[/]  "
            f"({self.timestamp})"
        )
        console.print(
            f"  batch_size={self.batch_size}  seq_len={self.seq_len}  "
            f"modes={','.join(self.modes)}"
        )
        console.print()

        for name in self.modes:
            result = self.results.get(name)
            if result is None:
                continue
            stage = EFFICIENCY_STAGES.get(name)()
            table = stage.format_table(result)
            if table is not None:
                console.print(table)
                console.print()

        if self.resource is not None:
            console.print(_resource_table(self.resource))
            console.print()
        console.print(_environment_table(self.environment))
        console.print()


def _resource_table(stats: ResourceStats) -> Table:
    table = Table(
        title="Resource Usage", title_style="bold magenta", show_header=False, box=None
    )
    table.add_column("Key", style="yellow", no_wrap=True)
    table.add_column("Mean", style="white")
    table.add_column("Peak", style="white")
    table.add_row("CPU%", *_rs(stats.cpu_percent))
    table.add_row("Process RSS", *_rs(stats.process_rss_mib, "MiB"))
    table.add_row("GPU util", *_rs(stats.gpu_util_pct, "%"))
    table.add_row("GPU power", *_rs(stats.gpu_power_w, "W"))
    table.add_row("GPU mem used", *_rs(stats.gpu_mem_used_mib, "MiB"))
    table.add_row("GPU temp", *_rs(stats.gpu_temp_c, "C"))
    if all(
        getattr(stats, f).n == 0
        for f in (
            "cpu_percent",
            "process_rss_mib",
            "gpu_util_pct",
            "gpu_power_w",
            "gpu_mem_used_mib",
            "gpu_temp_c",
        )
    ):
        table.add_row("(no samples)", "", "")
    return table


def _rs(summary, unit: str = "") -> tuple[str, str]:
    if summary.n == 0:
        return ("—", "—")
    mean = f"{summary.mean:.1f} {unit}".strip()
    peak = f"{summary.peak:.1f} {unit}".strip()
    return (mean, peak)


def _environment_table(env: EnvironmentInfo) -> Table:
    table = Table(
        title="Environment", title_style="bold blue", show_header=False, box=None
    )
    table.add_column("Key", style="yellow", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Device", env.device_type)
    if env.gpu_name:
        table.add_row("GPU", f"{env.gpu_name} ({env.gpu_capability})")
    if env.gpu_total_memory_gib:
        table.add_row("GPU memory", f"{env.gpu_total_memory_gib:.1f} GiB")
    table.add_row(
        "CPU",
        f"{env.cpu_model or 'unknown'} ({env.cpu_physical_cores or '?'}c/{env.cpu_logical_cores or '?'}t)",
    )
    if env.total_ram_gib:
        table.add_row("RAM", f"{env.total_ram_gib:.1f} GiB")
    table.add_row("PyTorch", env.torch_version)
    if env.cuda_version:
        table.add_row("CUDA", env.cuda_version)
    if env.cudnn_version:
        table.add_row(
            "cuDNN",
            f"{env.cudnn_version} (benchmark={env.cudnn_benchmark}, det={env.cudnn_deterministic})",
        )
    table.add_row("Deterministic algos", str(env.deterministic_algorithms))
    table.add_row("Python", env.python_version)
    table.add_row("Platform", env.platform)
    return table
