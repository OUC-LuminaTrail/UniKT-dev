"""Efficiency report: dataclass assembly + JSON/Rich console output."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from utils.core import get_logger

from .environment import EnvironmentInfo, ResourceStats
from .inference import InferenceMetrics
from .model_profile import ModelProfile
from .training import TrainingMetrics

logger = get_logger(__name__)


@dataclass
class EfficiencyReport:
    """完整效率评估报告。"""

    model_name: str
    dataset_name: str
    timestamp: str
    batch_size: int
    seq_len: int | None
    modes: list[str]
    config: dict[str, Any]
    determinism: dict[str, Any]
    environment: EnvironmentInfo
    model_profile: ModelProfile | None = None
    inference: InferenceMetrics | None = None
    training: TrainingMetrics | None = None
    resource: ResourceStats | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"[Report] saved to {path}")

    def print_console(self) -> None:
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

        if self.model_profile is not None:
            console.print(_profile_table(self.model_profile))
            console.print()
        if self.inference is not None:
            console.print(_inference_table(self.inference))
            console.print()
        if self.training is not None:
            console.print(_training_table(self.training))
            console.print()
        if self.resource is not None:
            console.print(_resource_table(self.resource))
            console.print()
        console.print(_environment_table(self.environment))
        console.print()


def _profile_table(profile: ModelProfile) -> Table:
    table = Table(
        title="Model Profile", title_style="bold cyan", show_header=False, box=None
    )
    table.add_column("Key", style="yellow", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Parameters", f"{profile.params:,}")
    table.add_row("Trainable params", f"{profile.trainable_params:,}")
    table.add_row("Model size", f"{profile.model_size_mb:.2f} MB")
    if profile.flops_forward is not None:
        table.add_row("FLOPs / forward", _format_flops(profile.flops_forward))
    if profile.op_breakdown:
        top = list(profile.op_breakdown.items())[:3]
        breakdown = ", ".join(f"{k}={_format_flops(v)}" for k, v in top)
        table.add_row("Top ops", breakdown)
    return table


def _inference_table(inf: InferenceMetrics) -> Table:
    table = Table(
        title="Inference", title_style="bold green", show_header=False, box=None
    )
    table.add_column("Key", style="yellow", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Iterations", f"{inf.iters} × {inf.repeats}")
    table.add_row("Valid tokens / batch", f"{inf.valid_tokens_per_batch:,}")
    table.add_row("Latency mean", f"{inf.latency_mean_ms:.3f} ms")
    table.add_row(
        "Latency p50 / p95 / p99",
        f"{inf.latency_p50_ms:.3f} / {inf.latency_p95_ms:.3f} / {inf.latency_p99_ms:.3f} ms",
    )
    table.add_row("Latency cv", f"{inf.latency_cv:.3f}")
    table.add_row(
        "Throughput", f"{inf.throughput_interactions_per_sec:,.0f} interactions/s"
    )
    table.add_row("Per interaction", f"{inf.ns_per_interaction:,.0f} ns")
    if inf.gpu_peak_allocated_mib is not None:
        table.add_row("GPU peak (allocated)", f"{inf.gpu_peak_allocated_mib:,.0f} MiB")
    if inf.gpu_peak_reserved_mib is not None:
        table.add_row("GPU peak (reserved)", f"{inf.gpu_peak_reserved_mib:,.0f} MiB")
    return table


def _training_table(tr: TrainingMetrics) -> Table:
    table = Table(
        title="Training (pseudo loop)",
        title_style="bold green",
        show_header=False,
        box=None,
    )
    table.add_column("Key", style="yellow", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Iterations", f"{tr.iters}")
    table.add_row("Per step", f"{tr.ms_per_train_step:.3f} ms")
    table.add_row(
        "Throughput", f"{tr.throughput_interactions_per_sec:,.0f} interactions/s"
    )
    table.add_row("Samples/s", f"{tr.samples_per_sec:,.0f}")
    table.add_row("Wall time", _format_duration(tr.wall_time_s))
    if tr.gpu_peak_allocated_mib is not None:
        table.add_row("GPU peak (allocated)", f"{tr.gpu_peak_allocated_mib:,.0f} MiB")
    if tr.gpu_peak_reserved_mib is not None:
        table.add_row("GPU peak (reserved)", f"{tr.gpu_peak_reserved_mib:,.0f} MiB")
    return table


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


def _format_flops(flops: int) -> str:
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} G"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} M"
    if flops >= 1e3:
        return f"{flops / 1e3:.2f} K"
    return f"{flops}"


def _format_duration(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"
