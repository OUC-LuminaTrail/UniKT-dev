"""Trace stage: torch.profiler operator-level breakdown for forward + train."""

from pathlib import Path

from rich.table import Table

from utils.core import register_efficiency_stage

from ..trace import OperatorStat, TraceMetrics, TraceProfile, benchmark_trace
from .base import EfficiencyStage, StageContext, format_flops


@register_efficiency_stage("trace")
class TraceStage(EfficiencyStage):
    """torch.profiler operator-level breakdown for the forward pass and a train step."""

    name = "trace"
    priority = 40

    def run(self, ctx: StageContext) -> TraceProfile:
        """Profile the forward pass and a training step with torch.profiler."""
        return benchmark_trace(
            ctx.trainer,
            ctx.sample_batch,
            ctx.cfg.warmup_iters,
            ctx.cfg.trace_iters,
            ctx.cfg.trace_top_ops,
            ctx.cfg.trace_export,
            ctx.output_dir,
            ctx.device,
        )

    def format_table(self, result: TraceProfile) -> Table | None:
        """Render the forward + train operator breakdown as a Rich table."""
        table = Table(
            title="Computation Trace",
            title_style="bold cyan",
            show_header=False,
            box=None,
        )
        table.add_column("Key", style="yellow", no_wrap=True)
        table.add_column("Value", style="white")
        if result.forward:
            _add_segment_rows(table, result.forward, "Forward")
        if result.train:
            _add_segment_rows(table, result.train, "Training")
        return table


def _add_segment_rows(table: Table, m: TraceMetrics, label: str) -> None:
    """Append one segment's summary rows (per-op details live in JSON top_operators)."""
    is_cuda = m.total_cuda_time_us > 0
    total_us = m.total_cuda_time_us if is_cuda else m.total_cpu_time_us
    kind = "CUDA" if is_cuda else "CPU"

    table.add_row(f"{label} — total self time", f"{total_us / 1e3:.2f} ms ({kind})")
    table.add_row(f"{label} — operators", f"{m.operator_count}")
    if m.total_flops:
        table.add_row(f"{label} — FLOPs", format_flops(m.total_flops))
    if m.top_operators:
        top3 = ", ".join(
            f"{op.name} ({_pct(op, total_us, is_cuda)})" for op in m.top_operators[:3]
        )
        table.add_row(f"{label} — top ops", top3)
    if m.gpu_peak_allocated_mib is not None:
        table.add_row(
            f"{label} — GPU peak (alloc/reserved)",
            f"{m.gpu_peak_allocated_mib:,.0f} / {m.gpu_peak_reserved_mib:,.0f} MiB",
        )
    if m.trace_path:
        table.add_row(f"{label} — trace", Path(m.trace_path).name)
    if m.note:
        table.add_row(f"{label} — note", m.note)


def _pct(op: OperatorStat, total_us: float, is_cuda: bool) -> str:
    """Self device-time share of the segment, as a percentage string."""
    self_us = op.self_cuda_us if is_cuda else op.self_cpu_us
    pct = (self_us / total_us * 100) if total_us > 0 else 0.0
    return f"{pct:.1f}%"
