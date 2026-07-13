"""Inference stage: latency distribution, throughput, peak memory."""

from rich.table import Table

from utils.core import register_efficiency_stage

from ..inference import InferenceMetrics, benchmark_inference
from .base import EfficiencyStage, StageContext


@register_efficiency_stage("inference")
class InferenceStage(EfficiencyStage):
    """Inference efficiency: latency distribution, throughput, peak memory."""

    name = "inference"
    priority = 20

    def run(self, ctx: StageContext) -> InferenceMetrics:
        """Benchmark inference latency, throughput, and peak GPU memory."""
        return benchmark_inference(
            ctx.trainer,
            ctx.sample_batch,
            ctx.batch_size,
            ctx.valid_tokens,
            ctx.cfg.warmup_iters,
            ctx.cfg.benchmark_iters,
            ctx.cfg.repeats,
            ctx.device,
        )

    def format_table(self, result: InferenceMetrics) -> Table | None:
        """Render inference latency/throughput as a Rich table."""
        table = Table(
            title="Inference", title_style="bold green", show_header=False, box=None
        )
        table.add_column("Key", style="yellow", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("Iterations", f"{result.iters} x {result.repeats}")
        table.add_row("Valid tokens / batch", f"{result.valid_tokens_per_batch:,}")
        table.add_row("Latency mean", f"{result.latency_mean_ms:.3f} ms")
        table.add_row(
            "Latency p50 / p95 / p99",
            f"{result.latency_p50_ms:.3f} / {result.latency_p95_ms:.3f} / {result.latency_p99_ms:.3f} ms",
        )
        table.add_row("Latency cv", f"{result.latency_cv:.3f}")
        table.add_row(
            "Throughput", f"{result.throughput_interactions_per_sec:,.0f} interactions/s"
        )
        table.add_row("Per interaction", f"{result.ns_per_interaction:,.0f} ns")
        if result.gpu_peak_allocated_mib is not None:
            table.add_row("GPU peak (allocated)", f"{result.gpu_peak_allocated_mib:,.0f} MiB")
        if result.gpu_peak_reserved_mib is not None:
            table.add_row("GPU peak (reserved)", f"{result.gpu_peak_reserved_mib:,.0f} MiB")
        return table
