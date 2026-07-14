"""Training stage: peak memory + throughput via a pseudo train loop."""

from rich.table import Table

from utils.core import register_efficiency_stage

from ..training import TrainingMetrics, benchmark_training
from .base import EfficiencyStage, StageContext, format_duration


@register_efficiency_stage("train")
class TrainingStage(EfficiencyStage):
    """Training efficiency: peak memory + throughput via a pseudo train loop."""

    name = "train"
    priority = 30

    def run(self, ctx: StageContext) -> TrainingMetrics:
        """Benchmark training peak memory and throughput over a pseudo loop."""
        return benchmark_training(
            ctx.trainer,
            ctx.sample_batch,
            ctx.batch_size,
            ctx.valid_tokens,
            ctx.cfg.warmup_iters,
            ctx.cfg.train_iters,
            ctx.device,
        )

    def format_table(self, result: TrainingMetrics) -> Table | None:
        """Render training throughput/memory as a Rich table."""
        table = Table(
            title="Training (pseudo loop)",
            title_style="bold green",
            show_header=False,
            box=None,
        )
        table.add_column("Key", style="yellow", no_wrap=True)
        table.add_column("Value", style="white")
        table.add_row("Iterations", f"{result.iters}")
        table.add_row("Per step", f"{result.ms_per_train_step:.3f} ms")
        table.add_row(
            "Throughput (sustained)",
            f"{result.throughput_interactions_per_sec:,.0f} interactions/s",
        )
        table.add_row("Samples/s", f"{result.samples_per_sec:,.0f}")
        table.add_row("Wall time", format_duration(result.wall_time_s))
        if result.gpu_peak_allocated_mib is not None:
            table.add_row(
                "GPU peak (allocated)", f"{result.gpu_peak_allocated_mib:,.0f} MiB"
            )
        if result.gpu_peak_reserved_mib is not None:
            table.add_row(
                "GPU peak (reserved)", f"{result.gpu_peak_reserved_mib:,.0f} MiB"
            )
        return table
