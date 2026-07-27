"""Windowlate stage: efficiency of the sliding-window evaluation path.

Skill-level models are scored with windowlate data, where every sample carries a
full history but is evaluated at its final position only. Their real serving cost
is therefore one forward pass per *single* prediction, while the ``inference``
stage measures the training-shaped path that yields a prediction per timestep.
Reporting only the latter overstates skill-level throughput by roughly the
sequence length, so this stage measures the test path on its own terms and
reports the amortization gap explicitly.

Skipped (with a reason on the report) when the target has no test loader or its
test batches are not windowlate-shaped — question-level models score dense
sequences and are already covered by ``inference``.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from rich.table import Table

from utils.core import get_logger, register_efficiency_stage

from ..device import DeviceBackend
from ..measures.timing import summarize_latencies
from .base import EfficiencyStage, StageContext

logger = get_logger(__name__)

# Windowlate batches carry (sequence, response, mask, late_group_id, true_labels,
# question); models with extra features append to that, never fewer.
WINDOWLATE_MIN_FIELDS = 6


@dataclass
class WindowlateMetrics:
    """Windowlate evaluation-path efficiency metrics."""

    supported: bool = True
    skip_reason: str = ""
    iters: int = 0
    repeats: int = 0
    batch_size: int = 0
    predictions_per_batch: int = 0
    train_path_tokens_per_batch: int = 0
    amortization_ratio: float = 0.0
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_cv: float = 0.0
    latency_repeat_std_ms: float = 0.0
    latency_repeat_cv: float = 0.0
    per_repeat_mean_ms: list[float] = field(default_factory=list)
    throughput_predictions_per_sec: float = 0.0
    us_per_prediction: float = 0.0
    gpu_peak_allocated_mib: float | None = None
    gpu_peak_reserved_mib: float | None = None


@dataclass
class WindowlateStageConfig:
    """Windowlate stage knobs."""

    iters: int = 200
    repeats: int = 3


def benchmark_windowlate(
    target,
    test_batch,
    batch_size: int,
    predictions: int,
    train_tokens: int,
    warmup_iters: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> WindowlateMetrics:
    """Latency/throughput of the windowlate evaluation forward pass.

    Mirrors :func:`benchmark_inference`'s timing structure — an unsynchronized
    sustained loop for throughput, CUDA-event timing per iteration for the
    latency distribution — so the two stages differ only in which path they
    exercise and what the throughput denominator counts.
    """
    dev = DeviceBackend(device)

    with torch.inference_mode():
        for _ in range(warmup_iters):
            target.test_forward(test_batch)
    dev.sync()

    with torch.inference_mode():
        sustained_start = time.perf_counter()
        for _ in range(iters):
            target.test_forward(test_batch)
        dev.sync()
    sustained_wall = time.perf_counter() - sustained_start

    all_latencies: list[float] = []
    per_repeat_means: list[float] = []
    peak_alloc: float | None = None
    peak_reserved: float | None = None

    for _ in range(repeats):
        with dev.peak_memory() as mem:
            latencies_ms: list[float] = []
            with torch.inference_mode():
                for _ in range(iters):
                    latencies_ms.append(
                        dev.time_step_events(lambda: target.test_forward(test_batch))
                    )
        all_latencies.extend(latencies_ms)
        if latencies_ms:
            per_repeat_means.append(sum(latencies_ms) / len(latencies_ms))

        if dev.is_cuda:
            peak_alloc = (
                mem.allocated_mib
                if peak_alloc is None
                else max(peak_alloc, mem.allocated_mib)
            )
            peak_reserved = (
                mem.reserved_mib
                if peak_reserved is None
                else max(peak_reserved, mem.reserved_mib)
            )

    summary = summarize_latencies(all_latencies)
    mean_ms = summary["mean"]
    repeat_std = (
        float(np.std(per_repeat_means, ddof=1)) if len(per_repeat_means) > 1 else 0.0
    )
    repeat_cv = repeat_std / mean_ms if mean_ms > 0 else 0.0
    throughput = (predictions * iters) / sustained_wall if sustained_wall > 0 else 0.0
    us_per = (
        (sustained_wall * 1e6) / (predictions * iters)
        if predictions > 0 and iters > 0
        else 0.0
    )
    # How much cheaper a prediction looks when the same forward is credited with
    # every timestep instead of the one position windowlate actually scores.
    amortization = train_tokens / predictions if predictions > 0 else 0.0

    logger.info(
        f"[Windowlate] latency_mean={mean_ms:.3f}ms latency_p95={summary['p95']:.3f}ms "
        f"repeat_cv={repeat_cv:.3f} | throughput={throughput:,.0f} pred/s "
        f"| {predictions} pred/batch (amortization x{amortization:,.1f})"
        + (f" | gpu_peak={peak_alloc:.0f} MiB" if peak_alloc is not None else "")
    )

    return WindowlateMetrics(
        iters=iters,
        repeats=repeats,
        batch_size=batch_size,
        predictions_per_batch=predictions,
        train_path_tokens_per_batch=train_tokens,
        amortization_ratio=amortization,
        latency_mean_ms=mean_ms,
        latency_std_ms=summary["std"],
        latency_p50_ms=summary["p50"],
        latency_p95_ms=summary["p95"],
        latency_p99_ms=summary["p99"],
        latency_min_ms=summary["min"],
        latency_max_ms=summary["max"],
        latency_cv=summary["cv"],
        latency_repeat_std_ms=repeat_std,
        latency_repeat_cv=repeat_cv,
        per_repeat_mean_ms=per_repeat_means,
        throughput_predictions_per_sec=throughput,
        us_per_prediction=us_per,
        gpu_peak_allocated_mib=peak_alloc,
        gpu_peak_reserved_mib=peak_reserved,
    )


def is_windowlate_batch(batch) -> bool:
    """Whether a test batch carries windowlate's group_id / true_label columns."""
    return isinstance(batch, (tuple, list)) and len(batch) >= WINDOWLATE_MIN_FIELDS


@register_efficiency_stage("windowlate")
class WindowlateStage(EfficiencyStage):
    """Windowlate evaluation-path efficiency: per-prediction latency and throughput."""

    name = "windowlate"
    priority = 25
    requires_test_data = True
    config_cls = WindowlateStageConfig

    def run(self, ctx: StageContext) -> WindowlateMetrics:
        """Benchmark the windowlate test path, or record why it was skipped."""
        if ctx.test_batch is None:
            return self._skip("target has no test loader")
        if not is_windowlate_batch(ctx.test_batch):
            return self._skip(
                "test batches are not windowlate-shaped "
                f"(expected >={WINDOWLATE_MIN_FIELDS} fields); "
                "question-level models are covered by the inference stage"
            )
        if ctx.test_valid_tokens <= 0:
            return self._skip("test forward produced no scored predictions")

        cfg = ctx.stage_cfg(self.name)
        return benchmark_windowlate(
            ctx.target,
            ctx.test_batch,
            ctx.test_batch_size,
            ctx.test_valid_tokens,
            ctx.valid_tokens,
            ctx.general.warmup_iters,
            cfg.iters,
            cfg.repeats,
            ctx.device,
        )

    @staticmethod
    def _skip(reason: str) -> WindowlateMetrics:
        logger.info(f"[Windowlate] skipped: {reason}")
        return WindowlateMetrics(supported=False, skip_reason=reason)

    @classmethod
    def format_table(cls, result: WindowlateMetrics) -> Table | None:
        """Render windowlate per-prediction latency/throughput as a Rich table."""
        table = cls.make_kv_table("Windowlate (evaluation path)")
        if not result.supported:
            table.add_row("Skipped", result.skip_reason)
            return table
        table.add_row("Iterations", f"{result.iters} x {result.repeats}")
        table.add_row("Predictions / batch", f"{result.predictions_per_batch:,}")
        table.add_row(
            "Train-path tokens / batch", f"{result.train_path_tokens_per_batch:,}"
        )
        table.add_row(
            "Amortization vs train path", f"x{result.amortization_ratio:,.1f}"
        )
        table.add_row("Latency mean", f"{result.latency_mean_ms:.3f} ms")
        table.add_row(
            "Latency p50 / p95 / p99",
            f"{result.latency_p50_ms:.3f} / {result.latency_p95_ms:.3f} / {result.latency_p99_ms:.3f} ms",
        )
        table.add_row("Latency cv", f"{result.latency_cv:.3f}")
        table.add_row("Repeat cv (stability)", f"{result.latency_repeat_cv:.3f}")
        table.add_row(
            "Throughput (sustained)",
            f"{result.throughput_predictions_per_sec:,.0f} predictions/s",
        )
        table.add_row("Per prediction", f"{result.us_per_prediction:,.1f} us")
        if result.gpu_peak_allocated_mib is not None:
            table.add_row(
                "GPU peak (allocated)", f"{result.gpu_peak_allocated_mib:,.0f} MiB"
            )
        if result.gpu_peak_reserved_mib is not None:
            table.add_row(
                "GPU peak (reserved)", f"{result.gpu_peak_reserved_mib:,.0f} MiB"
            )
        return table
