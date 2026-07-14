"""Inference stage: latency distribution, throughput, peak memory."""

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


@dataclass
class InferenceMetrics:
    """Inference efficiency metrics."""

    iters: int = 0
    repeats: int = 0
    batch_size: int = 0
    valid_tokens_per_batch: int = 0
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
    throughput_interactions_per_sec: float = 0.0
    ns_per_interaction: float = 0.0
    gpu_peak_allocated_mib: float | None = None
    gpu_peak_reserved_mib: float | None = None


@dataclass
class InferenceStageConfig:
    """Inference stage knobs."""

    iters: int = 200
    repeats: int = 3


def benchmark_inference(
    target,
    sample_batch,
    batch_size: int,
    valid_tokens: int,
    warmup_iters: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> InferenceMetrics:
    """Inference latency/throughput/memory benchmark.

    Reuses the prefetched ``sample_batch`` to avoid DataLoader IPC noise; a CUDA
    Event per iteration reads elapsed_time after ``end.synchronize()``, covering
    host launch through kernel completion.
    """
    model = target.model
    model.eval()
    dev = DeviceBackend(device)

    # warmup: cuDNN autotune / Inductor JIT / GPU clock ramp
    with torch.inference_mode():
        for _ in range(warmup_iters):
            target.forward(sample_batch)
    dev.sync()

    # Sustained throughput: wall-clock over iters without per-step sync,
    # matching the training benchmark so the two throughputs are comparable.
    with torch.inference_mode():
        sustained_start = time.perf_counter()
        for _ in range(iters):
            target.forward(sample_batch)
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
                        dev.time_step_events(lambda: target.forward(sample_batch))
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
    # Run-to-run stability: spread of per-repeat means (latency_cv captures within-repeat jitter).
    repeat_std = (
        float(np.std(per_repeat_means, ddof=1)) if len(per_repeat_means) > 1 else 0.0
    )
    repeat_cv = repeat_std / mean_ms if mean_ms > 0 else 0.0
    throughput = (valid_tokens * iters) / sustained_wall if sustained_wall > 0 else 0.0
    ns_per = (
        (sustained_wall * 1e9) / (valid_tokens * iters)
        if valid_tokens > 0 and iters > 0
        else 0.0
    )

    logger.info(
        f"[Inference] latency_mean={mean_ms:.3f}ms latency_p95={summary['p95']:.3f}ms "
        f"latency_cv={summary['cv']:.3f} repeat_cv={repeat_cv:.3f} | "
        f"throughput={throughput:,.0f} int/s"
        + (f" | gpu_peak={peak_alloc:.0f} MiB" if peak_alloc is not None else "")
    )

    return InferenceMetrics(
        iters=iters,
        repeats=repeats,
        batch_size=batch_size,
        valid_tokens_per_batch=valid_tokens,
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
        throughput_interactions_per_sec=throughput,
        ns_per_interaction=ns_per,
        gpu_peak_allocated_mib=peak_alloc,
        gpu_peak_reserved_mib=peak_reserved,
    )


@register_efficiency_stage("inference")
class InferenceStage(EfficiencyStage):
    """Inference efficiency: latency distribution, throughput, peak memory."""

    name = "inference"
    priority = 20
    config_cls = InferenceStageConfig

    def run(self, ctx: StageContext) -> InferenceMetrics:
        """Benchmark inference latency, throughput, and peak GPU memory."""
        cfg = ctx.stage_cfg(self.name)
        return benchmark_inference(
            ctx.target,
            ctx.sample_batch,
            ctx.batch_size,
            ctx.valid_tokens,
            ctx.general.warmup_iters,
            cfg.iters,
            cfg.repeats,
            ctx.device,
        )

    @classmethod
    def format_table(cls, result: InferenceMetrics) -> Table | None:
        """Render inference latency/throughput as a Rich table."""
        table = cls.make_kv_table("Inference")
        table.add_row("Iterations", f"{result.iters} x {result.repeats}")
        table.add_row("Valid tokens / batch", f"{result.valid_tokens_per_batch:,}")
        table.add_row("Latency mean", f"{result.latency_mean_ms:.3f} ms")
        table.add_row(
            "Latency p50 / p95 / p99",
            f"{result.latency_p50_ms:.3f} / {result.latency_p95_ms:.3f} / {result.latency_p99_ms:.3f} ms",
        )
        table.add_row("Latency cv", f"{result.latency_cv:.3f}")
        table.add_row("Repeat cv (stability)", f"{result.latency_repeat_cv:.3f}")
        table.add_row(
            "Throughput (sustained)",
            f"{result.throughput_interactions_per_sec:,.0f} interactions/s",
        )
        table.add_row("Per interaction", f"{result.ns_per_interaction:,.0f} ns")
        if result.gpu_peak_allocated_mib is not None:
            table.add_row(
                "GPU peak (allocated)", f"{result.gpu_peak_allocated_mib:,.0f} MiB"
            )
        if result.gpu_peak_reserved_mib is not None:
            table.add_row(
                "GPU peak (reserved)", f"{result.gpu_peak_reserved_mib:,.0f} MiB"
            )
        return table
