"""Latency statistics and the shared forward-pass timing loop."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..device import DeviceBackend


@dataclass
class LatencyMetricsBase:
    """Latency/peak-memory fields shared by the forward-benchmark stages."""

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
    gpu_peak_allocated_mib: float | None = None
    gpu_peak_reserved_mib: float | None = None


@dataclass
class ForwardLoopStats(LatencyMetricsBase):
    """Raw timing output of :func:`benchmark_forward_loop`.

    Not a metrics type: ``sustained_wall_s`` exists so stages can derive their
    own throughput denominators, and must not leak into report JSON via
    metrics inheritance.
    """

    sustained_wall_s: float = 0.0


def benchmark_forward_loop(
    forward: Callable[[], Any],
    warmup_iters: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> ForwardLoopStats:
    """Time repeated forward calls: warmup, sustained throughput, latency loop.

    An unsynchronized sustained loop yields the wall-clock throughput figure;
    CUDA-event timing per iteration (``perf_counter`` on CPU) yields the latency
    distribution. Both stages that score forward paths share this rig so their
    numbers stay comparable; callables are passed bound to their batch.
    """
    dev = DeviceBackend(device)

    # warmup: cuDNN autotune / Inductor JIT / GPU clock ramp
    with torch.inference_mode():
        for _ in range(warmup_iters):
            forward()
    dev.sync()

    # Sustained throughput: wall-clock over iters without per-step sync,
    # matching the training benchmark so the two throughputs are comparable.
    with torch.inference_mode():
        sustained_start = time.perf_counter()
        for _ in range(iters):
            forward()
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
                    latencies_ms.append(dev.time_step_events(forward))
        all_latencies.extend(latencies_ms)
        if latencies_ms:
            per_repeat_means.append(sum(latencies_ms) / len(latencies_ms))

        if dev.is_cuda:
            if mem.allocated_mib is not None:
                peak_alloc = max(peak_alloc or 0.0, mem.allocated_mib)
            if mem.reserved_mib is not None:
                peak_reserved = max(peak_reserved or 0.0, mem.reserved_mib)

    summary = summarize_latencies(all_latencies)
    mean_ms = summary["mean"]
    # Run-to-run stability: spread of per-repeat means (latency_cv captures within-repeat jitter).
    repeat_std = (
        float(np.std(per_repeat_means, ddof=1)) if len(per_repeat_means) > 1 else 0.0
    )
    repeat_cv = repeat_std / mean_ms if mean_ms > 0 else 0.0
    return ForwardLoopStats(
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
        gpu_peak_allocated_mib=peak_alloc,
        gpu_peak_reserved_mib=peak_reserved,
        sustained_wall_s=sustained_wall,
    )


def summarize_latencies(xs_ms: list[float]) -> dict:
    """Latency distribution statistics: mean/std/p50/p95/p99/min/max/cv."""
    if not xs_ms:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "cv": 0.0,
        }
    a = np.asarray(xs_ms, dtype=np.float64)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
        "cv": cv,
    }
