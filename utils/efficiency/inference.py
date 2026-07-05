"""Inference benchmark: latency distribution, throughput, peak memory."""

import time
from dataclasses import dataclass

import numpy as np
import torch

from utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceMetrics:
    """推理效率指标。"""

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
    throughput_interactions_per_sec: float = 0.0
    ns_per_interaction: float = 0.0
    gpu_peak_allocated_mib: float | None = None
    gpu_peak_reserved_mib: float | None = None


def benchmark_inference(
    trainer,
    sample_batch,
    batch_size: int,
    valid_tokens: int,
    warmup_iters: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> InferenceMetrics:
    """推理延迟/吞吐/显存基准。

    预取的 ``sample_batch`` 复用，规避 DataLoader IPC 噪声；CUDA Event 在每次迭代
    ``end.synchronize()`` 后读 elapsed_time，覆盖主机发射到 kernel 完成的真实时间。
    """
    model = trainer.model
    model.eval()

    # warmup: cuDNN autotune / Inductor JIT / GPU 时钟爬升
    with torch.inference_mode():
        for _ in range(warmup_iters):
            trainer.forward_pass(sample_batch)
    synchronize(device)

    all_latencies: list[float] = []
    peak_alloc: float | None = None
    peak_reserved: float | None = None

    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        latencies_ms: list[float] = []
        with torch.inference_mode():
            for _ in range(iters):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    trainer.forward_pass(sample_batch)
                    end.record()
                    end.synchronize()
                    latencies_ms.append(start.elapsed_time(end))
                else:
                    t0 = time.perf_counter()
                    trainer.forward_pass(sample_batch)
                    latencies_ms.append((time.perf_counter() - t0) * 1000)
        all_latencies.extend(latencies_ms)

        if device.type == "cuda":
            alloc = torch.cuda.max_memory_allocated(device) / 1024**2
            reserved = torch.cuda.max_memory_reserved(device) / 1024**2
            peak_alloc = alloc if peak_alloc is None else max(peak_alloc, alloc)
            peak_reserved = (
                reserved if peak_reserved is None else max(peak_reserved, reserved)
            )

    summary = summarize_latencies(all_latencies)
    mean_ms = summary["mean"]
    throughput = valid_tokens / (mean_ms / 1000) if mean_ms > 0 else 0.0
    ns_per = (mean_ms * 1e6) / valid_tokens if valid_tokens > 0 else 0.0

    logger.info(
        f"[Inference] latency_mean={mean_ms:.3f}ms latency_p95={summary['p95']:.3f}ms "
        f"latency_cv={summary['cv']:.3f} | throughput={throughput:,.0f} int/s"
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
        throughput_interactions_per_sec=throughput,
        ns_per_interaction=ns_per,
        gpu_peak_allocated_mib=peak_alloc,
        gpu_peak_reserved_mib=peak_reserved,
    )


def synchronize(device: torch.device) -> None:
    """CUDA 上等所有 kernel 完成；CPU 空操作。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def extract_mask(batch) -> torch.Tensor | None:
    """从原始 batch 提取训练 mask ``[B, S]``。

    tuple 形式 mask 在索引 2（SAKT/AKT/DKT/DKTForget 一致）；GIKT 是 dict，键 "mask"。
    """
    if isinstance(batch, dict):
        return batch.get("mask")
    if isinstance(batch, (tuple, list)) and len(batch) > 2:
        return batch[2]
    return None


def count_valid_tokens(batch) -> int:
    """有效交互数 = ``mask.sum()``；不可得时退化为 ``B*S``。"""
    mask = extract_mask(batch)
    if mask is not None and isinstance(mask, torch.Tensor):
        return int(mask.sum().item())
    first = _first_tensor(batch)
    if first is not None and first.dim() >= 2:
        return int(first.size(0) * first.size(1))
    return 0


def batch_size_of(batch) -> int:
    """batch 行数（学生序列数 B）。"""
    first = _first_tensor(batch)
    return int(first.size(0)) if first is not None and first.dim() >= 1 else 0


def _first_tensor(batch) -> torch.Tensor | None:
    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v
    elif isinstance(batch, (tuple, list)):
        for v in batch:
            if isinstance(v, torch.Tensor):
                return v
    return None


def summarize_latencies(xs_ms: list[float]) -> dict:
    """延迟分布统计：mean/std/p50/p95/p99/min/max/cv。"""
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
