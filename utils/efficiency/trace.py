"""Computation trace: torch.profiler operator-level breakdown per segment.

Profiles two segments — a forward pass (eval + inference_mode) and a training
step (train mode, forward+backward+optimizer) — with ``torch.profiler``,
extracting the top operators by self device time and exporting a chrome/perfetto
trace per segment. Complements the static ``profile`` stage (parameter/FLOPs) and
the macroscopic ``inference``/``train`` stages (latency/throughput).
"""

import contextlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch

from utils.core import get_logger

from .inference import synchronize
from .training import _one_train_step

logger = get_logger(__name__)


@dataclass
class OperatorStat:
    """One operator's aggregated stats from torch.profiler key_averages."""

    name: str = ""
    calls: int = 0
    cpu_total_us: float = 0.0
    self_cpu_us: float = 0.0
    cuda_total_us: float = 0.0
    self_cuda_us: float = 0.0
    flops: int = 0
    self_cpu_mem_bytes: int = 0
    self_cuda_mem_bytes: int = 0


@dataclass
class TraceMetrics:
    """One computation segment's profiling result (forward or train step)."""

    mode: str = ""
    iters: int = 0
    total_cpu_time_us: float = 0.0
    total_cuda_time_us: float = 0.0
    operator_count: int = 0
    total_flops: int = 0
    gpu_peak_allocated_mib: float | None = None
    gpu_peak_reserved_mib: float | None = None
    top_operators: list[OperatorStat] = field(default_factory=list)
    trace_path: str | None = None
    note: str | None = None


@dataclass
class TraceProfile:
    """torch.profiler result: forward + train operator-level breakdown."""

    forward: TraceMetrics | None = None
    train: TraceMetrics | None = None


def benchmark_trace(
    trainer,
    sample_batch,
    warmup_iters: int,
    iters: int,
    top_ops: int,
    export: bool,
    output_dir: str | Path | None,
    device: torch.device,
) -> TraceProfile:
    """Profile the forward pass and a training step with torch.profiler.

    Each segment runs its own profiler session on the shared representative
    batch: forward under ``inference_mode``, training under grad-enabled mode
    mirroring ``_one_train_step``. Per-segment failures degrade to a ``note``
    rather than aborting the whole stage.
    """
    forward = _safe_segment(
        trainer,
        sample_batch,
        warmup_iters,
        iters,
        top_ops,
        export,
        output_dir,
        device,
        mode="forward",
    )
    train = _safe_segment(
        trainer,
        sample_batch,
        warmup_iters,
        iters,
        top_ops,
        export,
        output_dir,
        device,
        mode="train",
    )
    return TraceProfile(forward=forward, train=train)


def _safe_segment(
    trainer,
    sample_batch,
    warmup_iters: int,
    iters: int,
    top_ops: int,
    export: bool,
    output_dir: str | Path | None,
    device: torch.device,
    mode: str,
) -> TraceMetrics:
    """Run _profile_segment, degrading to a note on failure (non-fatal)."""
    try:
        return _profile_segment(
            trainer,
            sample_batch,
            warmup_iters,
            iters,
            top_ops,
            export,
            output_dir,
            device,
            mode,
        )
    except Exception as e:  # non-fatal: keep the stage running
        logger.warning(f"[Trace] {mode} segment failed (non-fatal): {e}")
        return TraceMetrics(mode=mode, note=f"profiling failed: {e}")


def _profile_segment(
    trainer,
    sample_batch,
    warmup_iters: int,
    iters: int,
    top_ops: int,
    export: bool,
    output_dir: str | Path | None,
    device: torch.device,
    mode: str,
) -> TraceMetrics:
    """Profile one segment (``mode`` = forward|train) and assemble TraceMetrics."""
    is_forward = mode == "forward"
    model = trainer.model
    if is_forward:
        model.eval()
    else:
        model.train()

    def step_fn() -> None:
        if is_forward:
            trainer.forward_pass(sample_batch)
        else:
            _one_train_step(trainer, sample_batch)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # warmup outside the profiler (cuDNN autotune / clock ramp), mirroring the
    # inference and training stages so the trace reflects steady-state kernels.
    _run_under_grad(step_fn, warmup_iters, is_forward)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    grad_ctx = torch.inference_mode() if is_forward else contextlib.nullcontext()
    with grad_ctx:
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            for _ in range(iters):
                step_fn()
                prof.step()
    synchronize(device)

    trace_path = _maybe_export_trace(prof, export, output_dir, mode)

    peak_alloc = None
    peak_reserved = None
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress profiler deprecation notices
        key_averages = prof.key_averages()
        ops = _extract_operators(key_averages, top_ops, device)
        total_cpu, total_cuda, op_count, total_flops = _aggregate(key_averages, device)

    metrics = TraceMetrics(
        mode=mode,
        iters=iters,
        total_cpu_time_us=total_cpu,
        total_cuda_time_us=total_cuda,
        operator_count=op_count,
        total_flops=total_flops,
        gpu_peak_allocated_mib=round(peak_alloc, 3) if peak_alloc is not None else None,
        gpu_peak_reserved_mib=round(peak_reserved, 3)
        if peak_reserved is not None
        else None,
        top_operators=ops,
        trace_path=str(trace_path) if trace_path else None,
    )
    logger.info(
        f"[Trace] {mode}: ops={op_count} "
        f"self_cpu={total_cpu / 1e3:.2f}ms self_cuda={total_cuda / 1e3:.2f}ms"
        + (f" gpu_peak={peak_alloc:.0f}MiB" if peak_alloc is not None else "")
        + (f" trace={Path(trace_path).name}" if trace_path else "")
    )
    return metrics


def _maybe_export_trace(
    prof, export: bool, output_dir: str | Path | None, mode: str
) -> Path | None:
    """Export a chrome trace for the segment, or None when disabled/no dir."""
    if not export or output_dir is None:
        return None
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / f"trace_{mode}.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof.export_chrome_trace(str(trace_path))
    return trace_path


def _run_under_grad(step_fn, iters: int, use_inference_mode: bool) -> None:
    """Run ``step_fn`` ``iters`` times under inference_mode (forward) or default grad."""
    ctx = torch.inference_mode() if use_inference_mode else contextlib.nullcontext()
    with ctx:
        for _ in range(iters):
            step_fn()


def _extract_operators(
    key_averages, top_ops: int, device: torch.device
) -> list[OperatorStat]:
    """Top-N operators sorted by self device time (CUDA on GPU, else CPU).

    Reads ``self_device_time_total`` (PyTorch >=2.8 renamed the deprecated
    ``self_cuda_time_total`` accessor). A host op such as ``aten::mm`` mirrors its
    child kernel's device time, so both the op and the kernel may appear in the
    list — matching ``prof.key_averages().table()`` behavior.
    """
    events = list(key_averages)
    has_cuda = device.type == "cuda"
    sort_attr = "self_device_time_total" if has_cuda else "self_cpu_time_total"
    events.sort(key=lambda e: _attr(e, sort_attr), reverse=True)
    return [
        OperatorStat(
            name=str(getattr(e, "key", "")),
            calls=int(getattr(e, "count", 0) or 0),
            cpu_total_us=_attr(e, "cpu_time_total"),
            self_cpu_us=_attr(e, "self_cpu_time_total"),
            cuda_total_us=_attr(e, "device_time_total") if has_cuda else 0.0,
            self_cuda_us=_attr(e, "self_device_time_total") if has_cuda else 0.0,
            flops=int(_attr(e, "flops")),
            self_cpu_mem_bytes=int(_attr(e, "self_cpu_memory_usage")),
            self_cuda_mem_bytes=(
                int(_attr(e, "self_device_memory_usage")) if has_cuda else 0
            ),
        )
        for e in events[:top_ops]
    ]


def _aggregate(key_averages, device: torch.device) -> tuple[float, float, int, int]:
    """Sum per-operator self CPU/CUDA time and FLOPs across all operators.

    CPU self time sums directly. CUDA self time counts only pure device-kernel
    events (``cpu_time_total == 0``): a host op such as ``aten::mm`` mirrors its
    child kernel's device time, so summing every event would double-count.
    """
    events = list(key_averages)
    total_cpu = sum(_attr(e, "self_cpu_time_total") for e in events)
    total_cuda = 0.0
    if device.type == "cuda":
        total_cuda = sum(
            _attr(e, "self_device_time_total")
            for e in events
            if _attr(e, "cpu_time_total") == 0.0
        )
    total_flops = sum(int(_attr(e, "flops")) for e in events)
    return total_cpu, total_cuda, len(events), total_flops


def _attr(event, name: str) -> float:
    """Read a profiler event attribute as float, tolerating absent/renamed fields."""
    value = getattr(event, name, 0)
    return float(value) if value else 0.0
