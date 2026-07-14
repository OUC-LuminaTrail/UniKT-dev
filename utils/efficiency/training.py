"""Training benchmark: peak memory + throughput via a pseudo train loop."""

import time
from dataclasses import dataclass

import torch

from utils.core import get_logger

from .inference import synchronize

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Training efficiency metrics (measured via a pseudo train loop)."""

    iters: int = 0
    batch_size: int = 0
    valid_tokens_per_batch: int = 0
    wall_time_s: float = 0.0
    ms_per_train_step: float = 0.0
    throughput_interactions_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    ns_per_interaction: float = 0.0
    gpu_peak_allocated_mib: float | None = None
    gpu_peak_reserved_mib: float | None = None


def benchmark_training(
    trainer,
    sample_batch,
    batch_size: int,
    valid_tokens: int,
    warmup_iters: int,
    iters: int,
    device: torch.device,
) -> TrainingMetrics:
    """Training peak memory + throughput benchmark.

    Mirrors ``BaseTrainer._run_train_batch``: ``zero_grad -> forward_pass -> _compute_loss ->
    backward -> clip -> step``. Does not call ``_run_train_batch`` directly to avoid
    ``metrics_accumulator`` overhead, keeping throughput measurement clean.
    """
    model = trainer.model
    model.train()

    # warmup: includes backward pass to fill cudnn backward autotune + Adam momentum to steady state
    for _ in range(warmup_iters):
        _one_train_step(trainer, sample_batch)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    start = time.perf_counter()
    for _ in range(iters):
        _one_train_step(trainer, sample_batch)
    synchronize(device)
    wall = time.perf_counter() - start

    peak_alloc = None
    peak_reserved = None
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    throughput = (valid_tokens * iters) / wall if wall > 0 else 0.0
    samples_per_sec = (batch_size * iters) / wall if wall > 0 else 0.0
    ms_per_step = (wall / iters) * 1000 if iters > 0 else 0.0
    ns_per = (
        (wall / iters) * 1e9 / valid_tokens if valid_tokens > 0 and iters > 0 else 0.0
    )

    logger.info(
        f"[Training] step={ms_per_step:.3f}ms | throughput={throughput:,.0f} int/s"
        + (f" | gpu_peak={peak_alloc:.0f} MiB" if peak_alloc is not None else "")
    )

    return TrainingMetrics(
        iters=iters,
        batch_size=batch_size,
        valid_tokens_per_batch=valid_tokens,
        wall_time_s=wall,
        ms_per_train_step=ms_per_step,
        throughput_interactions_per_sec=throughput,
        samples_per_sec=samples_per_sec,
        ns_per_interaction=ns_per,
        gpu_peak_allocated_mib=peak_alloc,
        gpu_peak_reserved_mib=peak_reserved,
    )


def _one_train_step(trainer, batch) -> None:
    """Mirror the pure computation path of ``BaseTrainer._run_train_batch``."""
    trainer.opt.zero_grad(set_to_none=True)
    out = trainer.forward_pass(batch)
    loss = trainer._compute_loss(out)
    loss.backward()
    if trainer.max_clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), max_norm=trainer.max_clip_grad_norm
        )
    trainer.opt.step()
