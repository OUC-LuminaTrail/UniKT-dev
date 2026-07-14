"""Forward FLOPs and disk-size estimation via ``torch.utils.flop_counter``."""

from collections.abc import Callable
from typing import Any

import torch

from utils.core import get_logger

logger = get_logger(__name__)


def estimate_disk_size_mb(model: torch.nn.Module) -> float:
    """Estimate state_dict serialization size (params + buffers, by element_size)."""
    size_bytes = 0
    for p in model.parameters():
        size_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        size_bytes += b.numel() * b.element_size()
    return size_bytes / 1024**2


def count_flops(
    forward_fn: Callable[[], Any], device: torch.device
) -> tuple[int | None, dict[str, int], str | None]:
    """Count forward FLOPs with ``torch.utils.flop_counter.FlopCounterMode``.

    Under cuDNN, fused ops like ``aten::_cudnn_lstm`` are not decomposable and
    FlopCounterMode undercounts, so cuDNN is temporarily disabled during measurement
    to force LSTM decomposition into countable ``aten::mm``/``addmm``. Latency/memory
    measurements still use the default cuDNN config — this trade-off is documented
    in ``flops_note``.
    """
    from torch.utils.flop_counter import FlopCounterMode

    note = (
        "Forward FLOPs measured with cuDNN disabled so LSTM/cuDNN-fused ops decompose "
        "into countable aten::mm; latency/memory use the default cuDNN config."
    )
    saved_cudnn = torch.backends.cudnn.enabled
    try:
        torch.backends.cudnn.enabled = False

        # Grad-enabled (NOT inference_mode/no_grad): FlopCounterMode's ModuleTracker
        # fw_pre_hook registers grad hooks that read grad_fn.next_functions; under
        # inference_mode grad_fn is None -> AttributeError. Forward builds a graph
        # (one pass, acceptable).
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        flops = flop_counter.get_total_flops()
        breakdown = format_breakdown(flop_counter)
        return flops, breakdown, note
    except Exception as e:
        logger.warning(f"[Profile] FLOPs measurement failed (non-fatal): {e}")
        return None, {}, note
    finally:
        torch.backends.cudnn.enabled = saved_cudnn


def format_breakdown(flop_counter) -> dict[str, int]:
    """Top-level aten op to FLOPs mapping from FlopCounterMode, sorted desc."""
    try:
        counts = flop_counter.get_flop_counts()
        global_counts = counts.get("Global", {})
        result: dict[str, int] = {}
        for op, val in global_counts.items():
            key = str(op).split(".")[-1] if "." in str(op) else str(op)
            result[key] = int(val)
        return dict(sorted(result.items(), key=lambda x: -x[1]))
    except Exception:
        return {}
