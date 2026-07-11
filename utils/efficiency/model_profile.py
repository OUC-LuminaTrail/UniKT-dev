"""Static model profile: parameter counts, disk size, FLOPs."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from utils.core import get_logger

logger = get_logger(__name__)


@dataclass
class ModelProfile:
    """模型静态画像。"""

    params: int = 0
    trainable_params: int = 0
    model_size_mb: float = 0.0
    flops_forward: int | None = None
    op_breakdown: dict[str, int] = field(default_factory=dict)
    flops_note: str | None = None


def profile_model(
    model: torch.nn.Module,
    forward_fn: Callable[[], Any],
    device: torch.device,
    count_flops: bool = True,
) -> ModelProfile:
    """统计参数量、磁盘大小，可选前向 FLOPs。

    Args:
        model: PyTorch 模型。
        forward_fn: 零参数可调用，执行一次完整前向（通常是 ``trainer.forward_pass(batch)``）。
            由调用方提供，保证模型 forward 签名正确，并触发 trainer 侧状态（如 GIKT 的 graph_data）。
        device: 模型所在设备。
        count_flops: 是否估算 FLOPs（默认 True）。
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = _estimate_disk_size_mb(model)

    profile = ModelProfile(
        params=total,
        trainable_params=trainable,
        model_size_mb=round(size_mb, 3),
    )

    if count_flops:
        flops, breakdown, note = _count_flops(forward_fn, device)
        profile.flops_forward = flops
        profile.op_breakdown = breakdown
        profile.flops_note = note

    if flops := profile.flops_forward:
        gflops = flops / 1e9
        flops_str = (
            f"{gflops:.2f} GFLOPs" if gflops >= 1 else f"{flops / 1e6:.2f} MFLOPs"
        )
        logger.info(
            f"[Profile] params={total:,} trainable={trainable:,} "
            f"size={size_mb:.2f}MB flops={flops_str}"
        )
    else:
        logger.info(
            f"[Profile] params={total:,} trainable={trainable:,} size={size_mb:.2f}MB"
        )

    return profile


def _estimate_disk_size_mb(model: torch.nn.Module) -> float:
    """模拟 state_dict 序列化大小（参数 + buffer，按各自动 element_size）。"""
    size_bytes = 0
    for p in model.parameters():
        size_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        size_bytes += b.numel() * b.element_size()
    return size_bytes / 1024**2


def _count_flops(
    forward_fn: Callable[[], Any],
    device: torch.device,
) -> tuple[int | None, dict[str, int], str | None]:
    """用 ``torch.utils.flop_counter.FlopCounterMode`` 统计前向 FLOPs。

    cuDNN 下 ``aten::_cudnn_lstm`` 等融合算子不可分解、FlopCounterMode 漏计，故测量期间
    临时关闭 cuDNN，强制 LSTM 分解为可计数的 ``aten::mm``/``addmm``。延迟/显存测量仍用默认
    cuDNN 配置 —— 该取舍在报告 ``flops_note`` 中明示。
    """
    from torch.utils.flop_counter import FlopCounterMode

    note = (
        "Forward FLOPs measured with cuDNN disabled so LSTM/cuDNN-fused ops decompose "
        "into countable aten::mm; latency/memory use the default cuDNN config."
    )
    saved_cudnn = torch.backends.cudnn.enabled
    try:
        torch.backends.cudnn.enabled = False

        # NOTE: must run grad-enabled (NOT inference_mode/no_grad). FlopCounterMode
        # starts a ModuleTracker whose fw_pre_hook calls register_multi_grad_hook →
        # _get_grad_fn_or_grad_acc reads t.grad_fn.next_functions; under inference_mode
        # grad_fn is None → AttributeError. forward builds a graph (one pass, acceptable).
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

        flops = flop_counter.get_total_flops()
        breakdown = _format_breakdown(flop_counter)
        return flops, breakdown, note
    except Exception as e:
        logger.warning(f"[Profile] FLOPs measurement failed (non-fatal): {e}")
        return None, {}, note
    finally:
        torch.backends.cudnn.enabled = saved_cudnn


def _format_breakdown(flop_counter) -> dict[str, int]:
    """从 FlopCounterMode 提取顶层 aten op → FLOPs 映射，按 FLOPs 降序。"""
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
