"""Efficiency session: orchestrates profile + inference + training + resource."""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from utils.core import get_logger, seed_everything

from .environment import ResourceSampler, collect_environment
from .inference import (
    batch_size_of,
    benchmark_inference,
    count_valid_tokens,
    extract_mask,
)
from .model_profile import profile_model
from .report import EfficiencyReport
from .training import benchmark_training

logger = get_logger(__name__)


@dataclass
class EfficiencyConfig:
    """效率基准运行参数。"""

    warmup_iters: int = 50
    benchmark_iters: int = 200
    train_iters: int = 50
    repeats: int = 3
    sample_interval: float = 0.05
    profile_flops: bool = True


class EfficiencySession:
    """协调单次效率评估会话。

    构造时接收已 build 的 trainer，按选定 modes 依次跑 profile/inference/training，
    后台 ``ResourceSampler`` 贯穿测量，最后组装 ``EfficiencyReport``。
    """

    def __init__(self, trainer, args, output_dir: str | Path | None = None) -> None:
        self.trainer = trainer
        self.device = trainer.device_
        self.args = args
        self.output_dir = Path(output_dir) if output_dir else None
        self.cfg = EfficiencyConfig(
            warmup_iters=getattr(args, "warmup_iters", 50),
            benchmark_iters=getattr(args, "benchmark_iters", 200),
            train_iters=getattr(args, "train_iters", 50),
            repeats=getattr(args, "repeats", 3),
            sample_interval=getattr(args, "sample_interval", 0.05),
            profile_flops=getattr(args, "profile_flops", True),
        )
        self.modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    def run(self) -> EfficiencyReport:
        # seed + cuDNN/deterministic_algorithms 由 seed_everything 统一设置
        seed_everything(self.args.seed, deterministic=self.args.deterministic)

        device = self.device
        # trainer.run() 本会迁移 model/loss 到 device；这里不走 run()，需手动迁移，
        # 否则 forward_pass 把 input 迁到 device 后与 cpu 上的 model weight 冲突
        self.trainer.model.to(device)
        if isinstance(self.trainer.loss, torch.nn.Module):
            self.trainer.loss.to(device)
        environment = collect_environment(device)

        # 预取一个 representative batch；计时循环复用它以规避 DataLoader IPC 噪声
        sample_batch = _to_device(next(iter(self.trainer.train_data)), device)
        batch_size = batch_size_of(sample_batch)
        valid_tokens = count_valid_tokens(sample_batch)
        seq_len = _seq_len_of(sample_batch)
        logger.info(
            f"[Setup] batch_size={batch_size} seq_len={seq_len} "
            f"valid_tokens={valid_tokens}"
        )

        sampler = ResourceSampler(device, self.cfg.sample_interval)
        sampler.start()
        profile = inference = training = None
        try:
            if "profile" in self.modes:
                logger.info("[Profile] measuring params / size / FLOPs ...")
                profile = profile_model(
                    self.trainer.model,
                    forward_fn=lambda: self.trainer.forward_pass(sample_batch),
                    device=device,
                    count_flops=self.cfg.profile_flops,
                )
            if "inference" in self.modes:
                logger.info(
                    f"[Inference] starting "
                    f"(warmup={self.cfg.warmup_iters}, "
                    f"iters={self.cfg.benchmark_iters}×{self.cfg.repeats}) ..."
                )
                inference = benchmark_inference(
                    self.trainer,
                    sample_batch,
                    batch_size,
                    valid_tokens,
                    self.cfg.warmup_iters,
                    self.cfg.benchmark_iters,
                    self.cfg.repeats,
                    device,
                )
            if "train" in self.modes:
                logger.info(
                    f"[Training] starting "
                    f"(warmup={self.cfg.warmup_iters}, "
                    f"iters={self.cfg.train_iters}, with backward) ..."
                )
                training = benchmark_training(
                    self.trainer,
                    sample_batch,
                    batch_size,
                    valid_tokens,
                    self.cfg.warmup_iters,
                    self.cfg.train_iters,
                    device,
                )
        finally:
            resource = sampler.stop()
        logger.info("[Report] assembling efficiency report ...")

        report = EfficiencyReport(
            model_name=getattr(self.args, "model", ""),
            dataset_name=getattr(self.args, "dataset", ""),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            batch_size=batch_size,
            seq_len=seq_len,
            modes=self.modes,
            config=asdict(self.cfg),
            determinism={
                "seed": self.args.seed,
                "deterministic": bool(self.args.deterministic),
                "cudnn_benchmark": environment.cudnn_benchmark,
                "cudnn_deterministic": environment.cudnn_deterministic,
                "deterministic_algorithms": environment.deterministic_algorithms,
            },
            environment=environment,
            model_profile=profile,
            inference=inference,
            training=training,
            resource=resource,
        )

        if self.output_dir is not None:
            report.write_json(self.output_dir / "efficiency_report.json")
        return report


def _to_device(batch, device: torch.device):
    """递归把 batch 中的 tensor 移到 device（tuple/list/dict 感知）。"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def _seq_len_of(batch) -> int | None:
    """序列长度：仅序列级模型有 ``[B,S]`` mask 时定义；交互级模型（无 mask，如 DyGKT）返回 None。"""
    mask = extract_mask(batch)
    if isinstance(mask, torch.Tensor) and mask.dim() >= 2:
        return int(mask.size(1))
    return None
