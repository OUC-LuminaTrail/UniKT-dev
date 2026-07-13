"""Efficiency session: orchestrates registered stages + resource sampling.

Stages are resolved from :data:`utils.core.EFFICIENCY_STAGES` (filtered by
``rc.efficiency.modes``, ordered by ``priority``); the session is agnostic to
which stages exist. ``environment`` is collected once as shared context, and the
background ``ResourceSampler`` spans all stages.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from utils.core import (
    EFFICIENCY_STAGES,
    get_logger,
    get_supported_stages,
    seed_everything,
)

from .environment import ResourceSampler, collect_environment
from .inference import batch_size_of, count_valid_interactions
from .report import EfficiencyReport
from .stages.base import EfficiencyStage, StageContext

logger = get_logger(__name__)


class EfficiencySession:
    """Coordinate one efficiency benchmark run over the enabled stages."""

    def __init__(self, trainer, rc, output_dir: str | Path | None = None) -> None:
        """Resolve the enabled stages from ``rc.efficiency.modes``."""
        self.trainer = trainer
        self.rc = rc
        self.cfg = rc.efficiency
        self.device = trainer.device_
        self.output_dir = Path(output_dir) if output_dir else None
        self.stages = _resolve_stages(list(self.cfg.modes))

    def run(self) -> EfficiencyReport:
        """Run the enabled stages under a shared resource sampler and assemble the report."""
        seed_everything(self.rc.general.seed, deterministic=not self.rc.general.no_deterministic)

        device = self.device
        # trainer.run() would migrate model/loss to device; we skip run(), so do it
        # manually — otherwise forward_pass moves inputs to device and clashes with
        # CPU-resident weights.
        self.trainer.model.to(device)
        if isinstance(self.trainer.loss, torch.nn.Module):
            self.trainer.loss.to(device)
        environment = collect_environment(device)

        # Prefetch one representative batch; timing loops reuse it to avoid
        # DataLoader IPC noise.
        sample_batch = _to_device(next(iter(self.trainer.train_data)), device)
        batch_size = batch_size_of(sample_batch)
        valid_tokens = count_valid_interactions(self.trainer, sample_batch)
        seq_len = getattr(self.rc.data, "max_seq_len", None)
        logger.info(
            f"[Setup] batch_size={batch_size} seq_len={seq_len} "
            f"valid_tokens={valid_tokens}"
        )

        ctx = StageContext(
            trainer=self.trainer,
            device=device,
            sample_batch=sample_batch,
            batch_size=batch_size,
            valid_tokens=valid_tokens,
            seq_len=seq_len,
            cfg=self.cfg,
            environment=environment,
        )

        sampler = ResourceSampler(device, self.cfg.resource_sample_interval)
        sampler.start()
        results: dict[str, Any] = {}
        try:
            for name, stage in self.stages:
                logger.info(f"[{name}] running ...")
                results[name] = stage.run(ctx)
        finally:
            resource = sampler.stop()
        logger.info("[Report] assembling efficiency report ...")

        report = EfficiencyReport(
            model_name=self.rc.experiment.model_name,
            dataset_name=self.rc.data.dataset,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            batch_size=batch_size,
            seq_len=seq_len,
            modes=[name for name, _ in self.stages],
            config=OmegaConf.to_container(self.cfg, resolve=True),
            determinism={
                "seed": self.rc.general.seed,
                "deterministic": not self.rc.general.no_deterministic,
                "cudnn_benchmark": environment.cudnn_benchmark,
                "cudnn_deterministic": environment.cudnn_deterministic,
                "deterministic_algorithms": environment.deterministic_algorithms,
            },
            environment=environment,
            resource=resource,
            results=results,
        )

        if self.output_dir is not None:
            report.write_json(self.output_dir / "efficiency_report.json")
        return report


def _resolve_stages(modes: list[str]) -> list[tuple[str, EfficiencyStage]]:
    """Resolve ``(registry_name, stage)`` pairs (empty ``modes`` = all), by priority.

    Keys/results use the registry name, not the stage's ``name`` ClassVar, so a
    stage whose ClassVar drifts from its registration key still renders. ``modes``
    is de-duplicated (preserving order) so a repeated stage runs once.
    """
    available = get_supported_stages()
    unknown = [m for m in modes if m not in available]
    if unknown:
        raise SystemExit(
            f"Unknown efficiency stage(s): {unknown}. Available: {available}"
        )
    selected = list(dict.fromkeys(modes if modes else available))
    stages = [(n, EFFICIENCY_STAGES.get(n)()) for n in selected]
    return sorted(stages, key=lambda ns: ns[1].priority)


def _to_device(batch, device: torch.device):
    """Recursively move batch tensors to device (tuple/list/dict aware)."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch
