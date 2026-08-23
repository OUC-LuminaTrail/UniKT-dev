"""Efficiency session: orchestrates registered stages + resource sampling.

Stages are resolved from :data:`utils.core.EFFICIENCY_STAGES` (filtered by
``rc.efficiency.modes``, ordered by ``priority``); the session is agnostic to
which stages exist. ``environment`` is collected once as shared context, and the
background ``ResourceSampler`` spans all stages.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config import config_to_dict
from utils.core import (
    EFFICIENCY_STAGES,
    TRAINERS,
    get_logger,
    get_supported_stages,
    seed_everything,
)

from .device import reclaim_memory
from .environment import ResourceSampler, collect_environment
from .measures.batch import batch_size_of, count_valid_interactions, to_device
from .report import EfficiencyReport
from .stages.base import EfficiencyStage, StageContext
from .target import TrainerBenchmarkAdapter

logger = get_logger(__name__)


def build_target(rc, data_src, exp_manager, weights_path: str | None = None):
    """Build a trainer from ``rc`` and wrap it as a :class:`BenchmarkTarget`.

    Shared by the single-run entry and the per-point sweep so the two never
    duplicate the build + optional weight-load + adapter-wrap sequence.
    """
    # Seed before model init so weights, the prefetched batch, and dropout are
    # stable across runs; deterministic mode stays off during benchmarking.
    seed_everything(rc.general.seed, deterministic=False)
    trainer = TRAINERS.get(rc.experiment.model_name)(
        rc=rc, data_src=data_src, exp_manager=exp_manager
    )
    if weights_path:
        trainer.load_weights(weights_path)
    return TrainerBenchmarkAdapter(trainer)


class EfficiencySession:
    """Coordinate one efficiency benchmark run over the enabled stages."""

    def __init__(
        self, target, rc, eff_cfg, output_dir: str | Path | None = None
    ) -> None:
        """Bind the benchmark target, run config, and enabled stages."""
        self.target = target
        self.rc = rc
        self.cfg = eff_cfg
        self.device = target.device
        self.output_dir = Path(output_dir) if output_dir else None
        modes = [m.strip() for m in eff_cfg.general.modes.split(",") if m.strip()]
        self.stages = _resolve_stages(modes)

    def run(self) -> EfficiencyReport:
        """Run the enabled stages under a shared resource sampler and assemble the report.

        A stage failure (e.g. CUDA OOM) is recorded in ``report.errors`` and
        skipped — later stages still run. Only when every requested stage fails
        is the report written and an error re-raised (non-zero exit).
        """
        device = self.device
        # Move model/loss onto the device via the target; the benchmark never
        # calls trainer.run(), so without this the forward moves inputs to device
        # and clashes with CPU-resident weights.
        self.target.prepare(device)
        environment = collect_environment(device, self.target.model)

        # Prefetch one representative batch; timing loops reuse it to avoid
        # DataLoader IPC noise.
        sample_batch = to_device(next(iter(self.target.train_data)), device)
        batch_size = batch_size_of(sample_batch)
        valid_tokens = count_valid_interactions(self.target, sample_batch)
        seq_len = getattr(self.rc.data, "max_seq_len", None)
        logger.info(
            f"[Setup] batch_size={batch_size} seq_len={seq_len} "
            f"valid_tokens={valid_tokens}"
        )

        ctx = StageContext(
            target=self.target,
            device=device,
            sample_batch=sample_batch,
            batch_size=batch_size,
            valid_tokens=valid_tokens,
            seq_len=seq_len,
            cfg=self.cfg,
            environment=environment,
            output_dir=self.output_dir,
        )

        sampler = ResourceSampler(device, self.cfg.general.resource_sample_interval)
        sampler.start()
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        resources: dict[str, Any] = {}
        last_exc: Exception | None = None
        try:
            for name, stage in self.stages:
                logger.info(f"[{name}] running ...")
                sampler.begin_stage(name)
                try:
                    results[name] = stage.run(ctx)
                except Exception as e:
                    # One failed stage (e.g. CUDA OOM) must not abort the rest:
                    # record it, reclaim memory, keep measuring.
                    errors[name] = f"{type(e).__name__}: {e}"
                    last_exc = e
                    logger.error(
                        f"[{name}] failed, continuing with next stage",
                        exc_info=True,
                    )
                    reclaim_memory(device)
                finally:
                    sampler.end_stage()
        finally:
            resources = sampler.stop()
        logger.info("[Report] assembling efficiency report ...")

        report = EfficiencyReport(
            model_name=self.rc.experiment.model_name,
            dataset_name=self.rc.data.dataset,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            batch_size=batch_size,
            seq_len=seq_len,
            modes=[name for name, _ in self.stages],
            config=config_to_dict(self.cfg),
            determinism={
                "seed": self.rc.general.seed,
                # Benchmarking always runs with deterministic algorithms off.
                "deterministic": False,
                **environment.determinism_dict(),
            },
            environment=environment,
            resource=resources,
            results=results,
            errors=errors,
        )

        if self.output_dir is not None:
            report.write_json(self.output_dir / "efficiency_report.json")
        if errors and not results:
            raise RuntimeError(
                f"All efficiency stages failed: {list(errors)}"
            ) from last_exc
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
