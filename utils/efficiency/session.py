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

from .environment import ResourceSampler, collect_environment
from .measures.batch import (
    batch_size_of,
    count_test_predictions,
    count_valid_interactions,
    to_device,
)
from .report import EfficiencyReport
from .stages.base import EfficiencyStage, StageContext
from .target import TrainerBenchmarkAdapter

logger = get_logger(__name__)


def build_target(rc, data_src, exp_manager, weights_path: str | None = None):
    """Build a trainer from ``rc`` and wrap it as a :class:`BenchmarkTarget`.

    Shared by the single-run entry and the per-point sweep so the two never
    duplicate the build + optional weight-load + adapter-wrap sequence.
    """
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
        """Run the enabled stages under a shared resource sampler and assemble the report."""
        seed_everything(
            self.rc.general.seed, deterministic=not self.rc.general.no_deterministic
        )

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

        test_batch, test_batch_size, test_predictions = self._prefetch_test_batch(
            device
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
            test_batch=test_batch,
            test_batch_size=test_batch_size,
            test_valid_tokens=test_predictions,
        )

        sampler = ResourceSampler(device, self.cfg.general.resource_sample_interval)
        sampler.start()
        results: dict[str, Any] = {}
        resources: dict[str, Any] = {}
        try:
            for name, stage in self.stages:
                logger.info(f"[{name}] running ...")
                sampler.begin_stage(name)
                try:
                    results[name] = stage.run(ctx)
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
                "deterministic": not self.rc.general.no_deterministic,
                **environment.determinism_dict(),
            },
            environment=environment,
            resource=resources,
            results=results,
        )

        if self.output_dir is not None:
            report.write_json(self.output_dir / "efficiency_report.json")
        return report

    def _prefetch_test_batch(self, device) -> tuple:
        """Prefetch one test batch, but only if a stage needs the test split.

        The windowlate loader streams parquet, so materializing a batch is not
        free — profile/inference/train-only runs skip it entirely. A failure here
        degrades to "no test batch" (the stage reports itself skipped) rather
        than aborting stages that never touch the test path.
        """
        if not any(
            getattr(stage, "requires_test_data", False) for _, stage in self.stages
        ):
            return None, 0, 0

        loader = self.target.test_data
        if loader is None:
            logger.warning("[Setup] target exposes no test loader; test stages skipped")
            return None, 0, 0

        try:
            batch = to_device(next(iter(loader)), device)
        except StopIteration:
            logger.warning("[Setup] test loader is empty; test stages skipped")
            return None, 0, 0

        batch_size = batch_size_of(batch)
        try:
            predictions = count_test_predictions(self.target, batch)
        except Exception as exc:
            logger.warning(f"[Setup] test forward failed ({exc}); test stages skipped")
            return None, 0, 0
        logger.info(
            f"[Setup] test_batch_size={batch_size} test_predictions={predictions}"
        )
        return batch, batch_size, predictions


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
