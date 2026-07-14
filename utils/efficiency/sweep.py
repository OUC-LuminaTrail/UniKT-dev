"""Axis sweep: run the efficiency benchmark across a set of config mutations.

A sweep is a list of :class:`SweepPoint` ``(label, mutate)`` entries. Each point
runs on a fresh deepcopy of the run config with its ``mutate`` applied, rebuilds
a trainer under its own sub-dir (clean CUDA allocator, isolated peak memory),
and prints its full efficiency report. :func:`batch_size_sweep` is the built-in
factory; ``seq_len``/precision/fold sweeps are the same shape.

A lightweight index lists the runs; per-point metrics stay in each point's own
report (no cross-point aggregation).
"""

import copy
import gc
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console

from utils.config import config_to_dict
from utils.core import get_logger
from utils.experiment_manager import ExperimentManager, ExperimentType

from .report import EfficiencyReport
from .session import EfficiencySession, build_target

logger = get_logger(__name__)


@dataclass
class SweepPoint:
    """One sweep axis point: a label and a config mutator.

    ``mutate(rc, eff_cfg)`` applies absolute values to the (already deepcopied)
    run/efficiency config, so points are independent and need no restore.
    """

    label: str
    mutate: Callable[..., None]


@dataclass
class SweepRun:
    """Location of one point's efficiency report within the sweep dir."""

    label: str
    dir: str


@dataclass
class SweepReport:
    """Lightweight sweep index: which points ran and where their reports live.

    Holds no per-point metrics — each point's full EfficiencyReport stays in its
    own sub-directory; this index only points to them.
    """

    model_name: str
    dataset_name: str
    timestamp: str
    labels: list[str]
    modes: list[str]
    config: dict
    sweep_dir: str
    runs: list[SweepRun]

    def write_json(self, path: str | Path) -> None:
        """Write the sweep index as JSON, creating parent dirs as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "modes": self.modes,
            "config": self.config,
            "sweep_dir": self.sweep_dir,
            "runs": [{"label": r.label, "dir": r.dir} for r in self.runs],
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"[Sweep] index saved to {path}")


def batch_size_sweep(sizes: list[int]) -> list[SweepPoint]:
    """Build sweep points that vary ``rc.model.batch_size`` across ``sizes``."""
    return [
        SweepPoint(
            f"bs{size}",
            lambda rc, _eff_cfg, s=size: setattr(rc.model, "batch_size", s),
        )
        for size in sizes
    ]


class EfficiencySweep:
    """Run the efficiency benchmark across a set of :class:`SweepPoint` entries.

    Defaults to :func:`batch_size_sweep` parsed from
    ``eff_cfg.general.batch_sizes``; pass ``points`` for any other axis.
    """

    def __init__(
        self,
        rc,
        eff_cfg,
        data_src,
        weights_path: str | None = None,
        points: list[SweepPoint] | None = None,
    ) -> None:
        """Bind run config, data source, optional weights, and sweep points."""
        self.rc = rc
        self.cfg = eff_cfg
        self.data_src = data_src
        self.weights_path = weights_path
        self.points = (
            points
            if points is not None
            else batch_size_sweep(_parse_batch_sizes(eff_cfg.general.batch_sizes))
        )
        tags = [f"fold{rc.data.fold}"] if rc.data.fold is not None else []
        tags.append("sweep")
        self.parent_exp = ExperimentManager(
            exp_type=ExperimentType.EFFICIENCY,
            model_name=rc.experiment.model_name,
            dataset_name=rc.data.dataset,
            base_dir="runs",
            tags=tags,
        )
        self.sweep_dir = self.parent_exp.get_log_dir()

    def run(self) -> SweepReport:
        """Run every point (printing each report) and index the runs."""
        if self.cfg.general.output_dir:
            logger.warning(
                "[Sweep] --efficiency.general.output_dir ignored in sweep mode; "
                "each point writes to <sweep_dir>/<label>/."
            )
        logger.info(
            f"[Sweep] sweep_dir={self.sweep_dir} points={[p.label for p in self.points]}"
        )

        runs: list[SweepRun] = []
        modes: list[str] = []
        for point in self.points:
            logger.info(f"[Sweep] === {point.label} ===")
            try:
                report, child_dir = self._run_point(point)
            except Exception as e:
                logger.error(
                    f"[Sweep] {point.label} failed, skipping: {e}", exc_info=True
                )
                continue
            finally:
                self._cleanup_cuda()
            runs.append(SweepRun(point.label, child_dir))
            if not modes:
                modes = report.modes

        if not runs:
            raise SystemExit("[Sweep] no sweep point completed successfully.")
        sweep_report = SweepReport(
            model_name=self.rc.experiment.model_name,
            dataset_name=self.rc.data.dataset,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            labels=[r.label for r in runs],
            modes=modes,
            config=config_to_dict(self.cfg),
            sweep_dir=self.sweep_dir,
            runs=runs,
        )
        sweep_report.write_json(Path(self.sweep_dir) / "sweep_index.json")
        self._print_summary(sweep_report)
        return sweep_report

    def _run_point(self, point: SweepPoint) -> tuple[EfficiencyReport, str]:
        """Apply ``point`` to a fresh rc copy, rebuild, run + print its report."""
        rc = copy.deepcopy(self.rc)
        point.mutate(rc, self.cfg)
        child_exp = self.parent_exp.create_sub_experiment(point.label)
        target = build_target(rc, self.data_src, child_exp, self.weights_path)
        report = EfficiencySession(
            target=target,
            rc=rc,
            eff_cfg=self.cfg,
            output_dir=child_exp.get_log_dir(),
        ).run()
        report.print_console()
        return report, child_exp.get_log_dir()

    def _print_summary(self, sweep_report: SweepReport) -> None:
        """Print a short index of which points ran and where their reports live."""
        console = Console()
        console.print()
        console.print(
            f"[bold cyan]Efficiency Sweep[/] done  "
            f"[white]{sweep_report.model_name}[/] on [white]{sweep_report.dataset_name}[/]  "
            f"({sweep_report.timestamp})"
        )
        console.print(
            f"  points={','.join(sweep_report.labels)}  "
            f"modes={','.join(sweep_report.modes)}"
        )
        for run in sweep_report.runs:
            console.print(f"  {run.label} -> {run.dir}")
        console.print()

    def _cleanup_cuda(self) -> None:
        """Release the prior trainer's tensors and reset CUDA peak stats.

        ``reset_peak_memory_stats`` raises on CPU, so the CUDA calls stay guarded;
        ``gc.collect`` reclaims optimizer<->model cycles regardless of device.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def _parse_batch_sizes(s: str) -> list[int]:
    """Parse comma-separated batch sizes: validate int > 0, de-duplicate preserving order."""
    raw = [t.strip() for t in s.split(",") if t.strip()]
    if not raw:
        raise SystemExit("[Sweep] --efficiency.general.batch_sizes is empty.")
    sizes: list[int] = []
    for t in raw:
        try:
            v = int(t)
        except ValueError:
            raise SystemExit(f"[Sweep] invalid batch_size '{t}' (must be int).")
        if v <= 0:
            raise SystemExit(f"[Sweep] batch_size must be > 0, got {v}.")
        sizes.append(v)
    return list(dict.fromkeys(sizes))


__all__ = ["EfficiencySweep", "SweepPoint", "SweepReport", "batch_size_sweep"]
