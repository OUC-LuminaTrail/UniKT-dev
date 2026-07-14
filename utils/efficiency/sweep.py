"""Batch-size sweep: run the efficiency benchmark across multiple batch sizes.

Each size rebuilds a fresh trainer (reusing data_src) under its own sub-dir and
prints its full efficiency report, so per-size data is captured faithfully
without cross-size metric aggregation. A lightweight index lists the runs.
"""

import gc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from rich.console import Console

from utils.config import config_to_dict
from utils.core import TRAINERS, get_logger
from utils.experiment_manager import ExperimentManager, ExperimentType

from .report import EfficiencyReport
from .session import EfficiencySession

logger = get_logger(__name__)


@dataclass
class SweepRun:
    """Location of one per-size efficiency report within the sweep dir."""

    batch_size: int
    dir: str


@dataclass
class SweepReport:
    """Lightweight sweep index: which sizes ran and where their reports live.

    Holds no per-size metrics — each size's full EfficiencyReport stays in its
    own sub-directory; this index only points to them.
    """

    model_name: str
    dataset_name: str
    timestamp: str
    batch_sizes: list[int]
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
            "batch_sizes": self.batch_sizes,
            "modes": self.modes,
            "config": self.config,
            "sweep_dir": self.sweep_dir,
            "runs": [{"batch_size": r.batch_size, "dir": r.dir} for r in self.runs],
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"[Sweep] index saved to {path}")


class EfficiencySweep:
    """Run the efficiency benchmark across a set of batch sizes.

    Each size rebuilds a fresh trainer (reusing data_src) under its own
    sub-experiment directory and prints its full report; the sweep keeps a
    lightweight index rather than aggregating metrics across sizes.
    """

    def __init__(self, rc, eff_cfg, data_src, weights_path: str | None = None) -> None:
        """Bind run config, data source, and optional weights; create the sweep dir."""
        self.rc = rc
        self.cfg = eff_cfg
        self.data_src = data_src
        self.weights_path = weights_path
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
        """Sweep all parsed batch sizes, printing each report and indexing the runs."""
        batch_sizes = _parse_batch_sizes(self.cfg.batch_sizes)
        original_bs = self.rc.model.batch_size
        if self.cfg.output_dir:
            logger.warning(
                "[Sweep] --efficiency.output_dir ignored in sweep mode; "
                "each size writes to <sweep_dir>/bs{N}/."
            )
        logger.info(f"[Sweep] sweep_dir={self.sweep_dir} batch_sizes={batch_sizes}")

        runs: list[SweepRun] = []
        modes: list[str] = []
        for bs in batch_sizes:
            self.rc.model.batch_size = bs
            logger.info(f"[Sweep] === batch_size={bs} ===")
            try:
                report, child_dir = self._run_single(bs)
            except Exception as e:
                logger.error(f"[Sweep] bs={bs} failed, skipping: {e}", exc_info=True)
                continue
            finally:
                self._cleanup_cuda()
            runs.append(SweepRun(bs, child_dir))
            if not modes:
                modes = report.modes
        self.rc.model.batch_size = original_bs

        if not runs:
            raise SystemExit("[Sweep] no batch size completed successfully.")
        sweep_report = SweepReport(
            model_name=self.rc.experiment.model_name,
            dataset_name=self.rc.data.dataset,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            batch_sizes=[r.batch_size for r in runs],
            modes=modes,
            config=config_to_dict(self.cfg),
            sweep_dir=self.sweep_dir,
            runs=runs,
        )
        sweep_report.write_json(Path(self.sweep_dir) / "sweep_index.json")
        self._print_summary(sweep_report)
        return sweep_report

    def _run_single(self, bs: int) -> tuple[EfficiencyReport, str]:
        """Build a fresh trainer for one size; run + print its report; return (report, dir)."""
        child_exp = self.parent_exp.create_sub_experiment(f"bs{bs}")
        trainer = TRAINERS.get(self.rc.experiment.model_name)(
            rc=self.rc, data_src=self.data_src, exp_manager=child_exp
        )
        if self.weights_path:
            trainer.load_weights(self.weights_path)
        report = EfficiencySession(
            trainer=trainer,
            rc=self.rc,
            eff_cfg=self.cfg,
            output_dir=child_exp.get_log_dir(),
        ).run()
        report.print_console()
        return report, child_exp.get_log_dir()

    def _print_summary(self, sweep_report: SweepReport) -> None:
        """Print a short index of which sizes ran and where their reports live."""
        console = Console()
        console.print()
        console.print(
            f"[bold cyan]Efficiency Sweep[/] done  "
            f"[white]{sweep_report.model_name}[/] on [white]{sweep_report.dataset_name}[/]  "
            f"({sweep_report.timestamp})"
        )
        console.print(
            f"  batch_sizes={','.join(str(b) for b in sweep_report.batch_sizes)}  "
            f"modes={','.join(sweep_report.modes)}"
        )
        for run in sweep_report.runs:
            console.print(f"  bs{run.batch_size} -> {run.dir}")
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
        raise SystemExit("[Sweep] --efficiency.batch_sizes is empty.")
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


__all__ = ["EfficiencySweep", "SweepReport"]
