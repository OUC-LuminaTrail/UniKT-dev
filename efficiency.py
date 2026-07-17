"""UniKT model efficiency benchmark.

Two mutually exclusive entry modes:
    - ``-m/-d``                       build the model fresh (random weights, or
                                      ``--efficiency.general.weights`` to load a file) + data
    - ``--efficiency.general.run_dir`` seed the RunConfig from a trained run's
                                      ``run_config.yaml`` and benchmark its checkpoint

Usage:
    python efficiency.py -m GIKT -d assistments09
    python efficiency.py -m SAKT -d assistments09 --efficiency.general.weights runs/.../best_model.pth
    python efficiency.py --efficiency.general.run_dir runs/normal/GIKT_assist09_..._fold0_bs128
    python efficiency.py -m AKT -d assistments09 --efficiency.general.modes inference --efficiency.inference.iters 500
    python efficiency.py -m SAKT -d assistments09 --efficiency.general.compile_modes off,default,reduce-overhead
    python efficiency.py -m SAKT -d assistments09 --efficiency.general.batch_sizes 32,64 --efficiency.general.compile_modes off,default
"""

import sys
from pathlib import Path

import model  # noqa: F401  — triggers trainer/model-config discovery
from utils.config import ConfigParser, build_node
from utils.core import get_logger
from utils.data_process import get_data_source
from utils.efficiency import EfficiencySession, EfficiencySweep
from utils.efficiency.config import get_efficiency_config_cls
from utils.efficiency.session import build_target
from utils.experiment_manager import ExperimentManager, ExperimentType

logger = get_logger(__name__)


def main() -> None:
    """Build the model, run all enabled efficiency stages, and print the report."""
    rc, eff_cfg = _parse()
    # Suppress trainer side effects irrelevant to benchmarking.
    rc.general.swanlab = False
    rc.general.checkpoint_path = None
    rc.general.skip_test = True

    weights_path = _resolve_weights(eff_cfg)
    logger.info(
        f"[Benchmark] model={rc.experiment.model_name} dataset={rc.data.dataset}"
    )

    if eff_cfg.general.batch_sizes or eff_cfg.general.compile_modes:
        _run_sweep(rc, eff_cfg, weights_path)
    else:
        _run_single_efficiency(rc, eff_cfg, weights_path)


def _run_single_efficiency(rc, eff_cfg, weights_path: str | None) -> None:
    """Build one trainer and run a single efficiency session."""
    exp_manager = ExperimentManager.from_run_config(rc, ExperimentType.EFFICIENCY)
    output_dir = eff_cfg.general.output_dir or exp_manager.get_log_dir()
    logger.info(f"[Benchmark] output_dir={output_dir}")

    data_src = get_data_source(rc)
    if weights_path:
        logger.info(f"[Benchmark] loading weights: {weights_path}")
    target = build_target(rc, data_src, exp_manager, weights_path)
    EfficiencySession(
        target=target, rc=rc, eff_cfg=eff_cfg, output_dir=output_dir
    ).run().print_console()


def _run_sweep(rc, eff_cfg, weights_path: str | None) -> None:
    """Sweep a set of batch sizes, rebuilding the trainer per size."""
    data_src = get_data_source(rc)
    EfficiencySweep(
        rc=rc, eff_cfg=eff_cfg, data_src=data_src, weights_path=weights_path
    ).run()


def _parse() -> tuple:
    """Parse RunConfig + EfficiencyConfig; in run_dir mode seed from the archive."""
    EfficiencyConfig = get_efficiency_config_cls()

    run_dir = _peek_run_dir()
    default_config = None
    if run_dir:
        _reject_model_flag(
            sys.argv[1:]
        )  # run_dir mode reconstructs the model from the archive
        archive = Path(run_dir) / "run_config.yaml"
        if not archive.exists():
            raise SystemExit(f"[Benchmark] run_config.yaml not found in {run_dir}")
        default_config = archive

    rc, ns = ConfigParser(
        prog="efficiency.py",
        description="UniKT Model Efficiency Benchmark",
        extra_nodes={"efficiency": EfficiencyConfig},
        default_config=default_config,
    ).parse_with_extras()
    eff_cfg = build_node(EfficiencyConfig, ns["efficiency"])
    return rc, eff_cfg


def _peek_run_dir() -> str | None:
    """Read --efficiency.general.run_dir before ConfigParser (default_config path needs it)."""
    flag = "--efficiency.general.run_dir"
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _reject_model_flag(argv: list[str]) -> None:
    """run_dir mode is incompatible with an explicit model flag."""
    for a in argv:
        if a.startswith("-m") or a.startswith("--experiment.model"):
            raise SystemExit(
                "[Benchmark] --efficiency.general.run_dir cannot be combined with "
                "-m/--experiment.model_name"
            )


def _resolve_weights(eff_cfg) -> str | None:
    general = eff_cfg.general
    if general.weights:
        path = Path(general.weights)
    elif general.run_dir:
        path = Path(general.run_dir) / general.checkpoint
    else:
        return None
    # Fail fast before the expensive model+data build.
    if not path.exists():
        raise SystemExit(f"[Benchmark] checkpoint not found: {path}")
    return str(path)


if __name__ == "__main__":
    main()
