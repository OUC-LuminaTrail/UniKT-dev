"""UniKT 模型效率基准评估脚本。

两种互斥模式：
    - ``-m/-d``        全新构造模型（随机权重，或 ``--weights`` 指定文件）+ 数据
    - ``--run_dir``    从已训 run 重建参数并加载 best_model.pth

用法:
    python efficiency.py -m GIKT -d assistments09 --fold 0
    python efficiency.py -m SAKT -d assistments09 --weights runs/.../best_model.pth
    python efficiency.py --run_dir runs/efficiency/GIKT_assist09_..._fold0_bs128
    python efficiency.py -m AKT -d assistments09 --modes inference --benchmark_iters 500
"""

import argparse
import sys
from pathlib import Path

import model  # noqa: F401
from utils.config import (
    CompileParams,
    DataParams,
    EarlyStoppingParams,
    GeneralParams,
    get_model_params,
    list_models,
)
from utils.core import TRAINERS, get_logger
from utils.data_process import get_data_source
from utils.efficiency import EfficiencyParams, EfficiencySession
from utils.experiment_manager import ExperimentManager, ExperimentType

logger = get_logger(__name__)

# 模式 B 下 load_model_params 重建的 Namespace 不含这些字段（efficiency 专属 + 部分 General）；
# 需从 CLI args 继承（含用户覆盖）到 model_args。
_INHERITED_KEYS = (
    "modes",
    "benchmark_iters",
    "warmup_iters",
    "repeats",
    "train_iters",
    "sample_interval",
    "profile_flops",
    "output_dir",
    "checkpoint",
    "weights",
    "seed",
    "deterministic",
    "device",
)


def parse_args() -> argparse.Namespace:
    """解析 CLI：``-m/-d``（全新构造）或 ``--run_dir``（加载已训），二者互斥。"""
    temp = _preparse()
    parser = _build_parser(temp)
    args = parser.parse_args()
    args.run_dir = temp.run_dir
    _validate_mode(args, parser)
    return args


def _preparse() -> argparse.Namespace:
    """预解析 -m / -d / --run_dir，供后续参数注册策略决策。"""
    temp = argparse.ArgumentParser(add_help=False)
    temp.add_argument("-m", "--model", type=str)
    temp.add_argument("-d", "--dataset", type=str)
    temp.add_argument("--run_dir", type=str)
    temp_args, _ = temp.parse_known_args()
    return temp_args


def _build_parser(temp_args: argparse.Namespace) -> argparse.ArgumentParser:
    """构建完整 argparse：通用参数组 + efficiency 参数 + 模型特定参数。"""
    parser = argparse.ArgumentParser(
        description="UniKT Model Efficiency Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    GeneralParams.add_args(parser)
    DataParams.add_args(parser)
    # 模式 B：dataset/model 由 load_model_params 从 hyperparameters.json 重建，解除 -d required
    if temp_args.run_dir:
        for action in parser._actions:
            if action.dest == "dataset":
                action.required = False
    EarlyStoppingParams.add_args(parser)
    CompileParams.add_args(parser)
    EfficiencyParams.add_args(parser)

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        choices=list(TRAINERS.keys()),
        help=f"Model to benchmark. Available: {', '.join(list_models())}",
    )

    # 模式 A 才注册模型特定参数；模式 B 由 load_model_params 重建，跳过
    if temp_args.model:
        model_params = get_model_params(temp_args.model)
        if model_params is None:
            raise SystemExit(
                f"Unknown model: {temp_args.model}. Available: {', '.join(list_models())}"
            )
        model_params.add_args(parser)
    return parser


def _validate_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """``-m`` 与 ``--run_dir`` 互斥，且必须提供其一。"""
    if args.run_dir and args.model:
        parser.error("--run_dir cannot be combined with -m/--model")
    if not args.run_dir and not args.model:
        parser.error("Either -m/--model or --run_dir is required")


def _resolve_args(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, str, str, str | None]:
    """统一模式 A/B 的 args，返回 ``(args, model_name, dataset_name, load_weights_path)``。

    两种模式都关闭 trainer 构造副作用（``no_swanlab``/``checkpoint_path=None``/``skip_test``），
    差异仅在 ``model_name``/``dataset_name``/``load_weights_path`` 的来源。
    """
    if args.run_dir:
        return _resolve_run_dir_mode(args)
    return _resolve_fresh_mode(args)


def _resolve_fresh_mode(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, str, str, str | None]:
    """模式 A：``-m/-d`` 全新构造（随机权重，或 ``--weights`` 指定文件）。"""
    args.no_swanlab = True
    args.checkpoint_path = None
    args.skip_test = True
    return args, args.model, args.dataset, args.weights


def _resolve_run_dir_mode(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, str, str, str]:
    """模式 B：``--run_dir`` 从已训 run 重建 args 并加载 best_model.pth。"""
    from case_analysis import load_model_params

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / args.checkpoint
    hyperparams_path = run_dir / "hyperparameters.json"
    if not ckpt_path.exists():
        logger.error(f"[Benchmark] checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not hyperparams_path.exists():
        logger.error(f"[Benchmark] hyperparameters.json not found: {hyperparams_path}")
        sys.exit(1)

    model_args, model_name, dataset_name = load_model_params(
        checkpoint_path=str(ckpt_path),
        hyperparams_path=str(hyperparams_path),
    )
    # 继承 efficiency 专属字段 + cli 覆盖项（重建的 Namespace 不含这些）
    for key in _INHERITED_KEYS:
        if hasattr(args, key):
            setattr(model_args, key, getattr(args, key))
    # report 读 args.model/dataset，而 load_model_params 把它们作为独立 tuple 返回
    model_args.model = model_name
    model_args.dataset = dataset_name
    model_args.no_swanlab = True
    model_args.checkpoint_path = None
    model_args.skip_test = True
    return model_args, model_name, dataset_name, str(ckpt_path)


def main() -> None:
    args = parse_args()
    args, model_name, dataset_name, load_weights_path = _resolve_args(args)

    exp_manager = ExperimentManager.from_args(args, ExperimentType.EFFICIENCY)
    output_dir = args.output_dir or exp_manager.get_log_dir()
    logger.info(f"[Benchmark] model={model_name} dataset={dataset_name}")
    logger.info(f"[Benchmark] output_dir={output_dir}")

    data_src = get_data_source(dataset_name=dataset_name, args=args)
    trainer_cls = TRAINERS.get(model_name)
    trainer = trainer_cls(args=args, data_src=data_src, exp_manager=exp_manager)

    if load_weights_path:
        logger.info(f"[Benchmark] loading weights: {load_weights_path}")
        trainer.load_weights(load_weights_path)

    EfficiencySession(
        trainer=trainer, args=args, output_dir=output_dir
    ).run().print_console()


if __name__ == "__main__":
    main()
