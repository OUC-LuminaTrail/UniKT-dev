#!/usr/bin/env python3
"""KT 模型的消融研究运行器。

Usage:
    python ablation_study.py --model GIKT --dataset assistments09 --config configs/ablation/gikt_ablation.json
"""

import argparse
import sys
from utils.ablation import AblationExperiment, load_ablation_config
from utils.core import TRAINERS, get_logger
from utils.data_process import get_data_source
from utils.config import (
    DataParams,
    EarlyStoppingParams,
    GeneralParams,
    get_model_params,
)
import model  # noqa: F401

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数。"""
    # 预解析 config 参数以获取模型名称
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--config", type=str)
    temp_args, _ = temp_parser.parse_known_args()

    model_name = None
    if temp_args.config:
        try:
            import json
            from pathlib import Path

            config_path = Path(temp_args.config)
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    model_name = config_data.get("model_name")
        except Exception:
            pass

    # 构建完整的解析器
    parser = argparse.ArgumentParser(description="Ablation Study Runner for KT Models")

    # 添加通用参数
    DataParams.add_args(parser)
    EarlyStoppingParams.add_args(parser)
    GeneralParams.add_args(parser)

    # 添加模型选择参数
    if model_name:
        model_params_cls = get_model_params(model_name)
        if model_params_cls:
            model_params_cls.add_args(parser)

    # 添加消融研究特定参数
    ablation_group = parser.add_argument_group("Ablation Parameters")
    ablation_group.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ablation configuration file",
    )
    ablation_group.add_argument(
        "--ablations",
        type=str,
        default=None,
        help="Comma-separated list of ablation names to run (default: all)",
    )
    ablation_group.add_argument(
        "--output-dir",
        type=str,
        default="runs/ablation",
        help="Directory to save ablation results (default: runs/ablation)",
    )

    args = parser.parse_args()

    return args


def main():
    """主入口。"""
    args = parse_args()

    logger.info("Ablation Study Runner")

    # Load ablation configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_ablation_config(args.config)

    logger.info(f"Model: {config.model_name}")
    logger.info(f"Baseline: {config.baseline.name}")
    logger.info(f"Ablations: {len(config.ablations)}")

    # Filter ablations if specified
    if args.ablations:
        ablation_names = [name.strip() for name in args.ablations.split(",")]
        config.ablations = [ab for ab in config.ablations if ab.name in ablation_names]
        logger.info(f"Running selected ablations: {ablation_names}")

    # Get trainer class
    if config.model_name not in TRAINERS:
        available = ", ".join(TRAINERS.keys())
        raise ValueError(
            f"Model '{config.model_name}' not found. Available: {available}"
        )

    trainer_cls = TRAINERS.get(config.model_name)

    # Build dataset
    logger.info(f"Building dataset: {args.dataset}...")
    data_src = get_data_source(dataset_name=args.dataset, args=args)

    # Create experiment
    experiment = AblationExperiment(
        base_trainer=trainer_cls,
        config=config,
        args=args,
        data_src=data_src,
        output_dir=args.output_dir,
    )

    # Run experiments
    logger.info("Starting Ablation Study")

    try:
        experiment.run_all()
        logger.info("Ablation Study Completed Successfully!")

        return 0

    except Exception as e:
        logger.error(f"Error during ablation study: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
