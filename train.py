"""Unified knowledge tracing training script."""

import argparse

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
from utils.experiment_manager import ExperimentManager, ExperimentType

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments, dynamically adding model-specific params."""
    # Pre-parse model name to dynamically add model-specific arguments
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("-m", "--model", type=str)
    temp_args, _ = temp_parser.parse_known_args()

    model_name = temp_args.model

    # Build the full argument parser
    parser = argparse.ArgumentParser(description="Knowledge Tracing Training Script")

    # Add common parameters
    GeneralParams.add_args(parser)
    DataParams.add_args(parser)
    EarlyStoppingParams.add_args(parser)
    CompileParams.add_args(parser)

    # Add model selection parameter
    available_models = list(TRAINERS.keys())
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=available_models,
        help=f"Model to train. Available: {', '.join(available_models)}",
    )

    # Add model-specific parameters
    if model_name:
        model_params_cls = get_model_params(model_name)
        if model_params_cls:
            model_params_cls.add_args(parser)
        else:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {', '.join(list_models())}"
            )

    args = parser.parse_args()
    return args


def main():
    """Train a knowledge tracing model."""
    args = parse_args()

    # Create experiment manager
    exp_manager = ExperimentManager.from_args(args, ExperimentType.NORMAL)
    logger.info(f"Experiment directory: {exp_manager.get_log_dir()}")

    logger.info(f"Building dataset: {args.dataset}...")
    data_src = get_data_source(dataset_name=args.dataset, args=args)

    logger.info(f"Initializing trainer for model: {args.model}...")
    trainer_cls = TRAINERS.get(args.model)
    trainer = trainer_cls(
        args=args,
        data_src=data_src,
        exp_manager=exp_manager,
    )

    logger.info("Starting training...")
    trainer.run()


if __name__ == "__main__":
    main()
