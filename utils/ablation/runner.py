"""Batch ablation experiment runner.

Uses existing trainer infrastructure to run multiple ablations sequentially.
"""

from argparse import Namespace
from datetime import datetime
from pathlib import Path

from utils.ablation.config import AblationStudyConfig
from utils.core import TRAINERS, get_logger, get_supported_models, seed_everything

logger = get_logger(__name__)


class AblationRunner:
    """Run ablation experiments using existing trainers.

    Example:
        >>> from utils.ablation import load_config, AblationRunner
        >>> config = load_config("configs/ablation/hgikt_study.json")
        >>> runner = AblationRunner(config)
        >>> results = runner.run_all()
    """

    def __init__(self, config: AblationStudyConfig):
        """Initialize runner with ablation study config.

        Args:
            config: AblationStudyConfig instance
        """
        self.config = config

        # Create top-level experiment directory for this ablation study run
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        study_name = getattr(config, "study_name", "ablation_study")
        base_dir = Path("runs") / "ablation" / f"{study_name}_{timestamp}"
        base_dir.mkdir(parents=True, exist_ok=True)

        self.exp_base_dir = base_dir
        logger.info(f"Ablation study directory: {base_dir}")

    def run_all(self):
        """Run all ablations in the config.

        Returns:
            Tuple of (results list, base_dir path)
        """
        results = []

        for ablation in self.config.ablations:
            logger.info(f"{'=' * 60}")
            logger.info(f"Running: {ablation.name}")
            logger.info(f"Variant: {ablation.variant}")
            if ablation.description:
                logger.info(f"Description: {ablation.description}")
            logger.info(f"{'=' * 60}")

            # Merge shared params with ablation-specific params
            params = {**self.config.shared_params, **ablation.params}

            # Get trainer for this variant
            if ablation.variant not in TRAINERS:
                raise ValueError(
                    f"Variant '{ablation.variant}' not registered in TRAINERS. "
                    f"Available: {', '.join(get_supported_models())}"
                )
            trainer_cls = TRAINERS.get(ablation.variant)

            # Run experiment
            result = self._run_single(ablation, trainer_cls, params)
            results.append(result)

        return results, self.exp_base_dir

    def _run_single(self, ablation, trainer_cls, params):
        """Run a single ablation experiment.

        Args:
            ablation: AblationConfig instance
            trainer_cls: Trainer class to use
            params: Merged parameters (shared + ablation-specific)

        Returns:
            Result dictionary with name, variant, and metrics
        """
        from utils.data_process import get_data_source
        from utils.experiment_manager import ExperimentManager, ExperimentType

        # Create args namespace
        args = Namespace(**params)
        args.dataset = self.config.dataset
        args.model = ablation.variant  # Use variant as model name
        args.ablation_name = ablation.name

        # Reseed before constructing the trainer for reproducible weight init.
        seed_everything(
            getattr(args, "seed", 42),
            deterministic=not getattr(args, "no_deterministic", False),
        )

        # Create experiment manager with ABLATION type
        # Use subdirectory within the top-level ablation study directory
        exp_manager = ExperimentManager(
            exp_type=ExperimentType.ABLATION,
            model_name=ablation.variant,
            dataset_name=self.config.dataset,
            base_dir=str(self.exp_base_dir),
            tags=[f"fold{params.get('fold', 0)}", f"bs{params.get('batch_size', 64)}"],
        )

        # Get data source
        data_src = get_data_source(dataset_name=self.config.dataset, args=args)

        # Create and run trainer
        trainer = trainer_cls(args=args, data_src=data_src, exp_manager=exp_manager)
        trainer.run()

        # Get final validation metrics from trainer's metrics accumulator
        metrics = (
            trainer.metrics_accumulator.compute("val")
            if trainer.val_data is not None
            else {}
        )

        return {
            "name": ablation.name,
            "variant": ablation.variant,
            "metrics": metrics,
        }
