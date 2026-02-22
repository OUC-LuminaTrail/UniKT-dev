"""Batch ablation experiment runner.

Uses existing trainer infrastructure to run multiple ablations sequentially.
"""

from argparse import Namespace

from utils.ablation.config import AblationStudyConfig
from utils.core import TRAINERS, get_logger

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

    def run_all(self):
        """Run all ablations in the config.

        Returns:
            List of result dictionaries, one per ablation
        """
        results = []

        for ablation in self.config.ablations:
            print(f"\n{'=' * 60}")
            print(f"Running: {ablation.name}")
            print(f"Variant: {ablation.variant}")
            if ablation.description:
                print(f"Description: {ablation.description}")
            print(f"{'=' * 60}\n")

            # Merge shared params with ablation-specific params
            params = {**self.config.shared_params, **ablation.params}

            # Get trainer for this variant
            if ablation.variant not in TRAINERS:
                raise ValueError(
                    f"Variant '{ablation.variant}' not registered in TRAINERS. "
                    f"Available: {', '.join(TRAINERS.keys())}"
                )
            trainer_cls = TRAINERS.get(ablation.variant)

            # Run experiment
            result = self._run_single(ablation, trainer_cls, params)
            results.append(result)

        return results

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

        # Create experiment manager with ABLATION type
        exp_manager = ExperimentManager.from_args(args, ExperimentType.ABLATION)

        # Get data source
        data_src = get_data_source(dataset_name=self.config.dataset, args=args)

        # Create and run trainer
        trainer = trainer_cls(args=args, data_src=data_src, exp_manager=exp_manager)
        metrics = trainer.run()

        return {
            "name": ablation.name,
            "variant": ablation.variant,
            "metrics": metrics,
        }
