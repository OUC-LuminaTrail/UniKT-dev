"""Batch ablation experiment runner.

Uses existing trainer infrastructure to run multiple ablations sequentially.
"""

from datetime import datetime
from pathlib import Path

from utils.ablation.config import AblationStudyConfig
from utils.core import TRAINERS, get_logger, get_supported_models, seed_everything

logger = get_logger(__name__)


class AblationRunner:
    """Run ablation experiments using existing trainers.

    Example:
        >>> from utils.ablation import load_config, AblationRunner
        >>> config = load_config("configs/ablation/hgikt_study.yaml")
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
        from dataclasses import fields as dc_fields

        from omegaconf import OmegaConf

        from utils.config import build_run_config_schema
        from utils.data_process import get_data_source
        from utils.experiment_manager import ExperimentManager, ExperimentType

        # Build a RunConfig for this variant: the model's structured schema
        # defaults overlaid with the ablation's flat params, routed to the
        # correct node (model/data/general/...) by field-name lookup.
        schema = build_run_config_schema(ablation.variant)
        nested: dict = {}
        matched: set[str] = set()
        for node, cls in schema.items():
            for f in dc_fields(cls):
                if f.name in params:
                    nested.setdefault(node, {})[f.name] = params[f.name]
                    matched.add(f.name)
        unmatched = sorted(set(params) - matched)
        if unmatched:
            logger.warning(
                "Ablation '%s' params matched no %s config field (ignored): %s",
                ablation.name,
                ablation.variant,
                unmatched,
            )
        rc = OmegaConf.merge(OmegaConf.structured(schema), OmegaConf.create(nested))
        rc.experiment.model_name = ablation.variant
        rc.data.dataset = self.config.dataset

        # Reseed before constructing the trainer for reproducible weight init.
        seed_everything(rc.general.seed, deterministic=not rc.general.no_deterministic)

        # Create experiment manager with ABLATION type.
        # Use subdirectory within the top-level ablation study directory.
        exp_manager = ExperimentManager(
            exp_type=ExperimentType.ABLATION,
            model_name=ablation.variant,
            dataset_name=self.config.dataset,
            base_dir=str(self.exp_base_dir),
            tags=[f"fold{rc.data.fold}", f"bs{getattr(rc.model, 'batch_size', 64)}"],
        )

        # Get data source and run trainer
        data_src = get_data_source(rc)
        trainer = trainer_cls(rc=rc, data_src=data_src, exp_manager=exp_manager)
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
