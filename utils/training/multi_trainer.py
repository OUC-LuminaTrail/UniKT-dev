"""Multi-stage trainer module.

Provides sequential multi-stage training on top of :class:`BaseTrainer`:
each stage has its own model / optimizer / loss / data / early stopping
configuration, and stages communicate via hooks.

Example:
    >>> @register_trainer("ABKT")
    ... class ABKTTrainer(MultiTrainer):
    ...     def __init__(self, args, data_src, exp_manager):
    ...         super().__init__(device=args.device)
    ...         self.with_experiment(exp_manager, hyperparams=args,
    ...                               model_name="ABKT").build()
    ...
    ...     def build_stages(self):
    ...         return [StageConfig("km", self._build_km),
    ...                 StageConfig("am", self._build_am)]
    ...
    ...     def _build_km(self):
    ...         return StageComponents(model=..., optimizer=..., loss_fn=...,
    ...                                train_data=..., val_data=..., epochs=...,
    ...                                early_stopping=EarlyStoppingConfig(...))
    ...
    ...     def forward_pass(self, batch_data):
    ...         if self._current_stage == "km":
    ...             return self._forward_km(batch_data)
    ...         return self._forward_am(batch_data)
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ..config import EarlyStopping, EarlyStoppingConfig
from ..core import get_logger
from .base_trainer import BaseTrainer, StageResult
from .callbacks import (
    Callback,
    CallbackManager,
    CheckpointCallback,
    EarlyStoppingCallback,
    FunctionCallback,
    MemoryCleanupCallback,
)
from .checkpoint import CheckpointManager
from .metric_logger import build_default_metric_loggers, resolve_metric_logging_flags
from .metrics import MetricsAccumulator

logger = get_logger(__name__)


@dataclass
class StageComponents:
    """Pre-built components for a single training stage.

    Returned by a stage builder (``StageConfig.build``). Describes
    the model, optimizer, loss, data, epochs, and early stopping
    configuration for one stage.

    Attributes:
        model: PyTorch model for this stage.
        optimizer: Optimizer for this stage.
        loss_fn: Loss function for this stage.
        train_data: Training data (DataLoader or Dataset).
        val_data: Validation data (optional).
        test_data: Test data (optional; multi-stage typically uses the
            final stage's validation as the test proxy).
        epochs: Number of training epochs for this stage.
        lr_scheduler: Learning rate scheduler (optional).
        early_stopping: Early stopping configuration — passes the config
            object, not a constructed instance (optional).
        max_clip_grad_norm: Maximum gradient norm for clipping (optional).
        checkpoint_monitor: Metric to monitor for best model saving
            (optional). Defaults to the early stopping monitor; set
            explicitly to decouple checkpoint saving from early stopping.
        checkpoint_mode: Direction for the checkpoint monitor, ``"max"``
            or ``"min"`` (optional).
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    train_data: Any
    val_data: Any | None = None
    test_data: Any | None = None
    epochs: int = 100
    lr_scheduler: Any | None = None
    early_stopping: EarlyStoppingConfig | None = None
    max_clip_grad_norm: float | None = None
    checkpoint_monitor: str | None = None
    checkpoint_mode: str | None = None


@dataclass
class StageConfig:
    """Declaration of a training stage: name + lazy builder.

    ``build`` is a zero-argument callable invoked just before the stage
    starts training. This allows the builder to depend on data that is
    only determined after the previous stage completes (e.g. boosting
    residuals).

    Attributes:
        name: Stage name (used for log prefixes, checkpoint filenames,
            and metric series).
        build: Zero-argument callable returning :class:`StageComponents`.
    """

    name: str
    build: Callable[[], StageComponents]


class MultiTrainer(BaseTrainer):
    """Multi-stage trainer.

    Subclasses must implement:

    1. :meth:`build_stages`: Return the ordered list of stages.
    2. :meth:`forward_pass`: ``forward_pass(batch_data)``, distinguishing the current stage via ``self._current_stage``.

    Optional overrides:

    3. :meth:`on_stage_begin`: Preparation before a stage (default no-op).
    4. :meth:`on_stage_complete`: Post-stage processing (default no-op), often used to pass data to the next stage.
    5. :meth:`_compute_loss`: Custom loss computation (defaults to ``self.loss(y_hat, y_label)``).

    Construction accepts ``device`` / ``seed`` for device and random
    seed, then uses chained :meth:`with_experiment` and :meth:`build`
    for infrastructure initialization.
    """

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
    ):
        """Initialize the multi-stage trainer.

        Unlike the base trainer, no single model is passed — models
        are created per stage and attached to ``self.model`` at runtime.

        Args:
            device: Compute device (auto-detected if None).
        """
        super().__init__(model=None)

        self._device: str | torch.device | None = device

        # Stage state
        self._stages: list[StageConfig] = []
        self._stage_results: dict[str, StageResult] = {}
        self._elapsed_epochs: int = 0

    # ==================== Build ====================

    def build(self) -> "MultiTrainer":
        """Build the multi-stage trainer infrastructure.

        Unlike :meth:`BaseTrainer.build`, this does **not** configure
        model / data / optimizer (they change per stage). It only
        initializes cross-stage shared facilities (device, seed, log
        directory, metrics, checkpoints).

        Returns:
            Self for method chaining.
        """
        if self._built:
            logger.warning("MultiTrainer already built. Skipping rebuild.")
            return self

        if self._experiment_config is None:
            raise ValueError(
                "Experiment configuration not set. Call with_experiment() first."
            )

        # 1. Device
        self.device_ = (
            torch.device(self._device) if self._device is not None else self._try_gpu()
        )

        # 2. Resolve logging flags
        hyperparams = self._experiment_config.hyperparams
        self.no_swanlab, self.log_batch_metrics = resolve_metric_logging_flags(
            self._experiment_config, hyperparams
        )

        # 3. Log directory
        exp_manager = self._experiment_config.exp_manager
        if exp_manager is None:
            raise ValueError("exp_manager is required.")
        self.log_dir = exp_manager.get_log_dir()
        os.makedirs(self.log_dir, exist_ok=True)

        # 4. Shared components
        self.metrics_accumulator = MetricsAccumulator()
        self.checkpoint_manager = CheckpointManager(self.log_dir)
        self.metric_logger = build_default_metric_loggers(
            log_dir=self.log_dir,
            log_batch_metrics=self.log_batch_metrics,
            no_swanlab=self.no_swanlab,
        )

        # 5. Hyperparameters (model/opt may be None at this point;
        #    _setup_hyperparameters skips them gracefully)
        if hyperparams is not None:
            self._setup_hyperparameters(
                hyperparams,
                model_name=self._experiment_config.model_name,
                dataset_name=self._experiment_config.dataset_name,
            )

        logger.info("MultiTrainer built successfully")
        self._built = True
        return self

    # ==================== Subclass hooks ====================

    def build_stages(self) -> list[StageConfig]:
        """Return the ordered list of stages (must be implemented by subclasses).

        Returns:
            List of StageConfig objects defining the training pipeline.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement build_stages()")

    def on_stage_begin(self, name: str) -> None:
        """Hook called before a stage starts (optional override).

        Args:
            name: Name of the stage about to begin.
        """
        pass

    def on_stage_complete(self, name: str, result: StageResult) -> None:
        """Hook called after a stage completes (optional override).

        Invoked after the best model of the stage has been loaded back
        into ``self.model``. Commonly used to prepare data for the next
        stage (e.g. computing boosting residuals).

        Args:
            name: Name of the stage that just completed.
            result: :class:`StageResult` for the completed stage.
        """
        pass

    # ==================== Run ====================

    def run(self) -> None:
        """Execute all stages sequentially."""
        if not self._built:
            raise RuntimeError(
                "MultiTrainer has not been built. Please call build() explicitly "
                "before run()."
            )

        self._init_metric_logger()

        self._stages = self.build_stages()
        if not self._stages:
            raise ValueError("build_stages() returned no stages.")

        stage_names = [s.name for s in self._stages]
        logger.info(
            f"Starting multi-stage training with {len(self._stages)} stages: "
            f"{stage_names}"
        )

        for stage_idx, stage in enumerate(self._stages):
            logger.info("=" * 60)
            logger.info(
                f"Stage {stage_idx + 1}/{len(self._stages)}: {stage.name.upper()}"
            )
            logger.info("=" * 60)

            self._current_stage = stage.name
            self.on_stage_begin(stage.name)

            # Lazy build: stage components may depend on previous stages
            setup = stage.build()
            self._apply_stage(stage.name, setup)

            result = self._run_training_loop()

            # Load the best model back into self.model for subsequent stages
            checkpoint_cb = self.callback_manager.get_callback(CheckpointCallback)
            if checkpoint_cb is not None and checkpoint_cb.best_model_state is not None:
                self.model.load_state_dict(checkpoint_cb.best_model_state)

            result.name = stage.name
            self._stage_results[stage.name] = result
            self._elapsed_epochs += self.epochs

            self.on_stage_complete(stage.name, result)

        self._current_stage = None
        self._finish()

    def _apply_stage(self, name: str, setup: StageComponents) -> None:
        """Attach stage components to instance attributes and rebuild callbacks.

        Args:
            name: Stage name.
            setup: Pre-built StageComponents for this stage.
        """
        self.model = setup.model
        self.opt = setup.optimizer
        self.loss = setup.loss_fn
        self.train_data = setup.train_data
        self.val_data = setup.val_data
        self.test_data = setup.test_data
        self.epochs = setup.epochs
        self.lr_scheduler = setup.lr_scheduler
        self.max_clip_grad_norm = setup.max_clip_grad_norm
        self.early_stopping = (
            EarlyStopping(setup.early_stopping) if setup.early_stopping else None
        )

        self.start_epoch = 0
        # Metric steps accumulate across stages to keep SwanLab x-axis monotonic
        self._metric_step_offset = self._elapsed_epochs

        self.callback_manager = self._build_stage_callbacks(name, setup)

        logger.info(f"Stage '{name}' setup complete:")
        logger.info(f"  - Model: {type(self.model).__name__}")
        logger.info(f"  - Optimizer: {type(self.opt).__name__}")
        logger.info(f"  - Loss: {type(self.loss).__name__}")
        logger.info(f"  - Epochs: {self.epochs}")
        logger.info(f"  - Train batches: {len(self.train_data)}")
        if self.val_data is not None:
            logger.info(f"  - Val batches: {len(self.val_data)}")

    def _build_stage_callbacks(
        self, name: str, setup: StageComponents
    ) -> CallbackManager:
        """Build the callback list for a single stage, including stage-specific early stopping and checkpointing.

        Args:
            name: Stage name.
            setup: StageComponents for this stage.

        Returns:
            A CallbackManager configured for this stage.
        """
        callbacks: list[Callback] = []
        callbacks.extend(self._custom_callbacks)
        if self._custom_callback_functions:
            callbacks.append(FunctionCallback(self._custom_callback_functions))
        callbacks.append(MemoryCleanupCallback(cleanup_interval=5))
        if self.early_stopping is not None:
            callbacks.append(
                EarlyStoppingCallback(early_stopping=self.early_stopping, stage=name)
            )
        callbacks.append(
            CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                early_stopping=self.early_stopping,
                last_filename=f"{name}_last_checkpoint.pth",
                best_filename=(
                    f"best_{name}_model.pth"
                    if self.early_stopping is not None
                    else None
                ),
                keep_best_state=True,
                monitor=setup.checkpoint_monitor,
                mode=setup.checkpoint_mode,
            )
        )
        return CallbackManager(callbacks)

    # ==================== Metric logging / Cleanup ====================

    def _init_metric_logger(self) -> None:
        """Initialize the metric logging backend.

        Local CSV logging is always enabled; SwanLab is included unless
        ``--no_swanlab`` was set.
        """
        experiment_name = os.path.basename(self.log_dir) if self.log_dir else "run"
        config = (
            self.hyperparam_manager.get_hyperparameters_dict()
            if self.hyperparam_manager is not None
            else {}
        )
        group = type(self).__name__.replace("Trainer", "")
        self.metric_logger.init_run(
            log_dir=self.log_dir,
            experiment_name=experiment_name,
            group=group,
            tags=["cuda" if torch.cuda.is_available() else "cpu", "multi-stage"],
            config=config,
        )

    def _finish(self) -> None:
        """Summarize stage results and finalize metric logging.

        Logs the best metric per stage, records final metrics, and
        shuts down the metric logger and checkpoint manager.
        """
        logger.info("=" * 60)
        logger.info("Multi-stage training complete")
        for name, result in self._stage_results.items():
            if result.best_metric is not None:
                best_epoch_str = (
                    result.best_epoch + 1 if result.best_epoch is not None else "N/A"
                )
                logger.info(
                    f"  {name.upper()}: Best {result.monitor.upper()} = "
                    f"{result.best_metric:.4f} (Epoch {best_epoch_str})"
                )
        logger.info("=" * 60)

        final_metrics = {
            f"Final/{name}_best": result.best_metric
            for name, result in self._stage_results.items()
            if result.best_metric is not None
        }
        if final_metrics:
            self.metric_logger.log_final(metrics=final_metrics, step=self._global_step)

        self.metric_logger.finish()
        logger.info("Metric logging finished")
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.close()


__all__ = ["MultiTrainer", "StageComponents", "StageConfig"]
