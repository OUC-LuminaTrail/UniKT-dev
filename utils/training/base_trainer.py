"""Base trainer module.

Provides the core trainer functionality including device management,
data loading, training loops, callbacks, checkpointing, and metric
logging.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from rich.console import Group
from rich.live import Live
from rich.text import Text

from ..config import (
    DataConfig,
    EarlyStopping,
    EarlyStoppingConfig,
    ExperimentConfig,
    OptimizationConfig,
    TrainingConfig,
    create_optimized_dataloader,
)
from ..core import get_logger
from ..progress import create_progress
from .callbacks import (
    Callback,
    CallbackManager,
    CheckpointCallback,
    EarlyStoppingCallback,
    FunctionCallback,
    MemoryCleanupCallback,
    TestEvaluationCallback,
)
from .checkpoint import CheckpointManager
from .metric_logger import build_default_metric_loggers, resolve_metric_logging_flags
from .metrics import MetricsAccumulator

logger = get_logger(__name__)


@dataclass
class StageResult:
    """Result of a single training stage.

    Attributes:
        name: Stage name (``None`` for single-stage training).
        best_metric: Best monitored metric value on the validation set.
        best_epoch: Epoch (0-indexed) at which the best metric was achieved.
        final_epoch: Last epoch actually trained in this stage.
        monitor: Name of the monitored metric (e.g. ``"auc"``, ``"acc"``,
            ``"rmse"``).
    """

    name: str | None = None
    best_metric: float | None = None
    best_epoch: int | None = None
    final_epoch: int | None = None
    monitor: str = "auc"


class BaseTrainer(ABC):
    r"""Abstract base class for trainers.

    Subclasses must implement:
    1. ``__init__``: Directly initialize the model.
    2. ``forward_pass``: Model forward pass logic.

    Usage::

        trainer = MyTrainer(model)
            .with_training(epochs=150, seed=42)
            .with_data(train_dataset, val_dataset, batch_size=128)
            .with_optimization(optimizer, loss_fn, lr_scheduler)
            .with_experiment(exp_manager, hyperparams=args)
            .build()
        trainer.run()
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize the base trainer.

        Args:
            model: PyTorch model to train.
        """
        self.model = model

        # Configuration objects
        self._training_config: TrainingConfig | None = None
        self._data_config: DataConfig | None = None
        self._optimization_config: OptimizationConfig | None = None
        self._experiment_config: ExperimentConfig | None = None

        # Internal state
        self._built = False
        self._compile_config: dict | None = None
        self.device_: torch.device | None = None
        self.epochs: int | None = None
        self.train_data = None
        self.val_data = None
        self.opt = None
        self.max_clip_grad_norm: float | None = None
        self.loss = None
        self.lr_scheduler = None
        self.early_stopping: EarlyStopping | None = None
        self.start_epoch = 0
        self.log_dir = None
        self.metrics_accumulator = None
        self.checkpoint_manager = None
        self.callback_manager = None
        self.hyperparam_manager = None
        self.no_swanlab = False
        self.log_batch_metrics = False
        self.metric_logger = None
        self._global_step = 0
        self._custom_callbacks: list[Callback] = []
        self._custom_callback_functions: dict[str, list[Callable]] = {}

        # Multi-stage context (None/0 for single-stage)
        self._current_stage: str | None = None
        self._metric_step_offset: int = 0

    def with_training(
        self,
        epochs: int = 200,
        seed: int = 42,
        device: torch.device | None = None,
        checkpoint_path: str | None = None,
    ) -> "BaseTrainer":
        """Configure training parameters.

        Args:
            epochs: Number of training epochs.
            seed: Random seed.
            device: Compute device (auto-detected if None).
            checkpoint_path: Path to checkpoint for resuming training.

        Returns:
            Self for method chaining.
        """
        self._training_config = TrainingConfig(
            epochs=epochs,
            seed=seed,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        return self

    def with_compile(
        self,
        mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool | None = None,
        backend: str = "inductor",
    ) -> "BaseTrainer":
        """Configure ``torch.compile`` optimization.

        Args:
            mode: Compilation mode (``"default"``, ``"reduce-overhead"``,
                ``"max-autotune"``, ``"max-autotune-no-cudagraphs"``).
            fullgraph: Whether to require a single computational graph.
            dynamic: Dynamic shape tracing. None = auto, True = force,
                False = static.
            backend: Compilation backend.

        Returns:
            Self for method chaining.
        """
        self._compile_config = {
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "backend": backend,
        }
        return self

    def with_callbacks(
        self,
        callbacks: list[Callback] | None = None,
        functions: dict[str, Callable | list[Callable]] | None = None,
    ) -> "BaseTrainer":
        """Configure custom callbacks.

        Args:
            callbacks: List of callback objects (optional).
            functions: Dict mapping event names to callables or lists
                of callables (optional).

        Returns:
            Self for method chaining.
        """
        if callbacks:
            self._custom_callbacks.extend(callbacks)
        if functions:
            for name, funcs in functions.items():
                if funcs is None:
                    continue
                items = funcs if isinstance(funcs, list) else [funcs]
                self._custom_callback_functions.setdefault(name, []).extend(items)
        return self

    def register_callback(self, callback: Callback) -> None:
        """Register a single callback object.

        Args:
            callback: Callback instance to register.
        """
        self._custom_callbacks.append(callback)

    def register_callback_fn(self, event: str, func: Callable) -> None:
        """Register a single callback function for a named event.

        Args:
            event: Event name (e.g. ``"on_epoch_end"``).
            func: Callable to invoke on the event.
        """
        self._custom_callback_functions.setdefault(event, []).append(func)

    def with_data(
        self,
        train_data,
        batch_size,
        val_data,
        test_data=None,
        collate_fn=None,
        val_collate_fn=None,
        test_collate_fn=None,
    ) -> "BaseTrainer":
        """Configure data loaders.

        Args:
            train_data: Training data (DataLoader or Dataset).
            batch_size: Batch size.
            val_data: Validation data (DataLoader or Dataset).
            test_data: Test data (DataLoader or Dataset, optional).
            collate_fn: Custom collate function (optional).
            val_collate_fn: Custom validation collate function (optional).
            test_collate_fn: Custom test collate function (optional).

        Returns:
            Self for method chaining.
        """
        self._data_config = DataConfig(
            train_data=train_data,
            batch_size=batch_size,
            val_data=val_data,
            test_data=test_data,
            collate_fn=collate_fn,
            val_collate_fn=collate_fn if val_collate_fn is None else val_collate_fn,
            test_collate_fn=collate_fn if test_collate_fn is None else test_collate_fn,
        )
        return self

    def with_optimization(
        self,
        optimizer,
        loss_fn,
        max_clip_grad_norm: float | None = None,
        lr_scheduler=None,
        early_stopping: EarlyStoppingConfig | None = None,
    ) -> "BaseTrainer":
        """Configure optimizer, loss function, and scheduler.

        Args:
            optimizer: PyTorch optimizer.
            loss_fn: Loss function.
            max_clip_grad_norm: Maximum gradient norm for clipping (optional).
            lr_scheduler: Learning rate scheduler (optional).
            early_stopping: Early stopping configuration (optional).

        Returns:
            Self for method chaining.
        """
        self._optimization_config = OptimizationConfig(
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_clip_grad_norm=max_clip_grad_norm,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping,
        )
        return self

    def with_experiment(
        self,
        exp_manager,
        hyperparams=None,
        no_swanlab: bool | None = None,
        log_batch_metrics: bool | None = None,
        model_name: str = "",
        dataset_name: str = "",
        skip_test: bool = False,
    ) -> "BaseTrainer":
        """Configure experiment management and tracking.

        Args:
            exp_manager: Experiment manager instance.
            hyperparams: Hyperparameters (dict or namespace, optional).
            no_swanlab: Disable SwanLab (None = read from hyperparams).
            log_batch_metrics: Log per-batch loss (None = read from hyperparams).
            model_name: Model name.
            dataset_name: Dataset name.
            skip_test: Skip test set evaluation after training.

        Returns:
            Self for method chaining.
        """
        self._experiment_config = ExperimentConfig(
            exp_manager=exp_manager,
            hyperparams=hyperparams,
            no_swanlab=bool(no_swanlab),
            log_batch_metrics=bool(log_batch_metrics),
            model_name=model_name,
            dataset_name=dataset_name,
        )
        self.skip_test = skip_test
        return self

    def build(self) -> "BaseTrainer":
        """Build the trainer, initializing all components.

        Validates configurations, sets up device, data loaders,
        optimization, early stopping, callbacks, logging, and
        hyperparameters.

        Returns:
            Self for method chaining.
        """
        if self._built:
            logger.warning("Trainer already built. Skipping rebuild.")
            return self

        # Validate required configurations
        if self._training_config is None:
            raise ValueError(
                "Training configuration not set. Call with_training() first."
            )
        if self._data_config is None:
            raise ValueError("Data configuration not set. Call with_data() first.")
        if self._optimization_config is None:
            raise ValueError(
                "Optimization configuration not set. Call with_optimization() first."
            )
        if self._experiment_config is None:
            raise ValueError(
                "Experiment configuration not set. Call with_experiment() first."
            )

        # 1. Setup device
        if self._training_config.device is None:
            self.device_ = self._try_gpu()
        else:
            self.device_ = torch.device(self._training_config.device)

        # 2. Setup training parameters
        self.epochs = self._training_config.epochs

        # Resolve no_swanlab / log_batch_metrics
        hyperparams = self._experiment_config.hyperparams
        self.no_swanlab, self.log_batch_metrics = resolve_metric_logging_flags(
            self._experiment_config, hyperparams
        )

        # 3. Setup data loaders
        self._setup_data_loaders()

        # 4. Setup optimization
        self.opt = self._optimization_config.optimizer
        self.loss = self._optimization_config.loss_fn
        self.max_clip_grad_norm = self._optimization_config.max_clip_grad_norm
        self.lr_scheduler = self._optimization_config.lr_scheduler

        # 5. Setup early stopping
        self._setup_early_stopping()

        # 6. Create log directory
        exp_manager = self._experiment_config.exp_manager
        if exp_manager is None:
            raise ValueError("exp_manager is required.")
        self.log_dir = exp_manager.get_log_dir()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 7. Initialize components
        self.metrics_accumulator = MetricsAccumulator()
        self.checkpoint_manager = CheckpointManager(self.log_dir)
        self.metric_logger = build_default_metric_loggers(
            log_dir=self.log_dir,
            log_batch_metrics=self.log_batch_metrics,
            no_swanlab=self.no_swanlab,
        )

        # 8. Initialize callbacks
        callbacks: list[Callback] = []
        callbacks.extend(self._custom_callbacks)
        if self._custom_callback_functions:
            callbacks.append(FunctionCallback(self._custom_callback_functions))
        callbacks.append(MemoryCleanupCallback(cleanup_interval=5))
        if self.early_stopping is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping=self.early_stopping,
                    stage=None,
                )
            )
        callbacks.append(
            CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                early_stopping=self.early_stopping,
                last_filename="last_checkpoint.pth",
                best_filename=(
                    "best_model.pth" if self.early_stopping is not None else None
                ),
            )
        )
        if not getattr(self, "skip_test", False):
            callbacks.append(TestEvaluationCallback(use_best_model=True))
            if self.test_data is None:
                logger.warning(
                    "Test data was not provided during trainer initialization. "
                    "Test evaluation will be skipped. "
                    "Cause: The model's trainer was initialized without passing "
                    "test_data to with_data(). "
                    "Fix: Ensure to pass test_data to with_data(), or use "
                    "'--skip_test' to skip test evaluation explicitly."
                )
            else:
                test_len = (
                    len(self.test_data) if hasattr(self.test_data, "__len__") else None
                )
                if test_len is not None and test_len == 0:
                    logger.warning(
                        "Test set is empty (0 samples). Test evaluation will "
                        "produce no results. "
                        "Cause: No users were assigned to the test fold (fold=-1) "
                        "during data preprocessing. "
                        "This happens when: "
                        "(1) add_kfold_labels() was called with test_ratio=0, or "
                        "(2) test_ratio > 0 but int(num_users * test_ratio) == 0 "
                        "due to small dataset size. "
                        "Fix: Re-run data preprocessing with a larger test_ratio "
                        "value, or use '--skip_test' to skip test evaluation "
                        "explicitly."
                    )
        else:
            logger.info("Test evaluation will be skipped.")
        self.callback_manager = CallbackManager(callbacks)

        # 10. Setup hyperparameters
        if hyperparams is not None:
            self._setup_hyperparameters(
                hyperparams,
                model_name=self._experiment_config.model_name,
                dataset_name=self._experiment_config.dataset_name,
            )

        # 11. Load checkpoint if provided
        if self._training_config.checkpoint_path:
            self._load_checkpoint(self._training_config.checkpoint_path)

        # 12. Apply torch.compile if configured
        self._apply_compile()

        self._built = True
        logger.info("Trainer built successfully")
        return self

    def _setup_data_loaders(self):
        """Set up data loaders for train, validation, and test sets.

        Converts Dataset instances to optimized DataLoaders; passes
        DataLoader instances through unchanged.
        """
        train_data = self._data_config.train_data
        val_data = self._data_config.val_data
        test_data = self._data_config.test_data
        collate_fn = self._data_config.collate_fn
        val_collate_fn = self._data_config.val_collate_fn
        test_collate_fn = self._data_config.test_collate_fn
        batch_size = self._data_config.batch_size

        def _build_loader(data, shuffle, loader_collate_fn):
            if isinstance(data, torch.utils.data.Dataset):
                return create_optimized_dataloader(
                    data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    device=self.device_,
                    collate_fn=loader_collate_fn,
                )
            return data

        self.train_data = _build_loader(train_data, True, collate_fn)
        self.val_data = _build_loader(val_data, False, val_collate_fn)
        self.test_data = _build_loader(test_data, False, test_collate_fn)

    def _setup_early_stopping(self):
        """Set up early stopping from the optimization configuration."""
        early_stopping_cfg = self._optimization_config.early_stopping
        self.early_stopping = (
            EarlyStopping(early_stopping_cfg)
            if early_stopping_cfg is not None
            else None
        )

    def _apply_compile(self):
        """Apply ``torch.compile`` to the model if configured.

        Supports two configuration paths:
        1. Explicit via ``with_compile()`` chained call.
        2. Automatic via ``hyperparams.compile`` flag.

        When both are set, explicit configuration takes precedence.
        """
        if self._compile_config is None:
            hyperparams = None
            if self._experiment_config is not None:
                hyperparams = self._experiment_config.hyperparams
            if hyperparams is not None and getattr(hyperparams, "compile", False):
                self._compile_config = {
                    "mode": getattr(hyperparams, "compile_mode", "default"),
                    "fullgraph": getattr(hyperparams, "compile_fullgraph", False),
                    "dynamic": getattr(hyperparams, "compile_dynamic", None),
                    "backend": getattr(hyperparams, "compile_backend", "inductor"),
                }

        if self._compile_config is None:
            return

        logger.info(
            f"Applying torch.compile: mode={self._compile_config['mode']}, "
            f"backend={self._compile_config['backend']}, "
            f"fullgraph={self._compile_config['fullgraph']}, "
            f"dynamic={self._compile_config['dynamic']}"
        )
        self.model = torch.compile(
            self.model,
            mode=self._compile_config["mode"],
            fullgraph=self._compile_config["fullgraph"],
            dynamic=self._compile_config["dynamic"],
            backend=self._compile_config["backend"],
        )
        logger.info("torch.compile applied successfully")

    @abstractmethod
    def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        """Perform a forward pass for a single batch.

        Args:
            batch_data: A batch of data from the DataLoader.

        Returns:
            Dict containing at least ``"y_hat"``, ``"y_label"``,
            ``"y_predict"``.
        """
        raise NotImplementedError("Subclasses must implement forward_pass method")

    def test_forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        """Perform a forward pass for test data.

        Defaults to ``forward_pass``; override for test-specific logic
        (e.g. multi-stage where test uses a specific sub-model).

        Args:
            batch_data: A batch of test data from the DataLoader.

        Returns:
            Dict containing at least ``"y_hat"``, ``"y_label"``,
            ``"y_predict"``.
        """
        return self.forward_pass(batch_data)

    @staticmethod
    def _try_gpu() -> torch.device:
        """Get the best available GPU device, falling back to CPU.

        Returns:
            A torch.device, ``"cuda"`` if available else ``"cpu"``.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_tensor_to_device(
        self, tensor: torch.Tensor, dtype: torch.dtype = None
    ) -> torch.Tensor:
        """Move a tensor to the trainer's device, optionally casting dtype.

        Args:
            tensor: Input tensor.
            dtype: Target dtype (e.g. ``torch.bool``), optional.

        Returns:
            Tensor moved to device and optionally cast.
        """
        result = tensor.to(self.device_)
        if dtype is not None:
            result = result.to(dtype)
        return result

    def _extract_valid_predictions(
        self,
        y_hat_full: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        same_position: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract predictions and labels at valid positions.

        Convention: ``y_hat_full[t]`` predicts ``response[t+1]``
        (next-item). Extraction always follows next-item alignment:
        ``y_hat_full[:, :-1]`` paired with ``response[:, 1:]``, with
        valid mask ``mask[:, :-1] & mask[:, 1:]``.

        When ``same_position=True``, the input uses same-position
        convention (``out[t]`` predicts ``response[t]``). The output
        is left-shifted by one and padded with a placeholder column
        to normalize into next-item view before extraction — no second
        alignment is introduced.

        Args:
            y_hat_full: Model output tensor ``[B, S]``.
            response: Response label tensor ``[B, S]``.
            mask: Valid position mask ``[B, S]``.
            same_position: Whether input uses same-position convention
                (``out[t]`` predicts ``response[t]``).

        Returns:
            Tuple ``(y_hat, y_label, valid_mask)`` where:
                y_hat: Predictions at valid positions.
                y_label: Labels at valid positions.
                valid_mask: Mask of valid adjacent pairs.
        """
        # Normalize same-position input to next-item view
        if same_position:
            y_hat_full = self._pad_to_full_sequence(y_hat_full[:, 1:])

        # Next-item alignment: t-th prediction corresponds to (t+1)-th label
        y_hat_seq = y_hat_full[:, :-1]
        y_label_seq = response.float()[:, 1:]
        mask_curr = mask[:, :-1]
        mask_next = mask[:, 1:]
        valid_mask = mask_curr & mask_next

        # Select valid positions with masking
        y_hat = torch.masked_select(y_hat_seq, valid_mask)
        y_label = torch.masked_select(y_label_seq, valid_mask)

        return y_hat, y_label, valid_mask

    def _pad_to_full_sequence(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Pad a tensor with a trailing zero column, extending ``[B, L]`` to ``[B, L+1]``.

        Used for models (GKT, SAKT, SGKT, MIKT, KQN) whose output
        length is ``S-1`` under next-item convention. The trailing
        placeholder is discarded by ``_extract_valid_predictions``'s
        ``[:, :-1]`` slice.

        Args:
            y_hat: Model output ``[B, L]``.

        Returns:
            Tensor ``[B, L+1]`` with a zero placeholder at the last column.
        """
        dummy = torch.zeros(y_hat.size(0), 1, device=y_hat.device)
        return torch.cat([y_hat, dummy], dim=1)

    def _handle_empty_batch(
        self, y_hat: torch.Tensor, y_label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handle an empty batch by raising a descriptive error.

        Args:
            y_hat: Prediction tensor.
            y_label: Label tensor.

        Returns:
            The input ``(y_hat, y_label)`` tuple unchanged.

        Raises:
            ValueError: If the label tensor is empty.
        """
        if y_label.numel() == 0:
            raise ValueError(
                "Empty valid targets in current batch: no positions satisfy "
                "the training mask alignment. Please check data preprocessing/sampling "
                "settings (e.g., min_seq_len, sample_users, batch_size)."
            )
        return y_hat, y_label

    def _generate_binary_predictions(
        self, y_hat: torch.Tensor, threshold: float = 0.0
    ) -> torch.Tensor:
        """Generate binary predictions from logits using a threshold.

        Args:
            y_hat: Prediction logits.
            threshold: Classification threshold (default 0.0).

        Returns:
            Binary prediction tensor (0 or 1).
        """
        return torch.ge(y_hat, torch.tensor(threshold).to(self.device_)).to(torch.int)

    def _setup_hyperparameters(self, hyperparams, model_name=None, dataset_name=None):
        """Set up and save hyperparameters.

        Args:
            hyperparams: Hyperparameters (dict or Namespace object).
            model_name: Model name (optional).
            dataset_name: Dataset name (optional).
        """
        from utils.hyperparam_manager import create_hyperparameter_manager

        # Create hyperparameter manager
        self.hyperparam_manager = create_hyperparameter_manager(
            args=hyperparams,
            save_dir=self.log_dir,
            model_name=model_name,
            dataset_name=dataset_name,
        )

        # Add trainer-related metadata
        # Multi-stage trainers may not have model/optimizer at build time
        if self.model is not None:
            self.hyperparam_manager.add_metadata(
                "total_params",
                sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            )
        if self.opt is not None:
            self.hyperparam_manager.add_metadata("optimizer", type(self.opt).__name__)
        if self.loss is not None:
            self.hyperparam_manager.add_metadata(
                "loss_function", type(self.loss).__name__
            )
        if self.lr_scheduler is not None:
            self.hyperparam_manager.add_metadata(
                "lr_scheduler", type(self.lr_scheduler).__name__
            )
        if hasattr(self.opt, "defaults") and "weight_decay" in self.opt.defaults:
            self.hyperparam_manager.add_metadata(
                "weight_decay", self.opt.defaults["weight_decay"]
            )
        if hyperparams is not None and getattr(hyperparams, "seed", None) is not None:
            self.hyperparam_manager.add_metadata("seed", hyperparams.seed)

        # Add device info
        if self.device_ is not None:
            device_info = self._get_device_info()
            for key, value in device_info.items():
                self.hyperparam_manager.add_metadata(key, value)

        # Save hyperparameters
        self.hyperparam_manager.save()

        # Print summary
        self.hyperparam_manager.print_summary()

    def _get_device_info(self):
        """Get device information including CUDA device details.

        Returns:
            Dict with keys like ``cuda_available``, ``cuda_device_count``,
            ``cuda_device_name``, etc.
        """
        device_info = {}

        if self.device_.type == "cuda":
            device_info["cuda_available"] = True
            device_info["cuda_device_count"] = torch.cuda.device_count()
            device_index = self.device_.index if self.device_.index is not None else 0
            device_info["cuda_device_name"] = torch.cuda.get_device_name(device_index)
            device_info["cuda_device_capability"] = torch.cuda.get_device_capability(
                device_index
            )
        else:
            device_info["cuda_available"] = False
            device_info["device_type"] = "CPU"

        return device_info

    def _load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training.

        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.model,
            self.opt,
            self.lr_scheduler,
            self.early_stopping,
            self.device_,
        )

        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1

        logger.info(f"Resumed training from epoch {self.start_epoch}")

    def _init_metric_logger(self):
        """Initialize the metric logging backend.

        Local CSV logging is always enabled; SwanLab is included unless
        ``--no_swanlab`` was set.
        """
        experiment_name = os.path.basename(self.log_dir) if self.log_dir else "run"
        config = (
            self.hyperparam_manager.get_hyperparameters_dict()
            if self.hyperparam_manager
            else {}
        )
        self.metric_logger.init_run(
            log_dir=self.log_dir,
            experiment_name=experiment_name,
            group=self.model.__class__.__name__,
            tags=["cuda" if torch.cuda.is_available() else "cpu"],
            config=config,
        )

    def _finish_metric_logger(self):
        """Finalize and shut down the metric logging backend."""
        self.metric_logger.finish()
        logger.info("Metric logging finished")

    def run(self):
        """Run the full training loop.

        Initializes metric logging, runs the training loop, and
        finalizes logging and checkpoints.
        """
        if not self._built:
            raise RuntimeError(
                "Trainer has not been built. Please call build() explicitly "
                "before run()."
            )

        self._init_metric_logger()

        self._run_training_loop()

        self._finish()

    def _run_training_loop(self, start_epoch: int | None = None) -> StageResult:
        """Run the epoch training loop for a single stage.

        Includes progress bar, epoch loop, callbacks, learning rate
        scheduling, and early stopping. Multi-stage trainers call this
        after switching ``self.model`` / ``self.opt`` / ``self.train_data``
        via ``_apply_stage``.

        Args:
            start_epoch: Starting epoch (``None`` uses ``self.start_epoch``,
                used for checkpoint resume).

        Returns:
            A :class:`StageResult` for this stage.
        """
        if start_epoch is None:
            start_epoch = self.start_epoch

        self.model.to(self.device_)
        self.loss = self.loss.to(self.device_)

        # Trigger training start callback
        self.callback_manager.on_train_begin(self.epochs, trainer=self)

        # Create progress display
        progress = create_progress()

        # Stage prefix for multi-stage logging
        stage_prefix = (
            f"[{self._current_stage.upper()}] " if self._current_stage else ""
        )
        total_label = f"{stage_prefix}Epochs" if self._current_stage else "Total Epochs"

        # Get monitor name from checkpoint callback
        checkpoint_cb = self.callback_manager.get_callback(CheckpointCallback)
        monitor_name = (
            checkpoint_cb._monitor_name() if checkpoint_cb else self._monitor_name()
        )

        # Create best metric display
        best_metric_text = None
        renderables = [progress]
        if self.early_stopping is not None:
            best_metric_text = Text(
                f"{stage_prefix}Best {monitor_name.upper()}: N/A",
                style="bold yellow",
            )
            renderables.insert(0, best_metric_text)

        with Live(Group(*renderables)):
            total_task = progress.add_task(
                f"[bold red]{total_label}", total=self.epochs, completed=start_epoch
            )
            work_task = progress.add_task(
                "[bold green]Training", total=len(self.train_data)
            )

            epoch = start_epoch
            for epoch in range(start_epoch, self.epochs):
                logger.info(f"{stage_prefix}Epoch {epoch + 1}/{self.epochs}")

                self.callback_manager.on_epoch_begin(epoch, trainer=self)

                # Training phase
                progress.reset(
                    work_task,
                    total=len(self.train_data),
                    description="[bold green]Training",
                )
                train_loss = self._process_epoch(
                    epoch, is_train=True, progress=progress, task_id=work_task
                )

                # Validation phase
                val_loss = None
                if self.val_data is not None:
                    progress.reset(
                        work_task,
                        total=len(self.val_data),
                        description="[bold cyan]Validation",
                    )
                    val_loss = self._process_epoch(
                        epoch, is_train=False, progress=progress, task_id=work_task
                    )

                self.callback_manager.on_epoch_end(
                    epoch, train_loss, val_loss, trainer=self
                )

                # Update best metric display
                if (
                    self.early_stopping is not None
                    and best_metric_text is not None
                    and self.val_data is not None
                ):
                    best_metric = checkpoint_cb.best_metric if checkpoint_cb else None
                    best_epoch = checkpoint_cb.best_epoch if checkpoint_cb else None
                    patience = self.early_stopping.cfg.patience
                    remaining = max(0, patience - self.early_stopping.num_bad_epochs)
                    best_str = (
                        f"{best_metric:.4f}" if best_metric is not None else "N/A"
                    )
                    best_metric_text.plain = (
                        f"{stage_prefix}Best {monitor_name.upper()}: {best_str} "
                        f"(Epoch {best_epoch + 1 if best_epoch is not None else 'N/A'}, "
                        f"Patience: {remaining}/{patience})"
                    )
                    best_metric_text.stylize("bold yellow")

                # Learning rate scheduler step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Update total progress
                progress.advance(total_task)

                # Check early stopping
                if self.callback_manager.should_stop(trainer=self):
                    progress.console.log(
                        f"[bold red]{stage_prefix}Early stopping triggered at "
                        f"epoch {epoch + 1}"
                    )
                    break

        logger.info("Training complete")
        self.callback_manager.on_train_end(trainer=self)

        return StageResult(
            name=self._current_stage,
            best_metric=checkpoint_cb.best_metric if checkpoint_cb else None,
            best_epoch=checkpoint_cb.best_epoch if checkpoint_cb else None,
            final_epoch=epoch,
            monitor=monitor_name,
        )

    def _process_epoch(
        self, epoch: int, is_train: bool, progress=None, task_id=None
    ) -> float:
        """Process a single epoch of training or validation.

        Args:
            epoch: Current epoch number.
            is_train: Whether this is a training (vs validation) epoch.
            progress: Rich Progress object (optional).
            task_id: Progress task ID (optional).

        Returns:
            Total loss for this epoch.
        """
        phase = "train" if is_train else "val"
        data_loader = self.train_data if is_train else self.val_data

        self.metrics_accumulator.reset(phase)
        self.model.train() if is_train else self.model.eval()
        self.callback_manager.on_phase_begin(epoch, phase, trainer=self)

        total_loss = 0.0
        for batch_idx, batch_data in enumerate(data_loader):
            self.callback_manager.on_batch_begin(epoch, batch_idx, phase, trainer=self)

            with torch.set_grad_enabled(is_train):
                if is_train:
                    loss = self._run_train_batch(batch_data)
                else:
                    loss = self._run_eval_batch(batch_data)

            total_loss += loss

            # Log per-batch loss (optional) and update global step
            if self.log_batch_metrics:
                self.metric_logger.log_batch(
                    phase=phase,
                    global_step=self._global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss=loss,
                    stage=self._current_stage,
                )
            if is_train:
                self._global_step += 1

            if progress is not None and task_id is not None:
                progress.advance(task_id)

            self.callback_manager.on_batch_end(
                epoch, batch_idx, phase, loss, trainer=self
            )

        # Aggregate and log metrics
        metrics = self.metrics_accumulator.compute(phase)
        self.metric_logger.log_metrics(
            phase=phase,
            metrics=metrics,
            step=epoch + self._metric_step_offset,
            epoch=epoch,
            stage=self._current_stage,
        )

        self.callback_manager.on_phase_end(
            epoch, phase, total_loss, metrics, trainer=self
        )

        return total_loss

    def _run_train_batch(self, batch_data: tuple[Any, ...]) -> float:
        """Execute a single training batch.

        Performs forward pass, loss computation, backpropagation,
        gradient clipping, and optimizer step.

        Args:
            batch_data: A batch of training data.

        Returns:
            Loss value for this batch (Python scalar).
        """
        self.opt.zero_grad()
        output = self.forward_pass(batch_data)
        loss = self._compute_loss(output)
        loss.backward()
        if self.max_clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_clip_grad_norm
            )
        self.opt.step()

        self.metrics_accumulator.update("train", output)

        return loss.item()

    @torch.inference_mode()
    def _run_eval_batch(self, batch_data: tuple[Any, ...]) -> float:
        """Execute a single evaluation (validation) batch.

        Args:
            batch_data: A batch of validation data.

        Returns:
            Loss value for this batch (Python scalar).
        """
        output = self.forward_pass(batch_data)
        loss = self._compute_loss(output)

        self.metrics_accumulator.update("val", output)

        return loss.item()

    @torch.inference_mode()
    def _run_test_batch(self, batch_data: tuple[Any, ...]) -> float:
        """Execute a single test batch.

        Uses ``test_forward_pass`` instead of ``forward_pass`` to
        support test-specific logic.

        Args:
            batch_data: A batch of test data.

        Returns:
            Loss value for this batch (Python scalar).
        """
        output = self.test_forward_pass(batch_data)
        loss = self._compute_loss(output)

        self.metrics_accumulator.update("test", output)

        return loss.item()

    @torch.inference_mode()
    def _evaluate_on_test_set(self, use_best_model: bool = True) -> dict[str, float]:
        """Evaluate the model on the test set after training.

        Optionally loads the best model before evaluation.

        Args:
            use_best_model: Whether to load the best checkpoint first.

        Returns:
            Dictionary of test metrics (e.g. auc, acc, rmse).
            Empty dict if test data is not available.
        """
        if self.test_data is None:
            logger.info("Test data not provided. Skipping test evaluation.")
            return {}

        # Check if test set is empty
        test_len = len(self.test_data) if hasattr(self.test_data, "__len__") else None
        if test_len is not None and test_len == 0:
            logger.warning(
                "Test DataLoader is empty (0 batches). Skipping test evaluation. "
                "Cause: The test dataset contains no samples. "
                "During data preprocessing, add_kfold_labels(test_ratio=...) "
                "assigned 0 users to the test fold (fold=-1). "
                "Training and validation completed successfully, but no test "
                "metrics will be recorded."
            )
            return {}

        best_state = None
        if use_best_model and self.early_stopping is not None:
            checkpoint_cb = self.callback_manager.get_callback(CheckpointCallback)
            if checkpoint_cb is not None and checkpoint_cb.best_model_state is not None:
                best_state = checkpoint_cb.best_model_state

        if best_state is not None:
            current_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            self.model.load_state_dict(best_state)
        else:
            current_state = None

        self.metrics_accumulator.reset("test")
        self.model.eval()

        test_progress = create_progress()

        total_loss = 0.0
        with test_progress:
            test_task = test_progress.add_task(
                "[bold magenta]Testing", total=len(self.test_data)
            )
            for batch_data in self.test_data:
                loss = self._run_test_batch(batch_data)
                total_loss += loss
                test_progress.advance(test_task)

        metrics = self.metrics_accumulator.compute("test")
        self.metric_logger.log_metrics(
            phase="test", metrics=metrics, step=self.epochs or 0, epoch=self.epochs or 0
        )

        if metrics:
            metrics_str = ", ".join(
                f"{name.upper()}={value:.4f}" for name, value in metrics.items()
            )
            logger.info(f"Test metrics: {metrics_str}")

        if current_state is not None:
            self.model.load_state_dict(current_state)

        return metrics

    def load_weights(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file.

        Handles both plain state_dict files (``best_model.pth``) and full
        checkpoint files (``last_checkpoint.pth``).  Delegates to
        :meth:`CheckpointManager.load_weights`.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        self.checkpoint_manager.load_weights(checkpoint_path, self.model, self.device_)
        self.model.to(self.device_)

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the test set and return metrics.

        Call ``load_weights()`` first to load trained weights.

        Returns:
            Dictionary with metric names and values (auc, acc, rmse).
            Empty dict if test data is not available.
        """
        if not self._built:
            raise RuntimeError("Trainer has not been built. Call build() first.")
        if self.test_data is None:
            logger.warning(
                "Test data not available. Ensure the trainer was initialized "
                "with test_data in with_data()."
            )
            return {}

        self.model.eval()
        self.metrics_accumulator.reset("test")

        eval_progress = create_progress()

        total_loss = 0.0
        with eval_progress:
            eval_task = eval_progress.add_task(
                "[bold magenta]Evaluating", total=len(self.test_data)
            )
            for batch_data in self.test_data:
                loss = self._run_test_batch(batch_data)
                total_loss += loss
                eval_progress.advance(eval_task)

        metrics = self.metrics_accumulator.compute("test")
        self.metric_logger.log_metrics(
            phase="test", metrics=metrics, step=self.epochs or 0, epoch=self.epochs or 0
        )

        if metrics:
            metrics_str = ", ".join(
                f"{name.upper()}={value:.4f}" for name, value in metrics.items()
            )
            logger.info(f"Evaluation results: {metrics_str}")
        else:
            logger.warning("No metrics computed during evaluation.")

        return metrics

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """Compute the training loss from model outputs.

        Args:
            outputs: Dict containing ``"y_hat"`` and ``"y_label"``.

        Returns:
            Loss tensor.
        """
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        return self.loss(y_hat, y_label)

    def _monitor_name(self) -> str:
        """Get the name of the metric being monitored for early stopping.

        Returns:
            Metric name string, defaulting to ``"auc"``.
        """
        if self.early_stopping is None:
            return "auc"
        return (self.early_stopping.cfg.monitor or "auc").lower()

    def _finish(self):
        """Clean up resources and finalize experiment tracking."""
        self._finish_metric_logger()
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.close()


__all__ = ["BaseTrainer", "StageResult"]
