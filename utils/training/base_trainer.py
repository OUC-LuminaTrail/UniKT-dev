"""训练器基类模块

提供训练器的核心功能，包括设备管理、数据加载、训练循环等。
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
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
from ..core import get_logger, seed_everything
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


class BaseTrainer(ABC):
    """训练器基类

    子类需要实现：
    1. __init__: 直接初始化模型
    2. forward_pass: 模型前向传播逻辑

    示例用法：
        trainer = MyTrainer(model) \\
            .with_training(epochs=150, seed=42) \\
            .with_data(train_dataset, val_dataset, batch_size=128) \\
            .with_optimization(optimizer, loss_fn, lr_scheduler) \\
            .with_experiment(exp_manager, hyperparams=args) \\
            .build()
        trainer.run()
    """

    def __init__(self, model: torch.nn.Module):
        """初始化训练器。

        Args:
            model: PyTorch 模型
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
        self.seed: int | None = None
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

    def with_training(
        self,
        epochs: int = 200,
        seed: int = 42,
        device: torch.device | None = None,
        checkpoint_path: str | None = None,
    ) -> "BaseTrainer":
        """配置训练参数。

        Args:
            epochs: 训练 epoch 数
            seed: 随机种子
            device: 计算设备（None 则自动检测）
            checkpoint_path: 检查点路径（用于断点续训）

        Returns:
            Self for method chaining
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
        """配置 torch.compile 编译优化。

        Args:
            mode: 编译模式 ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
            fullgraph: 是否要求将整个函数捕获为单一计算图
            dynamic: 是否使用动态形状追踪。None 表示自动检测，True 强制动态，False 强制静态
            backend: 编译后端

        Returns:
            Self for method chaining
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
        """配置自定义回调。

        Args:
            callbacks: 回调对象列表（可选）
            functions: 事件名 -> 函数或函数列表（可选）

        Returns:
            Self for method chaining
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
        """注册单个回调对象。"""
        self._custom_callbacks.append(callback)

    def register_callback_fn(self, event: str, func: Callable) -> None:
        """注册单个回调函数。"""
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
        """配置数据加载器。

        Args:
            train_data: 训练数据（DataLoader 或 Dataset）
            batch_size: 批次大小
            val_data: 验证数据（DataLoader 或 Dataset）
            test_data: 测试数据（DataLoader 或 Dataset，可选）
            collate_fn: 自定义 collate 函数（可选）
            val_collate_fn: 自定义验证 collate 函数（可选）
            test_collate_fn: 自定义测试 collate 函数（可选）

        Returns:
            Self for method chaining
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
        """配置优化器、损失函数和调度器。

        Args:
            optimizer: PyTorch 优化器
            loss_fn: 损失函数
            max_clip_grad_norm: 最大梯度范数（可选）
            lr_scheduler: 学习率调度器（可选）
            early_stopping: 早停配置（可选）

        Returns:
            Self for method chaining
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
        """配置实验管理和追踪。

        Args:
            exp_manager: 实验管理器实例
            hyperparams: 超参数（字典或对象，可选）
            no_swanlab: 是否关闭 SwanLab（None 时从 hyperparams 读取）
            log_batch_metrics: 是否记录每 batch loss（None 时从 hyperparams 读取）
            model_name: 模型名称
            dataset_name: 数据集名称
            skip_test: 是否跳过训练完成后的测试集评估（默认 False）

        Returns:
            Self for method chaining
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
        """构建训练器。

        Returns:
            Self for method chaining
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

        # 解析 no_swanlab / log_batch_metrics：显式传入优先，否则回退到 CLI 参数，
        # 使 --no_swanlab / --log_batch_metrics 对所有模型生效。
        hyperparams = self._experiment_config.hyperparams
        self.no_swanlab, self.log_batch_metrics = resolve_metric_logging_flags(
            self._experiment_config, hyperparams
        )

        # 3. Set random seed
        seed = self._training_config.seed
        if seed is None and hyperparams is not None:
            seed = getattr(hyperparams, "seed", 42)
        deterministic = True
        if hyperparams is not None:
            deterministic = getattr(hyperparams, "deterministic", True)
        self.seed = seed_everything(seed, deterministic=deterministic)

        # 4. Setup data loaders
        self._setup_data_loaders()

        # 5. Setup optimization
        self.opt = self._optimization_config.optimizer
        self.loss = self._optimization_config.loss_fn
        self.max_clip_grad_norm = self._optimization_config.max_clip_grad_norm
        self.lr_scheduler = self._optimization_config.lr_scheduler

        # 6. Setup early stopping
        self._setup_early_stopping()

        # 7. Create log directory
        exp_manager = self._experiment_config.exp_manager
        if exp_manager is None:
            raise ValueError("exp_manager is required.")
        self.log_dir = exp_manager.get_log_dir()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 8. Initialize components
        self.metrics_accumulator = MetricsAccumulator()
        self.checkpoint_manager = CheckpointManager(self.log_dir)
        # 本地指标记录始终启用；SwanLab 除非 --no_swanlab
        self.metric_logger = build_default_metric_loggers(
            log_dir=self.log_dir,
            log_batch_metrics=self.log_batch_metrics,
            no_swanlab=self.no_swanlab,
        )

        # 9. Initialize callbacks
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
            # 检查测试集是否未传入或为空
            if self.test_data is None:
                logger.warning(
                    "Test data was not provided during trainer initialization. Test evaluation will be skipped. \n"
                    "Cause: The model's trainer was initialized without passing test_data to with_data(). \n"
                    "Fix: Ensure to pass test_data to with_data(), or use '--skip_test' to skip test evaluation explicitly."
                )
            else:
                test_len = (
                    len(self.test_data) if hasattr(self.test_data, "__len__") else None
                )
                if test_len is not None and test_len == 0:
                    logger.warning(
                        "Test set is empty (0 samples). Test evaluation will produce no results. \n"
                        "Cause: No users were assigned to the test fold (fold=-1) during data preprocessing. \n"
                        "This happens when: \n"
                        "(1) add_kfold_labels() was called with test_ratio=0, or \n"
                        "(2) test_ratio > 0 but int(num_users * test_ratio) == 0 due to small dataset size. \n"
                        "Fix: Re-run data preprocessing with a larger test_ratio value, \n"
                        "or use '--skip_test' to skip test evaluation explicitly."
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
        """设置数据加载器。"""
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
        """设置早停。"""
        early_stopping_cfg = self._optimization_config.early_stopping
        self.early_stopping = (
            EarlyStopping(early_stopping_cfg)
            if early_stopping_cfg is not None
            else None
        )

    def _apply_compile(self):
        """如果已配置，则将 torch.compile 应用于模型。

        支持两种配置方式：
        1. 通过 with_compile() 链式调用显式配置
        2. 通过 args（hyperparams）中的 compile 参数自动配置
        """
        # 如果没有通过 with_compile 显式配置，尝试从 hyperparams 读取
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
        """模型前向传播。

        Args:
            batch_data: 从 DataLoader 获取的一个批次数据

        Returns:
            包含 "y_hat", "y_label", "y_predict" 的字典
        """
        raise NotImplementedError("Subclasses must implement forward_pass method")

    def test_forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        """测试集前向传播。

        Args:
            batch_data: 从 DataLoader 获取的一个批次数据

        Returns:
            包含 "y_hat", "y_label", "y_predict" 的字典
        """
        return self.forward_pass(batch_data)

    @staticmethod
    def _try_gpu() -> torch.device:
        """获取可用的 GPU 设备。"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_tensor_to_device(
        self, tensor: torch.Tensor, dtype: torch.dtype = None
    ) -> torch.Tensor:
        """将张量移动到设备，并可选择类型转换。

        Args:
            tensor: 输入张量
            dtype: 可选的目标类型（如 torch.bool）

        Returns:
            移动到设备后的张量
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
        skip_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        提取有效位置的预测和标签，统一处理序列对齐问题

        参数:
            y_hat_full: 模型输出的完整预测 [B, S]
            response: 响应标签 [B, S]
            mask: 有效位置掩码 [B, S]
            skip_first: 是否跳过第一个时间步（模型在t预测t+1）

        返回:
            y_hat: 有效位置的预测值
            y_label: 有效位置的标签
            valid_mask: 有效掩码
        """
        # 根据需求决定是否跳过第一个时间步
        if skip_first:
            y_hat_seq = y_hat_full[:, :-1]
            y_label_seq = response.float()[:, 1:]
            mask_curr = mask[:, :-1]
            mask_next = mask[:, 1:]
            valid_mask = mask_curr & mask_next
        else:
            y_hat_seq = y_hat_full
            y_label_seq = response.float()
            valid_mask = mask

        # 使用 mask 选择有效位置
        y_hat = torch.masked_select(y_hat_seq, valid_mask)
        y_label = torch.masked_select(y_label_seq, valid_mask)

        return y_hat, y_label, valid_mask

    def _handle_empty_batch(
        self, y_hat: torch.Tensor, y_label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        处理空批次情况，使用占位张量避免后续计算报错

        参数:
            y_hat: 预测值张量
            y_label: 标签张量

        返回:
            处理后的 (y_hat, y_label) 元组
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
        """
        生成二分类预测，使用阈值将 logits 转换为 0/1 预测

        参数:
            y_hat: 预测 logits
            threshold: 阈值（默认 0.0）

        返回:
            二分类预测张量（0 或 1）
        """
        return torch.ge(y_hat, torch.tensor(threshold).to(self.device_)).to(torch.int)

    def _setup_hyperparameters(self, hyperparams, model_name=None, dataset_name=None):
        """设置并保存超参数。

        Args:
            hyperparams: 超参数（字典或 Namespace 对象）
            model_name: 模型名称（可选）
            dataset_name: 数据集名称（可选）
        """
        from utils.hyperparam_manager import create_hyperparameter_manager

        # 创建超参数管理器
        self.hyperparam_manager = create_hyperparameter_manager(
            args=hyperparams,
            save_dir=self.log_dir,
            model_name=model_name,
            dataset_name=dataset_name,
        )

        # 添加训练器相关元数据
        try:
            from torch_geometric.profile import count_parameters

            self.hyperparam_manager.add_metadata(
                "total_params", count_parameters(self.model)
            )
        except ImportError:
            pass

        self.hyperparam_manager.add_metadata("optimizer", type(self.opt).__name__)
        self.hyperparam_manager.add_metadata("loss_function", type(self.loss).__name__)
        if self.lr_scheduler is not None:
            self.hyperparam_manager.add_metadata(
                "lr_scheduler", type(self.lr_scheduler).__name__
            )
        if hasattr(self.opt, "defaults") and "weight_decay" in self.opt.defaults:
            self.hyperparam_manager.add_metadata(
                "weight_decay", self.opt.defaults["weight_decay"]
            )
        if self.seed is not None:
            self.hyperparam_manager.add_metadata("seed", self.seed)

        # 添加设备信息
        if self.device_ is not None:
            device_info = self._get_device_info()
            for key, value in device_info.items():
                self.hyperparam_manager.add_metadata(key, value)

        # 保存超参数
        self.hyperparam_manager.save()

        # 打印摘要
        logger.info(self.hyperparam_manager.get_summary())

    def _get_device_info(self):
        """获取设备信息，包括 CUDA 设备型号。"""
        device_info = {}

        if self.device_.type == "cuda":
            device_info["cuda_available"] = True
            device_info["cuda_device_count"] = torch.cuda.device_count()
            # 获取当前设备的索引
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
        """加载检查点。"""
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
        """初始化指标记录后端（本地始终启用，SwanLab 除非 --no_swanlab）。"""
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
        """结束指标记录后端。"""
        self.metric_logger.finish()
        logger.info("Metric logging finished")

    def run(self):
        """运行训练循环。"""
        # Explicit build required
        if not self._built:
            raise RuntimeError(
                "Trainer has not been built. Please call build() explicitly "
                "before run()."
            )

        # Initialize metric loggers
        self._init_metric_logger()

        self.model.to(self.device_)
        self.loss = self.loss.to(self.device_)

        # 触发训练开始回调
        self.callback_manager.on_train_begin(self.epochs, trainer=self)

        # 创建进度条
        progress = create_progress()

        # 创建最佳指标显示
        best_metric_text = None
        renderables = [progress]
        if self.early_stopping is not None:
            best_metric_text = Text(
                f"Best {self._monitor_name().upper()}: N/A", style="bold yellow"
            )
            renderables.insert(0, best_metric_text)

        with Live(Group(*renderables)):
            # 创建进度任务
            total_task = progress.add_task(
                "[bold red]Total Epochs", total=self.epochs, completed=self.start_epoch
            )
            train_task = progress.add_task(
                "[bold green]Training", total=len(self.train_data), visible=False
            )
            val_task = progress.add_task(
                "[bold cyan]Validation",
                total=len(self.val_data) if self.val_data is not None else 0,
                visible=False,
            )

            for epoch in range(self.start_epoch, self.epochs):
                logger.info(f"Epoch {epoch + 1}/{self.epochs}")

                # Epoch 开始回调
                self.callback_manager.on_epoch_begin(epoch, trainer=self)

                # 训练阶段
                progress.update(train_task, visible=True)
                progress.reset(train_task)
                train_loss = self._process_epoch(
                    epoch, is_train=True, progress=progress, task_id=train_task
                )
                progress.update(train_task, visible=False)

                # 验证阶段
                val_loss = None
                if self.val_data is not None:
                    progress.update(val_task, visible=True)
                    progress.reset(val_task)
                    val_loss = self._process_epoch(
                        epoch, is_train=False, progress=progress, task_id=val_task
                    )
                    progress.update(val_task, visible=False)

                # Epoch 结束回调
                self.callback_manager.on_epoch_end(
                    epoch, train_loss, val_loss, trainer=self
                )

                # 更新最佳指标显示
                if (
                    self.early_stopping is not None
                    and best_metric_text is not None
                    and self.val_data is not None
                ):
                    best_epoch = getattr(self, "_best_epoch", None)
                    patience = self.early_stopping.cfg.patience
                    remaining = max(0, patience - self.early_stopping.num_bad_epochs)
                    if hasattr(self, "_best_metric"):
                        best_str = f"{self._best_metric:.4f}"
                    else:
                        best_str = "N/A"

                    best_metric_text.plain = (
                        f"Best {self._monitor_name().upper()}: {best_str} "
                        f"(Epoch {best_epoch + 1 if best_epoch is not None else 'N/A'}, "
                        f"Patience: {remaining}/{patience})"
                    )
                    best_metric_text.stylize("bold yellow")

                # 学习率调度器更新
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # 更新总进度
                progress.advance(total_task)

                # 检查是否应该停止
                if self.callback_manager.should_stop(trainer=self):
                    progress.console.log(
                        f"[bold red]Early stopping triggered at epoch {epoch + 1}"
                    )
                    break

        logger.info("Training complete")
        self.callback_manager.on_train_end(trainer=self)
        self._finish()

    def _process_epoch(
        self, epoch: int, is_train: bool, progress=None, task_id=None
    ) -> float:
        """处理一个 epoch。

        Args:
            epoch: 当前 epoch
            is_train: 是否为训练模式
            progress: Rich Progress 对象（可选）
            task_id: 进度任务 ID（可选）

        Returns:
            总损失
        """
        phase = "train" if is_train else "val"
        data_loader = self.train_data if is_train else self.val_data

        self.metrics_accumulator.reset(phase)
        self.model.train() if is_train else self.model.eval()
        self.callback_manager.on_phase_begin(epoch, phase, trainer=self)

        total_loss = 0.0
        for batch_idx, batch_data in enumerate(data_loader):
            # Batch 回调
            self.callback_manager.on_batch_begin(epoch, batch_idx, phase, trainer=self)

            # 前向传播
            with torch.set_grad_enabled(is_train):
                if is_train:
                    loss = self._run_train_batch(batch_data)
                else:
                    loss = self._run_eval_batch(batch_data)

            total_loss += loss

            # 记录每 batch loss（可选）并更新全局训练步数
            if self.log_batch_metrics:
                self.metric_logger.log_batch(
                    phase=phase,
                    global_step=self._global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss=loss,
                )
            if is_train:
                self._global_step += 1

            # 更新进度条
            if progress is not None and task_id is not None:
                progress.advance(task_id)

            # Batch 结束回调
            self.callback_manager.on_batch_end(
                epoch, batch_idx, phase, loss, trainer=self
            )

        # 聚合并记录指标
        metrics = self.metrics_accumulator.compute(phase)
        self.metric_logger.log_metrics(
            phase=phase, metrics=metrics, step=epoch, epoch=epoch
        )

        # Phase 结束回调
        self.callback_manager.on_phase_end(
            epoch, phase, total_loss, metrics, trainer=self
        )

        return total_loss

    def _run_train_batch(self, batch_data: tuple[Any, ...]) -> float:
        """执行一个训练批次。"""
        self.opt.zero_grad()
        output = self.forward_pass(batch_data)
        loss = self._compute_loss(output)
        loss.backward()
        if self.max_clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_clip_grad_norm
            )
        self.opt.step()

        # 累积预测
        self.metrics_accumulator.update("train", output)

        return loss.item()

    @torch.inference_mode()
    def _run_eval_batch(self, batch_data: tuple[Any, ...]) -> float:
        """执行一个验证批次。"""
        output = self.forward_pass(batch_data)
        loss = self._compute_loss(output)

        # 累积预测
        self.metrics_accumulator.update("val", output)

        return loss.item()

    @torch.inference_mode()
    def _run_test_batch(self, batch_data: tuple[Any, ...]) -> float:
        """执行一个测试批次。"""
        output = self.test_forward_pass(batch_data)
        loss = self._compute_loss(output)

        # 累积预测
        self.metrics_accumulator.update("test", output)

        return loss.item()

    @torch.inference_mode()
    def _evaluate_on_test_set(self, use_best_model: bool = True) -> dict[str, float]:
        """训练结束后在测试集上评估并记录指标。"""
        if self.test_data is None:
            logger.info("Test data not provided. Skipping test evaluation.")
            return {}

        # 检查测试集是否为空
        test_len = len(self.test_data) if hasattr(self.test_data, "__len__") else None
        if test_len is not None and test_len == 0:
            logger.warning(
                "Test DataLoader is empty (0 batches). Skipping test evaluation. "
                "Cause: The test dataset contains no samples. "
                "During data preprocessing, add_kfold_labels(test_ratio=...) assigned 0 users to the test fold (fold=-1). "
                "Training and validation completed successfully, but no test metrics will be recorded."
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

        # 创建测试阶段进度条
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
        """计算损失。"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        return self.loss(y_hat, y_label)

    def _monitor_name(self) -> str:
        """获取早停监控指标名称。"""
        if self.early_stopping is None:
            return "auc"
        return (self.early_stopping.cfg.monitor or "auc").lower()

    def _finish(self):
        """清理资源，结束实验追踪。"""
        self._finish_metric_logger()


__all__ = ["BaseTrainer"]
