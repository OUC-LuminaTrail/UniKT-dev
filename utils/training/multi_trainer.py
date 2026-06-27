"""多阶段训练器基类模块

提供多阶段训练的基础设施，支持顺序执行多个训练阶段，
每个阶段有独立的模型、优化器、损失函数和早停配置。

Examples:
    >>> trainer = MyMultiStageTrainer() \\
    ...     .with_experiment(exp_manager, args) \\
    ...     .with_stage_builder("km", lambda: km_config) \\
    ...     .with_stage_builder("am", lambda: am_config) \\
    ...     .build()
    >>> trainer.run()
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from rich.console import Group
from rich.live import Live
from rich.text import Text

from ..config import EarlyStopping
from ..core import get_logger, seed_everything
from ..progress import create_progress
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
class StageConfig:
    """阶段配置数据类

    Attributes:
        name: 阶段名称（如 'km', 'am'）
        model: 该阶段的模型
        optimizer: 优化器
        loss_fn: 损失函数
        train_data: 训练数据加载器
        val_data: 验证数据加载器（可选）
        epochs: 该阶段的训练轮数
        lr_scheduler: 学习率调度器（可选）
        early_stopping: 早停对象（可选）
    """

    name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    train_data: torch.utils.data.DataLoader
    val_data: torch.utils.data.DataLoader | None = None
    epochs: int = 100
    lr_scheduler: Any | None = None
    early_stopping: EarlyStopping | None = None


@dataclass
class MultiStageExperimentConfig:
    """多阶段实验配置数据类"""

    exp_manager: Any
    hyperparams: Any = None
    no_swanlab: bool = False
    log_batch_metrics: bool = False
    model_name: str = ""
    dataset_name: str = ""
    seed: int | None = None
    device: torch.device | None = None


class MultiTrainer:
    """多阶段训练器基类

    支持顺序执行多个训练阶段，每个阶段有独立的：
    - 模型
    - 优化器
    - 损失函数
    - 数据加载器
    - 早停配置

    子类需要实现：
    1. forward_pass(batch_data, stage_name): 实现前向传播
    2. _build_stages(): 在 __init__ 中注册所有阶段构建器

    可选实现：
    3. prepare_next_stage(stage_name, stage_outputs): 阶段间数据传递
    4. compute_loss(outputs, stage_name): 自定义损失计算
    5. _finish(): 训练完成后的清理工作

    Example:
        >>> @TRAINERS.register("ABKT")
        ... class ABKTTrainer(MultiTrainer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         # 准备数据
        ...         self.data = prepare_data(args)
        ...         # 注册阶段构建器
        ...         self.with_stage_builder("km", self._build_km_stage) \\
        ...             .with_stage_builder("am", self._build_am_stage)
        ...
        ...     def _build_km_stage(self):
        ...         return StageConfig(name='km', model=..., ...)
        ...
        ...     def forward_pass(self, batch_data, stage_name):
        ...         if stage_name == 'km':
        ...             return {"y_hat": ..., "y_label": ..., "y_predict": ...}
        ...         elif stage_name == 'am':
        ...             return {"y_hat": ..., "y_label": ..., "y_predict": ...}
    """

    def __init__(self, model: torch.nn.Module | None = None):
        """初始化多阶段训练器

        Args:
            model: 主模型（可选，多阶段训练可能有多个模型）
        """
        self.model = model

        # Configuration objects
        self._experiment_config: MultiStageExperimentConfig | None = None
        self._stage_builders: dict[str, Callable[[], StageConfig]] = {}

        # Internal state
        self._built = False
        self.device_: torch.device | None = None
        self.seed: int | None = None
        self.log_dir = None
        self.no_swanlab = False
        self.log_batch_metrics = False
        self.metric_logger = None

        # Current stage components (set at runtime)
        self.opt: torch.optim.Optimizer | None = None
        self.loss: torch.nn.Module | None = None
        self.train_data: torch.utils.data.DataLoader | None = None
        self.val_data: torch.utils.data.DataLoader | None = None
        self.lr_scheduler = None
        self.early_stopping: EarlyStopping | None = None
        self.epochs: int = 0

        # Stage management
        self._stage_configs: dict[str, StageConfig] = {}
        self._stage_outputs: dict[str, dict] = {}
        self._current_stage: str | None = None
        self._global_step = 0

        # Components initialized in build()
        self.metrics_accumulator: MetricsAccumulator | None = None
        self.checkpoint_manager: CheckpointManager | None = None
        self.callback_manager: CallbackManager | None = None
        self.hyperparam_manager = None
        self._custom_callbacks: list[Callback] = []
        self._custom_callback_functions: dict[str, list[Callable]] = {}

    def with_experiment(
        self,
        exp_manager,
        hyperparams=None,
        no_swanlab: bool | None = None,
        log_batch_metrics: bool | None = None,
        model_name: str = "",
        dataset_name: str = "",
        seed: int | None = None,
        device: torch.device | None = None,
    ) -> "MultiTrainer":
        """配置实验参数

        Args:
            exp_manager: 实验管理器实例
            hyperparams: 超参数（字典或对象，可选）
            no_swanlab: 是否关闭 SwanLab（None 时从 hyperparams 读取）
            log_batch_metrics: 是否记录每 batch loss（None 时从 hyperparams 读取）
            model_name: 模型名称
            dataset_name: 数据集名称
            seed: 随机种子（可选）
            device: 计算设备（可选）

        Returns:
            Self for method chaining
        """
        self._experiment_config = MultiStageExperimentConfig(
            exp_manager=exp_manager,
            hyperparams=hyperparams,
            no_swanlab=bool(no_swanlab),
            log_batch_metrics=bool(log_batch_metrics),
            model_name=model_name,
            dataset_name=dataset_name,
            seed=seed,
            device=device,
        )
        return self

    def with_stage_builder(
        self, stage_name: str, builder: Callable[[], StageConfig]
    ) -> "MultiTrainer":
        """注册阶段构建器

        Args:
            stage_name: 阶段名称（如 'km', 'am'）
            builder: 返回 StageConfig 的无参数函数

        Returns:
            Self for method chaining
        """
        self._stage_builders[stage_name] = builder
        return self

    def with_callbacks(
        self,
        callbacks: list[Callback] | None = None,
        functions: dict[str, Callable | list[Callable]] | None = None,
    ) -> "MultiTrainer":
        """配置自定义回调。"""
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

    def build(self) -> "MultiTrainer":
        """构建训练器

        Returns:
            Self for method chaining
        """
        if self._built:
            logger.warning("MultiTrainer already built. Skipping rebuild.")
            return self

        if self._experiment_config is None:
            raise ValueError(
                "Experiment configuration not set. Call with_experiment() first."
            )

        if not self._stage_builders:
            raise ValueError(
                "No stage builders registered. Call with_stage_builder() to add stages."
            )

        # 1. Setup device
        if self._experiment_config.device is None:
            self.device_ = self._try_gpu()
        else:
            self.device_ = torch.device(self._experiment_config.device)

        # 2. Setup training parameters
        # 解析 no_swanlab / log_batch_metrics：显式传入优先，否则回退到 CLI 参数
        hyperparams = self._experiment_config.hyperparams
        self.no_swanlab, self.log_batch_metrics = resolve_metric_logging_flags(
            self._experiment_config, hyperparams
        )

        # 3. Set random seed
        seed = self._experiment_config.seed
        if seed is None and hyperparams is not None:
            seed = getattr(hyperparams, "seed", 42)
        deterministic = True
        if hyperparams is not None:
            deterministic = getattr(hyperparams, "deterministic", True)
        self.seed = seed_everything(seed, deterministic=deterministic)

        # 4. Create log directory
        exp_manager = self._experiment_config.exp_manager
        if exp_manager is None:
            raise ValueError("exp_manager is required.")
        self.log_dir = exp_manager.get_log_dir()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 5. Initialize components
        self.metrics_accumulator = MetricsAccumulator()
        self.checkpoint_manager = CheckpointManager(self.log_dir)
        # 本地指标记录始终启用；SwanLab 除非 --no_swanlab
        self.metric_logger = build_default_metric_loggers(
            log_dir=self.log_dir,
            log_batch_metrics=self.log_batch_metrics,
            no_swanlab=self.no_swanlab,
        )

        # 6. Setup hyperparameters
        if hyperparams is not None:
            self._setup_hyperparameters(
                hyperparams,
                model_name=self._experiment_config.model_name,
                dataset_name=self._experiment_config.dataset_name,
            )

        logger.info("MultiTrainer built successfully")
        self._built = True
        return self

    @staticmethod
    def _try_gpu() -> torch.device:
        """获取可用的 GPU 设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_tensor_to_device(
        self, tensor: torch.Tensor, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """将张量移动到设备

        Args:
            tensor: 输入张量
            dtype: 可选的目标类型

        Returns:
            移动到设备后的张量
        """
        result = tensor.to(self.device_)
        if dtype is not None:
            result = result.to(dtype)
        return result

    # ==================== 必须实现的方法 ====================

    def forward_pass(self, batch_data: tuple[Any, ...], stage_name: str) -> dict:
        """模型前向传播（子类必须实现）

        Args:
            batch_data: 从 DataLoader 获取的一个批次数据
            stage_name: 当前阶段名称

        Returns:
            包含 "y_hat", "y_label", "y_predict" 的字典
        """
        raise NotImplementedError("Subclasses must implement forward_pass method")

    # ==================== 可选重写方法 ====================

    def prepare_next_stage(self, stage_name: str, stage_outputs: dict) -> None:
        """阶段间数据传递回调（可选实现）

        在当前阶段完成后、下一阶段开始前调用。
        子类可以重写此方法来计算下一阶段需要的数据。

        Args:
            stage_name: 刚完成的阶段名称
            stage_outputs: 该阶段的输出，包含：
                - 'best_metric': 最佳验证指标
                - 'best_epoch': 最佳 epoch
                - 'model_state': 最佳模型状态字典
                - 其他自定义输出
        """
        pass

    def compute_loss(self, outputs: dict, stage_name: str) -> torch.Tensor:
        """计算损失（可选重写）

        默认使用 self.loss(y_hat, y_label)。
        子类可以重写以实现自定义损失计算。

        Args:
            outputs: forward_pass 的输出
            stage_name: 当前阶段名称

        Returns:
            损失张量
        """
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        return self.loss(y_hat, y_label)

    # ==================== 训练循环 ====================

    def run(self):
        """运行多阶段训练"""
        if not self._built:
            raise RuntimeError(
                "MultiTrainer has not been built. Please call build() explicitly "
                "before run(), or use with_experiment() which calls build() automatically."
            )

        self._init_metric_logger()

        stages = list(self._stage_builders.keys())
        logger.info(
            f"Starting multi-stage training with {len(stages)} stages: {stages}"
        )

        for stage_idx, stage_name in enumerate(stages):
            logger.info("=" * 60)
            logger.info(f"Stage {stage_idx + 1}/{len(stages)}: {stage_name.upper()}")
            logger.info("=" * 60)

            # 通过 builder 获取阶段配置
            builder = self._stage_builders[stage_name]
            stage_config = builder()
            self._stage_configs[stage_name] = stage_config
            self._current_stage = stage_name

            # 设置当前阶段的组件
            self._setup_stage(stage_config)

            # 运行该阶段的训练
            stage_outputs = self._run_stage(stage_name)
            self._stage_outputs[stage_name] = stage_outputs

            # 阶段间数据传递
            if stage_idx < len(stages) - 1:
                logger.info("Preparing data for next stage...")
                self.prepare_next_stage(stage_name, stage_outputs)

        # 训练完成
        self._finish()

    def _setup_stage(self, config: StageConfig):
        """设置当前阶段的训练组件

        Args:
            config: 阶段配置
        """
        self.model = config.model
        self.opt = config.optimizer
        self.loss = config.loss_fn
        self.train_data = config.train_data
        self.val_data = config.val_data
        self.epochs = config.epochs
        self.lr_scheduler = config.lr_scheduler
        self.early_stopping = config.early_stopping

        # 移动模型和损失函数到设备
        self.model.to(self.device_)
        if hasattr(self.loss, "to"):
            self.loss.to(self.device_)

        # 初始化回调
        callbacks: list[Callback] = []
        callbacks.extend(self._custom_callbacks)
        if self._custom_callback_functions:
            callbacks.append(FunctionCallback(self._custom_callback_functions))
        callbacks.append(MemoryCleanupCallback(cleanup_interval=5))
        if self.early_stopping is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping=self.early_stopping,
                    stage=config.name,
                )
            )
        callbacks.append(
            CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                early_stopping=self.early_stopping,
                last_filename=f"{config.name}_last_checkpoint.pth",
                best_filename=(
                    f"best_{config.name}_model.pth"
                    if self.early_stopping is not None
                    else None
                ),
                keep_best_state=True,
            )
        )
        self.callback_manager = CallbackManager(callbacks)

        logger.info(f"Stage '{config.name}' setup complete:")
        logger.info(f"  - Model: {type(self.model).__name__}")
        logger.info(f"  - Optimizer: {type(self.opt).__name__}")
        logger.info(f"  - Loss: {type(self.loss).__name__}")
        logger.info(f"  - Epochs: {self.epochs}")
        logger.info(f"  - Train batches: {len(self.train_data)}")
        if self.val_data is not None:
            logger.info(f"  - Val batches: {len(self.val_data)}")

    def _run_stage(self, stage_name: str) -> dict:
        """运行单个阶段的训练

        Args:
            stage_name: 阶段名称

        Returns:
            阶段输出字典
        """
        # SwanLab 阶段前缀
        stage_prefix = stage_name.upper()
        checkpoint_cb = self.callback_manager.get_callback(CheckpointCallback)

        # 触发训练开始回调
        self.callback_manager.on_train_begin(
            self.epochs, trainer=self, stage_name=stage_name
        )

        # 创建进度条
        progress = create_progress()

        # 创建最佳指标显示
        best_metric_text = None
        renderables = [progress]
        if self.early_stopping is not None:
            monitor_name = self._monitor_name()
            best_metric_text = Text(
                f"[{stage_prefix}] Best {monitor_name}: N/A", style="bold yellow"
            )
            renderables.insert(0, best_metric_text)

        with Live(Group(*renderables)):
            # 创建进度任务
            total_task = progress.add_task(
                f"[bold red]{stage_prefix} Epochs", total=self.epochs
            )
            train_task = progress.add_task(
                "[bold green]Training", total=len(self.train_data), visible=False
            )
            val_task = progress.add_task(
                "[bold cyan]Validation",
                total=len(self.val_data) if self.val_data is not None else 0,
                visible=False,
            )

            for epoch in range(self.epochs):
                logger.info(f"[{stage_prefix}] Epoch {epoch + 1}/{self.epochs}")

                # Epoch 开始回调
                self.callback_manager.on_epoch_begin(
                    epoch, trainer=self, stage_name=stage_name
                )

                # 训练阶段
                progress.update(train_task, visible=True)
                progress.reset(train_task)
                train_loss = self._process_epoch(
                    epoch,
                    stage_name,
                    is_train=True,
                    progress=progress,
                    task_id=train_task,
                )
                progress.update(train_task, visible=False)

                # 验证阶段
                val_loss = None
                if self.val_data is not None:
                    progress.update(val_task, visible=True)
                    progress.reset(val_task)
                    val_loss = self._process_epoch(
                        epoch,
                        stage_name,
                        is_train=False,
                        progress=progress,
                        task_id=val_task,
                    )
                    progress.update(val_task, visible=False)

                # Epoch 结束回调
                self.callback_manager.on_epoch_end(
                    epoch, train_loss, val_loss, trainer=self, stage_name=stage_name
                )

                # 更新最佳指标显示（回调执行后）
                if best_metric_text is not None and checkpoint_cb is not None:
                    self._update_best_metric_display(
                        best_metric_text,
                        stage_prefix,
                        checkpoint_cb.best_metric,
                        checkpoint_cb.best_epoch,
                    )

                # 学习率调度器更新
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # 更新总进度
                progress.advance(total_task)
                self._global_step += 1

                # 检查是否应该停止
                if self.callback_manager.should_stop(
                    trainer=self, stage_name=stage_name
                ):
                    progress.console.log(
                        f"[bold red][{stage_prefix}] Early stopping triggered at epoch {epoch + 1}"
                    )
                    break

        best_metric = checkpoint_cb.best_metric if checkpoint_cb is not None else None
        best_epoch = checkpoint_cb.best_epoch if checkpoint_cb is not None else None
        best_model_state = (
            checkpoint_cb.best_model_state if checkpoint_cb is not None else None
        )

        best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
        best_epoch_str = str(best_epoch + 1) if best_epoch is not None else "N/A"
        logger.info(
            f"[{stage_prefix}] Training complete. "
            f"Best {self._monitor_name().upper()}: {best_metric_str} "
            f"at epoch {best_epoch_str}"
        )

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.callback_manager.on_train_end(trainer=self, stage_name=stage_name)

        return {
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "model_state": best_model_state,
            "final_epoch": epoch,
        }

    def _process_epoch(
        self,
        epoch: int,
        stage_name: str,
        is_train: bool,
        progress=None,
        task_id=None,
    ) -> float:
        """处理一个 epoch

        Args:
            epoch: 当前 epoch
            stage_name: 阶段名称
            is_train: 是否为训练模式
            progress: Rich Progress 对象
            task_id: 进度任务 ID

        Returns:
            总损失
        """
        phase = "train" if is_train else "val"
        data_loader = self.train_data if is_train else self.val_data

        self.metrics_accumulator.reset(phase)
        self.model.train() if is_train else self.model.eval()
        self.callback_manager.on_phase_begin(
            epoch, phase, trainer=self, stage_name=stage_name
        )

        total_loss = 0.0
        for batch_idx, batch_data in enumerate(data_loader):
            # Batch 回调
            self.callback_manager.on_batch_begin(
                epoch, batch_idx, phase, trainer=self, stage_name=stage_name
            )

            # 前向传播
            with torch.set_grad_enabled(is_train):
                if is_train:
                    loss = self._run_train_batch(batch_data, stage_name)
                else:
                    loss = self._run_eval_batch(batch_data, stage_name)

            total_loss += loss

            # 记录每 batch loss（可选）
            if self.log_batch_metrics:
                self.metric_logger.log_batch(
                    phase=phase,
                    global_step=self._global_step,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    loss=loss,
                    stage=stage_name,
                )

            # 更新进度条
            if progress is not None and task_id is not None:
                progress.advance(task_id)

            # Batch 结束回调
            self.callback_manager.on_batch_end(
                epoch, batch_idx, phase, loss, trainer=self, stage_name=stage_name
            )

        # 聚合并记录指标
        metrics = self.metrics_accumulator.compute(phase)
        self.metric_logger.log_metrics(
            phase=phase,
            metrics=metrics,
            step=self._global_step,
            epoch=epoch,
            stage=stage_name,
        )

        # Phase 结束回调
        self.callback_manager.on_phase_end(
            epoch,
            phase,
            total_loss,
            metrics,
            trainer=self,
            stage_name=stage_name,
        )

        return total_loss

    def _run_train_batch(self, batch_data: tuple[Any, ...], stage_name: str) -> float:
        """执行一个训练批次

        Args:
            batch_data: 批次数据
            stage_name: 阶段名称

        Returns:
            损失值
        """
        self.opt.zero_grad()
        output = self.forward_pass(batch_data, stage_name)
        loss = self.compute_loss(output, stage_name)
        loss.backward()
        self.opt.step()

        # 累积预测
        self.metrics_accumulator.update("train", output)

        return loss.item()

    @torch.no_grad()
    def _run_eval_batch(self, batch_data: tuple[Any, ...], stage_name: str) -> float:
        """执行一个验证批次

        Args:
            batch_data: 批次数据
            stage_name: 阶段名称

        Returns:
            损失值
        """
        output = self.forward_pass(batch_data, stage_name)
        loss = self.compute_loss(output, stage_name)

        # 累积预测
        self.metrics_accumulator.update("val", output)

        return loss.item()

    # ==================== 辅助方法 ====================

    def _monitor_name(self) -> str:
        """获取监控指标名称"""
        if self.early_stopping is None:
            return "auc"
        return (self.early_stopping.cfg.monitor or "auc").lower()

    def _update_best_metric_display(
        self,
        text: Text,
        stage_prefix: str,
        best_metric: float | None,
        best_epoch: int | None,
    ):
        """更新最佳指标显示"""
        if self.early_stopping is None:
            return

        patience = self.early_stopping.cfg.patience
        remaining = max(0, patience - self.early_stopping.num_bad_epochs)
        best_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
        epoch_str = str(best_epoch + 1) if best_epoch is not None else "N/A"

        text.plain = (
            f"[{stage_prefix}] Best {self._monitor_name().upper()}: {best_str} "
            f"(Epoch {epoch_str}, Patience: {remaining}/{patience})"
        )
        text.stylize("bold yellow")

    # ==================== 指标记录 ====================

    def _init_metric_logger(self):
        """初始化指标记录后端（本地始终启用，SwanLab 除非 --no_swanlab）。"""
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

    def _finish_metric_logger(self, final_metrics=None):
        """记录最终摘要并结束指标记录后端。"""
        if final_metrics:
            self.metric_logger.log_final(metrics=final_metrics, step=self._global_step)
        self.metric_logger.finish()
        logger.info("Metric logging finished")

    def _setup_hyperparameters(self, hyperparams, model_name=None, dataset_name=None):
        """设置超参数"""
        from utils.hyperparam_manager import create_hyperparameter_manager

        self.hyperparam_manager = create_hyperparameter_manager(
            args=hyperparams,
            save_dir=self.log_dir,
            model_name=model_name,
            dataset_name=dataset_name,
        )

        # 添加设备信息
        device_info = self._get_device_info()
        for key, value in device_info.items():
            self.hyperparam_manager.add_metadata(key, value)

        if self.seed is not None:
            self.hyperparam_manager.add_metadata("seed", self.seed)

        self.hyperparam_manager.save()
        self.hyperparam_manager.print_summary()

    def _get_device_info(self) -> dict:
        """获取设备信息"""
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

    def _finish(self):
        """完成训练，清理资源"""
        logger.info("=" * 60)
        logger.info("Multi-stage training complete")

        # 输出各阶段最佳指标
        for stage_name, outputs in self._stage_outputs.items():
            best_metric = outputs.get("best_metric")
            best_epoch = outputs.get("best_epoch")
            if best_metric is not None:
                logger.info(
                    f"  {stage_name.upper()}: Best {self._monitor_name()} = "
                    f"{best_metric:.4f} (Epoch "
                    f"{best_epoch + 1 if best_epoch is not None else 'N/A'})"
                )

        logger.info("=" * 60)

        # 记录各阶段最终最佳指标并结束记录后端
        final_metrics = {}
        for stage_name, outputs in self._stage_outputs.items():
            if outputs.get("best_metric") is not None:
                final_metrics[f"Final/{stage_name}_best"] = outputs["best_metric"]

        self._finish_metric_logger(final_metrics=final_metrics)


__all__ = ["MultiTrainer", "StageConfig"]
