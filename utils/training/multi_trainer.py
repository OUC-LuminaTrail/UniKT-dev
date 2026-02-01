"""多阶段训练器基类模块

提供多阶段训练的基础设施，支持顺序执行多个训练阶段，
每个阶段有独立的模型、优化器、损失函数和早停配置。
"""

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column
from rich.text import Text

from ..config import EarlyStopping
from ..core import get_logger, seed_everything
from .callbacks import CallbackManager, EarlyStoppingCallback, MemoryCleanupCallback
from .checkpoint import CheckpointManager
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
        early_stopping: 早停配置（可选）
    """

    name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    train_data: torch.utils.data.DataLoader
    val_data: Optional[torch.utils.data.DataLoader] = None
    epochs: int = 100
    lr_scheduler: Optional[Any] = None
    early_stopping: Optional[EarlyStopping] = None


class MultiTrainer:
    """多阶段训练器基类

    支持顺序执行多个训练阶段，每个阶段有独立的：
    - 模型
    - 优化器
    - 损失函数
    - 数据加载器
    - 早停配置

    子类需要实现：
    1. get_stages(): 返回阶段名称列表
    2. init_stage(stage_name): 初始化指定阶段的配置
    3. forward_pass(batch_data, stage_name): 实现前向传播

    可选实现：
    4. prepare_next_stage(stage_name, stage_outputs): 阶段间数据传递
    5. compute_loss(outputs, stage_name): 自定义损失计算

    Example:
        >>> @TRAINERS.register("ABKT")
        ... class ABKTTrainer(MultiTrainer):
        ...     def get_stages(self):
        ...         return ['km', 'am']
        ...
        ...     def init_stage(self, stage_name):
        ...         if stage_name == 'km':
        ...             return StageConfig(name='km', model=..., ...)
        ...         elif stage_name == 'am':
        ...             return StageConfig(name='am', model=..., ...)
        ...
        ...     def forward_pass(self, batch_data, stage_name):
        ...         if stage_name == 'km':
        ...             return {"y_hat": ..., "y_label": ..., "y_predict": ...}
        ...         elif stage_name == 'am':
        ...             return {"y_hat": ..., "y_label": ..., "y_predict": ...}
    """

    def __init__(
        self,
        args=None,
        data_src=None,
        exp_manager=None,
        device: Optional[torch.device] = None,
        use_swanlab: bool = True,
        seed: Optional[int] = None,
    ):
        """初始化多阶段训练器

        Args:
            args: 命令行参数
            data_src: 数据源
            exp_manager: 实验管理器
            device: 计算设备（可选）
            use_swanlab: 是否使用 SwanLab
            seed: 随机种子（可选）
        """
        self.args = args
        self.data_src = data_src
        self.exp_manager = exp_manager

        # 设备管理
        if device is None:
            self.device_ = self._try_gpu()
        else:
            self.device_ = torch.device(device)

        # 设置随机种子
        if seed is None and args is not None:
            seed = getattr(args, "seed", None)
        self.seed = seed_everything(seed)

        # 日志目录
        if exp_manager is not None:
            self.log_dir = exp_manager.get_log_dir()
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        else:
            self.log_dir = "./runs/multi_trainer_default"
            os.makedirs(self.log_dir, exist_ok=True)

        # SwanLab 配置
        self.use_swanlab = use_swanlab
        self._swanlab_initialized = False

        # 阶段管理
        self._stage_configs: dict[str, StageConfig] = {}
        self._stage_outputs: dict[str, dict] = {}
        self._current_stage: Optional[str] = None
        self._global_step = 0  # 跨阶段的全局步数

        # 当前阶段的组件（运行时设置）
        self.model: Optional[torch.nn.Module] = None
        self.opt: Optional[torch.optim.Optimizer] = None
        self.loss: Optional[torch.nn.Module] = None
        self.train_data: Optional[torch.utils.data.DataLoader] = None
        self.val_data: Optional[torch.utils.data.DataLoader] = None
        self.lr_scheduler = None
        self.early_stopping: Optional[EarlyStopping] = None
        self.epochs: int = 0

        # 指标和检查点管理（所有阶段共享）
        self.metrics_accumulator = MetricsAccumulator(use_swanlab=use_swanlab)
        self.checkpoint_manager = CheckpointManager(self.log_dir)

        # 超参数管理
        self.hyperparam_manager = None
        if args is not None:
            self._setup_hyperparameters(args)

    @staticmethod
    def _try_gpu() -> torch.device:
        """获取可用的 GPU 设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_tensor_to_device(
        self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None
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

    # ==================== 抽象方法 ====================

    @abstractmethod
    def get_stages(self) -> list[str]:
        """返回阶段名称列表

        Returns:
            阶段名称列表，如 ['km', 'am']
        """
        raise NotImplementedError("Subclasses must implement get_stages method")

    @abstractmethod
    def init_stage(self, stage_name: str) -> StageConfig:
        """初始化指定阶段的配置

        Args:
            stage_name: 阶段名称

        Returns:
            StageConfig 对象，包含该阶段的所有配置
        """
        raise NotImplementedError("Subclasses must implement init_stage method")

    @abstractmethod
    def forward_pass(self, batch_data: tuple[Any, ...], stage_name: str) -> dict:
        """模型前向传播

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

    def init_model(self, args, data_src):
        """兼容 BaseTrainer 接口的占位方法

        MultiTrainer 使用 init_stage() 代替此方法。
        """
        raise NotImplementedError(
            "MultiTrainer uses init_stage() instead of init_model(). "
            "Please implement get_stages() and init_stage() methods."
        )

    # ==================== 训练循环 ====================

    def run(self):
        """运行多阶段训练"""
        stages = self.get_stages()
        logger.info(
            f"Starting multi-stage training with {len(stages)} stages: {stages}"
        )

        # 初始化 SwanLab（所有阶段共享一个 run）
        if self.use_swanlab and not self._swanlab_initialized:
            self._init_swanlab()

        for stage_idx, stage_name in enumerate(stages):
            logger.info("=" * 60)
            logger.info(f"Stage {stage_idx + 1}/{len(stages)}: {stage_name.upper()}")
            logger.info("=" * 60)

            # 初始化阶段配置
            stage_config = self.init_stage(stage_name)
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
        callbacks = [MemoryCleanupCallback(cleanup_interval=5)]
        if self.early_stopping is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping=self.early_stopping))
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
        # 初始化最佳指标跟踪
        best_metric = None
        best_epoch = None
        best_model_state = None

        # SwanLab 阶段前缀
        stage_prefix = stage_name.upper()

        # 触发训练开始回调
        self.callback_manager.on_train_begin(self.epochs)

        # 创建进度条
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(table_column=Column(justify="right")),
            TimeRemainingColumn(),
            expand=True,
        )

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
                self.callback_manager.on_epoch_begin(epoch)

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
                self.callback_manager.on_epoch_end(epoch, train_loss, val_loss)

                # 早停检查
                if self.early_stopping is not None and self.val_data is not None:
                    metrics = self.metrics_accumulator.compute("val")
                    monitor_value = self._select_monitor_value(metrics, val_loss)

                    if monitor_value is not None:
                        # 更新早停回调
                        for cb in self.callback_manager.callbacks:
                            if isinstance(cb, EarlyStoppingCallback):
                                cb.step(monitor_value, epoch)
                                break

                        # 保存最佳模型
                        is_better = self._is_better_metric(monitor_value, best_metric)
                        if is_better:
                            best_metric = monitor_value
                            best_epoch = epoch
                            best_model_state = {
                                k: v.cpu().clone()
                                for k, v in self.model.state_dict().items()
                            }
                            # 保存检查点
                            self.checkpoint_manager.save_weights(
                                self.model, f"best_{stage_name}_model.pth"
                            )
                            logger.info(
                                f"[{stage_prefix}] New best {self._monitor_name()}: "
                                f"{monitor_value:.4f} at epoch {epoch + 1}"
                            )

                    # 记录早停状态到 SwanLab
                    if self.use_swanlab:
                        self._log_early_stopping_state(stage_prefix, epoch)

                    # 更新最佳指标显示
                    if best_metric_text is not None:
                        self._update_best_metric_display(
                            best_metric_text, stage_prefix, best_metric, best_epoch
                        )

                # 学习率调度器更新
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # 保存检查点
                self.checkpoint_manager.save_checkpoint(
                    epoch,
                    self.model,
                    self.opt,
                    self.lr_scheduler,
                    early_stopping_state=self._get_early_stopping_state(),
                    filename=f"{stage_name}_last_checkpoint.pth",
                )

                # 更新总进度
                progress.advance(total_task)
                self._global_step += 1

                # 检查是否应该停止
                if self.callback_manager.should_stop():
                    progress.console.log(
                        f"[bold red][{stage_prefix}] Early stopping triggered "
                        f"at epoch {epoch + 1}"
                    )
                    break

        best_metric_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
        best_epoch_str = str(best_epoch + 1) if best_epoch is not None else "N/A"
        logger.info(
            f"[{stage_prefix}] Training complete. "
            f"Best {self._monitor_name()}: {best_metric_str} "
            f"at epoch {best_epoch_str}"
        )

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

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
        stage_prefix = stage_name.upper()

        self.metrics_accumulator.reset(phase)
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        for batch_idx, batch_data in enumerate(data_loader):
            # Batch 回调
            self.callback_manager.on_batch_begin(epoch, batch_idx, phase)

            # 前向传播
            with torch.set_grad_enabled(is_train):
                if is_train:
                    loss = self._run_train_batch(batch_data, stage_name)
                else:
                    loss = self._run_eval_batch(batch_data, stage_name)

            total_loss += loss

            # 更新进度条
            if progress is not None and task_id is not None:
                progress.advance(task_id)

            # Batch 结束回调
            self.callback_manager.on_batch_end(epoch, batch_idx, phase, loss)

        # 聚合并记录指标
        metrics = self.metrics_accumulator.compute(phase)

        # 使用阶段前缀记录指标
        if self.use_swanlab:
            self._log_stage_metrics(stage_prefix, phase, metrics, epoch)

        # Phase 结束回调
        self.callback_manager.on_phase_end(epoch, phase, total_loss, metrics)

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

    def _is_better_metric(self, current: float, best: Optional[float]) -> bool:
        """判断当前指标是否更好

        Args:
            current: 当前指标值
            best: 最佳指标值

        Returns:
            当前是否更好
        """
        if best is None:
            return True

        mode = "max"
        if self.early_stopping is not None:
            mode = self.early_stopping.cfg.mode
        else:
            name = self._monitor_name()
            if name in ["rmse", "loss"]:
                mode = "min"

        if mode == "max":
            return current > best
        else:
            return current < best

    def _monitor_name(self) -> str:
        """获取监控指标名称"""
        if self.early_stopping is None:
            return "auc"
        return (self.early_stopping.cfg.monitor or "auc").lower()

    def _select_monitor_value(self, metrics: dict, val_loss: Optional[float]) -> float:
        """选择监控指标的值"""
        name = self._monitor_name()

        value = None
        if name == "loss":
            value = float(val_loss) if val_loss is not None else None
        elif name in metrics:
            value = metrics[name]

        # 回退策略
        if value is None:
            if metrics.get("auc") is not None:
                value = float(metrics["auc"])
            elif metrics.get("acc") is not None:
                value = float(metrics["acc"])
            elif metrics.get("rmse") is not None:
                value = float(metrics["rmse"])

        if value is None:
            if name in ["loss", "rmse"]:
                return float("inf")
            return float("-inf")

        return float(value)

    def _get_early_stopping_state(self) -> Optional[dict]:
        """获取早停状态"""
        if self.early_stopping is None:
            return None
        return {
            "best_score": self.early_stopping.best_score,
            "best_epoch": self.early_stopping.best_epoch,
            "num_bad_epochs": self.early_stopping.num_bad_epochs,
        }

    def _update_best_metric_display(
        self,
        text: Text,
        stage_prefix: str,
        best_metric: Optional[float],
        best_epoch: Optional[int],
    ):
        """更新最佳指标显示"""
        if self.early_stopping is None:
            return

        patience = self.early_stopping.cfg.patience
        remaining = max(0, patience - self.early_stopping.num_bad_epochs)
        best_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
        epoch_str = str(best_epoch + 1) if best_epoch is not None else "N/A"

        text.plain = (
            f"[{stage_prefix}] Best {self._monitor_name()}: {best_str} "
            f"(Epoch {epoch_str}, Patience: {remaining}/{patience})"
        )
        text.stylize("bold yellow")

    # ==================== SwanLab 集成 ====================

    def _init_swanlab(self):
        """初始化 SwanLab"""
        import swanlab
        from dotenv import load_dotenv

        load_dotenv()

        from swanlab.plugin.notification import LarkCallback

        callbacks = []
        lark_webhook = os.getenv("LARK_WEBHOOK_URL")
        lark_secret = os.getenv("LARK_SECRET")
        if lark_webhook:
            callbacks.append(LarkCallback(webhook_url=lark_webhook, secret=lark_secret))

        # 构建超参数配置
        config = {}
        if self.hyperparam_manager is not None:
            config = self.hyperparam_manager.get_hyperparameters_dict()

        experiment_name = os.path.basename(self.log_dir) if self.log_dir else "run"
        model_name = type(self).__name__.replace("Trainer", "")

        swanlab.init(
            workspace=os.getenv("SWANLAB_WORKSPACE", None),
            project_name="kt-exp-graph",
            experiment_name=f"Run_{experiment_name}",
            config=config,
            callbacks=callbacks,
            group=model_name,
            tags=["cuda" if torch.cuda.is_available() else "cpu", "multi-stage"],
        )

        self._swanlab_initialized = True
        logger.info(f"SwanLab initialized for multi-stage training: {experiment_name}")

    def _log_stage_metrics(
        self, stage_prefix: str, phase: str, metrics: dict, epoch: int
    ):
        """记录阶段指标到 SwanLab"""
        import swanlab

        phase_prefix = "Train" if phase == "train" else "Val"
        log_data = {}
        for metric_name, value in metrics.items():
            if value is not None:
                log_data[f"{stage_prefix}/{phase_prefix}/{metric_name}"] = value

        if log_data:
            swanlab.log(log_data, step=self._global_step)

    def _log_early_stopping_state(self, stage_prefix: str, epoch: int):
        """记录早停状态到 SwanLab"""
        import swanlab

        if self.early_stopping is not None:
            swanlab.log(
                {
                    f"{stage_prefix}/ES/Best": self.early_stopping.best_score
                    if self.early_stopping.best_score is not None
                    else 0,
                    f"{stage_prefix}/ES/Num_Bad_Epochs": self.early_stopping.num_bad_epochs,
                },
                step=self._global_step,
            )

    def _setup_hyperparameters(self, hyperparams):
        """设置超参数"""
        from utils.hyperparam_manager import create_hyperparameter_manager

        self.hyperparam_manager = create_hyperparameter_manager(
            args=hyperparams,
            save_dir=self.log_dir,
        )

        # 添加设备信息
        device_info = self._get_device_info()
        for key, value in device_info.items():
            self.hyperparam_manager.add_metadata(key, value)

        if self.seed is not None:
            self.hyperparam_manager.add_metadata("seed", self.seed)

        self.hyperparam_manager.save()
        logger.info(self.hyperparam_manager.get_summary())

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
                    f"{best_metric:.4f} (Epoch {best_epoch + 1 if best_epoch is not None else 'N/A'})"
                )

        logger.info("=" * 60)

        if self.use_swanlab:
            import swanlab

            # 记录最终指标
            final_metrics = {}
            for stage_name, outputs in self._stage_outputs.items():
                if outputs.get("best_metric") is not None:
                    final_metrics[f"Final/{stage_name}_best"] = outputs["best_metric"]

            if final_metrics:
                swanlab.log(final_metrics)

            swanlab.finish()
            logger.info("SwanLab run finished")


__all__ = ["MultiTrainer", "StageConfig"]
