"""训练器基类模块

提供训练器的核心功能，包括设备管理、数据加载、训练循环等。
"""

import argparse
import os
from abc import ABC, abstractmethod
from typing import Any

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

from ..config import (
    EarlyStopping,
    EarlyStoppingConfig,
)
from ..core import get_logger, seed_everything
from .callbacks import CallbackManager, EarlyStoppingCallback, MemoryCleanupCallback
from .checkpoint import CheckpointManager
from .metrics import MetricsAccumulator

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """训练器基类。

    子类需要实现：
    1. init_optimizer: 初始化优化器
    2. forward_pass: 模型前向传播逻辑

    可选实现：
    3. init_scheduler: 初始化学习率调度器

    Example:
        >>> @TRAINERS.register("MyModel")
        ... class MyTrainer(BaseTrainer):
        ...     def __init__(self, config, data_config, data_src):
        ...         # 准备数据和模型
        ...         model = self._init_model(config)
        ...         train_loader = ...
        ...         val_loader = ...
        ...         super().__init__(model, config, train_loader, val_loader)
        ...
        ...     def init_optimizer(self):
        ...         return torch.optim.Adam(self.model.parameters())
        ...
        ...     def forward_pass(self, batch):
        ...         # 前向传播逻辑
        ...         return {"y_hat": y_hat, "y_label": y_label, "y_predict": y_pred}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int,
        opt: torch.optim.Optimizer,
        loss,
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader | None = None,
        lr_scheduler=None,
        early_stopping=None,
        hyperparams=None,
        exp_manager=None,
        device: torch.device = None,
        checkpoint_path: str = None,
        use_swanlab: bool = True,
        seed: int | None = None,
    ):
        """初始化训练器。

        Args:
            model: PyTorch 模型
            epochs: 训练 epoch 数
            opt: 优化器
            loss: 损失函数
            train_data: 训练数据加载器
            val_data: 验证数据加载器（可选）
            lr_scheduler: 学习率调度器（可选）
            early_stopping: 早停配置或对象（可选）
            hyperparams: 超参数（可选）
            exp_manager: 实验管理器（必需）
            device: 计算设备（可选）
            checkpoint_path: 检查点路径（可选）
            use_swanlab: 是否使用 SwanLab（默认 True）
            seed: 随机种子（可选）
        """
        # 设备管理
        if device is None:
            self.device_ = self._try_gpu()
        else:
            self.device_ = torch.device(device)

        self.model: torch.nn.Module = model
        self.epochs: int = epochs
        self.opt = opt
        self.loss = loss
        self.train_data: torch.utils.data.DataLoader = train_data
        self.val_data: torch.utils.data.DataLoader = val_data
        self.lr_scheduler = lr_scheduler
        self.start_epoch = 0
        self.hyperparams = hyperparams

        # 设置随机种子
        if seed is None and hyperparams is not None:
            seed = getattr(hyperparams, "seed", None)
        deterministic = True
        if hyperparams is not None:
            deterministic = getattr(hyperparams, "deterministic", True)
        self.seed = seed_everything(seed, deterministic=deterministic)

        # 初始化早停
        self.early_stopping: EarlyStopping | None = None
        if early_stopping is not None:
            if isinstance(early_stopping, EarlyStopping):
                self.early_stopping = early_stopping
            elif isinstance(early_stopping, EarlyStoppingConfig):
                self.early_stopping = EarlyStopping(early_stopping)
            elif isinstance(early_stopping, dict):
                self.early_stopping = EarlyStopping(**early_stopping)
        elif hyperparams is not None:
            # 从超参数中读取早停配置
            es_patience = getattr(hyperparams, "es_patience", None)
            if es_patience is not None:
                monitor = getattr(hyperparams, "es_monitor", "auc")
                mode = getattr(hyperparams, "es_mode", "max")
                min_delta = getattr(hyperparams, "es_min_delta", 0.0)
                self.early_stopping = EarlyStopping(
                    EarlyStoppingConfig(
                        monitor=monitor,
                        mode=mode,
                        patience=es_patience,
                        min_delta=min_delta,
                    )
                )

        # 创建日志目录
        if exp_manager is None:
            raise ValueError("exp_manager is required.")

        self.log_dir = exp_manager.get_log_dir()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 初始化组件
        self.metrics_accumulator = MetricsAccumulator(use_swanlab=use_swanlab)
        self.checkpoint_manager = CheckpointManager(self.log_dir)

        # 初始化回调
        callbacks = [MemoryCleanupCallback(cleanup_interval=5)]
        if self.early_stopping is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping=self.early_stopping))
        self.callback_manager = CallbackManager(callbacks)

        # 初始化超参数管理器
        self.hyperparam_manager = None
        if hyperparams is not None:
            self._setup_hyperparameters(hyperparams)

        # 断点续训
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # SwanLab 初始化
        self.use_swanlab = use_swanlab
        if self.use_swanlab:
            from pathlib import Path

            experiment_name = Path(self.log_dir).name
            self._init_swanlab(
                project_name="kt-exp-graph",
                experiment_name=f"Run_{experiment_name}",
            )

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser) -> None:
        """添加模型特定的参数。

        子类可以重写此方法以添加额外的参数。

        Args:
            parser: ArgumentParser 实例
        """
        pass

    @abstractmethod
    def init_model(self, args, data_src):
        """初始化模型、优化器、损失函数和学习率调度器。

        注意：这个方法在旧 API 中使用，新架构中建议直接在 __init__ 中完成初始化。

        Args:
            args: 参数
            data_src: 数据源

        Returns:
            (model, optimizer, loss_function, lr_scheduler)
        """
        raise NotImplementedError("Subclasses must implement init_model method")

    @abstractmethod
    def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        """模型前向传播。

        Args:
            batch_data: 从 DataLoader 获取的一个批次数据

        Returns:
            包含 "y_hat", "y_label", "y_predict" 的字典
        """
        raise NotImplementedError("Subclasses must implement forward_pass method")

    def init_optimizer(self) -> torch.optim.Optimizer:
        """初始化优化器。

        子类应该实现此方法以返回自定义的优化器。

        Returns:
            优化器实例
        """
        raise NotImplementedError("Subclasses must implement init_optimizer method")

    def init_scheduler(self, optimizer: torch.optim.Optimizer):
        """初始化学习率调度器（可选）。

        Args:
            optimizer: 优化器

        Returns:
            学习率调度器实例（可选）
        """
        return None

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

    def _init_swanlab(self, project_name: str, experiment_name: str = None):
        """初始化 SwanLab 实验追踪。"""
        import swanlab
        from dotenv import load_dotenv

        # 加载环境变量
        load_dotenv()

        # 配置回调
        from swanlab.plugin.notification import LarkCallback

        callbacks = []
        lark_webhook = os.getenv("LARK_WEBHOOK_URL")
        lark_secret = os.getenv("LARK_SECRET")
        if lark_webhook:
            callbacks.append(LarkCallback(webhook_url=lark_webhook, secret=lark_secret))

        swanlab.init(
            workspace=os.getenv("SWANLAB_WORKSPACE", None),
            project_name=project_name,
            experiment_name=experiment_name,
            config=self.hyperparam_manager.get_hyperparameters_dict()
            if self.hyperparam_manager
            else {},
            callbacks=callbacks,
            group=self.model.__class__.__name__,
            tags=["cuda" if torch.cuda.is_available() else "cpu"],
        )
        logger.info(
            f"SwanLab initialized for project: {project_name}, run: {experiment_name}"
        )

    def run(self):
        """运行训练循环。"""
        self.model.to(self.device_)
        self.loss = self.loss.to(self.device_)

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
            best_metric_text = Text(
                f"Best {self._monitor_name()}: N/A", style="bold yellow"
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
                self.callback_manager.on_epoch_begin(epoch)

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

                    # 记录早停状态到 SwanLab
                    if self.use_swanlab:
                        try:
                            import swanlab

                            swanlab.log(
                                {
                                    "ES/Best": self.early_stopping.best_score,
                                    "ES/Num_Bad_Epochs": self.early_stopping.num_bad_epochs,
                                },
                                step=epoch,
                            )
                        except ImportError:
                            logger.warning(
                                "SwanLab is not installed. Skipping Early Stopping logging."
                            )

                    # 更新最佳指标显示（放在 step 之后，确保 num_bad_epochs 已刷新）
                    if best_metric_text is not None:
                        best_epoch = getattr(self, "_best_epoch", None)
                        patience = self.early_stopping.cfg.patience
                        remaining = max(
                            0, patience - self.early_stopping.num_bad_epochs
                        )
                        if hasattr(self, "_best_metric"):
                            best_str = f"{self._best_metric:.4f}"
                        else:
                            best_str = "N/A"

                        best_metric_text.plain = (
                            f"Best {self._monitor_name()}: {best_str} "
                            f"(Epoch {best_epoch + 1 if best_epoch is not None else 'N/A'}, "
                            f"Patience: {remaining}/{patience})"
                        )
                        best_metric_text.stylize("bold yellow")

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
                    filename="last_checkpoint.pth",
                )

                # 更新总进度
                progress.advance(total_task)

                # 检查是否应该停止
                if self.callback_manager.should_stop():
                    progress.console.log(
                        f"[bold red]Early stopping triggered at epoch {epoch + 1}"
                    )
                    break

        logger.info("Training complete")
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

        total_loss = 0.0
        for batch_idx, batch_data in enumerate(data_loader):
            # Batch 回调
            self.callback_manager.on_batch_begin(epoch, batch_idx, phase)

            # 前向传播
            with torch.set_grad_enabled(is_train):
                if is_train:
                    loss = self._run_train_batch(batch_data)
                else:
                    loss = self._run_eval_batch(batch_data)

            total_loss += loss

            # 更新进度条
            if progress is not None and task_id is not None:
                progress.advance(task_id)

            # Batch 结束回调
            self.callback_manager.on_batch_end(epoch, batch_idx, phase, loss)

        # 聚合并记录指标
        metrics = self.metrics_accumulator.compute(phase)
        self.metrics_accumulator.log(phase, metrics, epoch)

        # 保存最佳模型检查点（仅在验证阶段）
        if not is_train and self.early_stopping is not None:
            monitor_value = self._select_monitor_value(metrics, total_loss)
            if monitor_value is not None:
                self._save_best_model_checkpoint(monitor_value, epoch)

        # Phase 结束回调
        self.callback_manager.on_phase_end(epoch, phase, total_loss, metrics)

        return total_loss

    def _run_train_batch(self, batch_data: tuple[Any, ...]) -> float:
        """执行一个训练批次。"""
        self.opt.zero_grad()
        output = self.forward_pass(batch_data)
        loss = self._compute_loss(output)
        loss.backward()
        self.opt.step()

        # 累积预测
        self.metrics_accumulator.update("train", output)

        return loss.item()

    @torch.no_grad()
    def _run_eval_batch(self, batch_data: tuple[Any, ...]) -> float:
        """执行一个验证批次。"""
        output = self.forward_pass(batch_data)
        loss = self._compute_loss(output)

        # 累积预测
        self.metrics_accumulator.update("val", output)

        return loss.item()

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失。"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        return self.loss(y_hat, y_label)

    def _select_monitor_value(self, metrics: dict, val_loss: float | None) -> float:
        """根据配置选择监控指标的值。"""
        if self.early_stopping is None:
            name = "auc"
        else:
            name = (self.early_stopping.cfg.monitor or "auc").lower()

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

        # 如果仍然是 None，返回一个极差的值
        if value is None:
            if name in ["loss", "rmse"]:
                return float("inf")
            return float("-inf")
        return float(value)

    def _get_early_stopping_state(self) -> dict | None:
        """获取早停状态。"""
        if self.early_stopping is None:
            return None
        return {
            "best_score": self.early_stopping.best_score,
            "best_epoch": self.early_stopping.best_epoch,
            "num_bad_epochs": self.early_stopping.num_bad_epochs,
        }

    def _monitor_name(self) -> str:
        """获取早停监控指标名称。"""
        if self.early_stopping is None:
            return "auc"
        return (self.early_stopping.cfg.monitor or "auc").lower()

    def _save_best_model_checkpoint(self, metric: float, epoch: int):
        """保存最佳模型检查点。

        Args:
            metric: 用于判断最佳模型的指标值
            epoch: 当前 epoch
        """
        # 确定比较模式 (min 或 max)
        mode = "max"
        if self.early_stopping:
            mode = self.early_stopping.cfg.mode
        else:
            # 默认推断
            name = self._monitor_name()
            if name in ["rmse", "loss"]:
                mode = "min"

        is_better = False
        if not hasattr(self, "_best_metric"):
            is_better = True
        else:
            is_better = (
                metric > self._best_metric
                if mode == "max"
                else metric < self._best_metric
            )

        if is_better:
            self._best_metric = metric
            self._best_epoch = epoch
            logger.info(
                f"Saving best model at epoch {epoch + 1} with {self._monitor_name()} {metric:.4f}"
            )
            self.checkpoint_manager.save_weights(self.model, "best_model.pth")

    def _finish(self):
        """清理资源，结束实验追踪。"""
        if self.use_swanlab:
            import swanlab

            swanlab.finish()
            logger.info("SwanLab run finished")


__all__ = ["BaseTrainer"]
