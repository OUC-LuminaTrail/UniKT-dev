"""多阶段训练器模块。

在 :class:`BaseTrainer` 之上提供顺序多阶段训练：每个阶段拥有独立的
模型 / 优化器 / 损失 / 数据 / 早停配置，阶段之间通过钩子传递数据。

Example:
    >>> @TRAINERS.register("ABKT")
    ... class ABKTTrainer(MultiTrainer):
    ...     def __init__(self, args, data_src, exp_manager):
    ...         super().__init__(device=args.device, seed=args.seed)
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
from ..core import get_logger, seed_everything
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
    """单个阶段的已构建组件。

    由阶段构建器（``StageConfig.build``）返回，描述该阶段使用的模型、优化器、
    损失、数据、epoch 数与早停配置等。

    Attributes:
        model: 该阶段的 PyTorch 模型。
        optimizer: 该阶段的优化器。
        loss_fn: 该阶段的损失函数。
        train_data: 训练数据（DataLoader 或 Dataset）。
        val_data: 验证数据（可选）。
        test_data: 测试数据（可选，多阶段训练通常用末阶段的 val 充当测试）。
        epochs: 该阶段的训练轮数。
        lr_scheduler: 学习率调度器（可选）。
        early_stopping: 早停配置（可选，传配置而非已构建对象）。
        max_clip_grad_norm: 梯度裁剪最大范数（可选）。
        checkpoint_monitor: 保存最佳模型时监控的指标（可选）。默认与早停指标一致；
            显式传入可解耦“保存最佳模型”与“早停”所监控的指标。
        checkpoint_mode: 最佳模型指标方向 ``'max'``/``'min'``（可选）。
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
    """一个训练阶段的声明：名称 + 延迟构建器。

    ``build`` 是一个无参可调用对象，仅在阶段**即将开始训练时**才被调用，
    因此可以基于前序阶段结束后才确定的数据（例如 boosting 残差）来构建。

    Attributes:
        name: 阶段名称（用于日志前缀、检查点文件名、指标 series 区分）。
        build: 返回 :class:`StageComponents` 的无参可调用对象。
    """

    name: str
    build: Callable[[], StageComponents]


class MultiTrainer(BaseTrainer):
    """多阶段训练器。

    子类需要实现：

    1. :meth:`build_stages`：返回按顺序执行的阶段列表。
    2. :meth:`forward_pass`：``forward_pass(batch_data)``，通过 ``self._current_stage``
       区分当前阶段。

    可选重写：

    3. :meth:`on_stage_begin`：阶段开始前的准备（默认空）。
    4. :meth:`on_stage_complete`：阶段结束后的处理（默认空），常用于向下一阶段传递数据。
    5. :meth:`_compute_loss`：自定义损失（默认 ``self.loss(y_hat, y_label)``）。

    构造时通过 ``device`` / ``seed`` 指定设备与随机种子，再链式调用
    :meth:`with_experiment` 与 :meth:`build` 完成基础设施初始化。
    """

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        seed: int | None = None,
        deterministic: bool = True,
    ):
        # 多阶段训练没有单一模型：模型在各阶段构建后挂到 self.model
        super().__init__(model=None)

        self._device: str | torch.device | None = device
        self._seed: int | None = seed
        self._deterministic: bool = deterministic

        # 阶段状态
        self._stages: list[StageConfig] = []
        self._stage_results: dict[str, StageResult] = {}
        self._elapsed_epochs: int = 0

    # ==================== 构建 ====================

    def build(self) -> "MultiTrainer":
        """构建多阶段训练器的基础设施（设备 / 种子 / 日志 / 超参数）。

        与 :meth:`BaseTrainer.build` 不同，这里**不**配置模型 / 数据 / 优化器
        （它们随阶段切换），只初始化跨阶段共享的设施。
        """
        if self._built:
            logger.warning("MultiTrainer already built. Skipping rebuild.")
            return self

        if self._experiment_config is None:
            raise ValueError(
                "Experiment configuration not set. Call with_experiment() first."
            )

        # 1. 设备
        self.device_ = (
            torch.device(self._device) if self._device is not None else self._try_gpu()
        )

        # 2. 解析日志开关（显式传入优先，否则回退到 CLI 参数）
        hyperparams = self._experiment_config.hyperparams
        self.no_swanlab, self.log_batch_metrics = resolve_metric_logging_flags(
            self._experiment_config, hyperparams
        )

        # 3. 随机种子
        seed = self._seed
        if seed is None and hyperparams is not None:
            seed = getattr(hyperparams, "seed", 42)
        deterministic = self._deterministic
        if hyperparams is not None:
            deterministic = getattr(hyperparams, "deterministic", True)
        self.seed = seed_everything(seed, deterministic=deterministic)

        # 4. 日志目录
        exp_manager = self._experiment_config.exp_manager
        if exp_manager is None:
            raise ValueError("exp_manager is required.")
        self.log_dir = exp_manager.get_log_dir()
        os.makedirs(self.log_dir, exist_ok=True)

        # 5. 共享组件
        self.metrics_accumulator = MetricsAccumulator()
        self.checkpoint_manager = CheckpointManager(self.log_dir)
        # 本地指标记录始终启用；SwanLab 除非 --no_swanlab
        self.metric_logger = build_default_metric_loggers(
            log_dir=self.log_dir,
            log_batch_metrics=self.log_batch_metrics,
            no_swanlab=self.no_swanlab,
        )

        # 6. 超参数（model/opt 此时为 None，_setup_hyperparameters 会自动跳过）
        if hyperparams is not None:
            self._setup_hyperparameters(
                hyperparams,
                model_name=self._experiment_config.model_name,
                dataset_name=self._experiment_config.dataset_name,
            )

        logger.info("MultiTrainer built successfully")
        self._built = True
        return self

    # ==================== 子类钩子 ====================

    def build_stages(self) -> list[StageConfig]:
        """返回按顺序执行的阶段列表（子类必须实现）。"""
        raise NotImplementedError("Subclasses must implement build_stages()")

    def on_stage_begin(self, name: str) -> None:
        """阶段开始前的钩子（可选重写）。"""
        pass

    def on_stage_complete(self, name: str, result: StageResult) -> None:
        """阶段结束后的钩子（可选重写）。

        在该阶段最佳模型已加载回 ``self.model`` 之后调用，常用于为下一阶段
        准备数据（如计算 boosting 残差）。

        Args:
            name: 刚结束的阶段名称。
            result: 该阶段的 :class:`StageResult`。
        """
        pass

    # ==================== 运行 ====================

    def run(self) -> None:
        """顺序执行所有阶段。"""
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

            # 延迟构建：阶段组件可能依赖前序阶段的输出
            setup = stage.build()
            self._apply_stage(stage.name, setup)

            result = self._run_training_loop()

            # 把该阶段最佳模型载回 self.model，供后续阶段 / 收尾使用
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
        """将阶段组件挂到实例属性上，并重建该阶段的回调管理器。"""
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
        # 指标记录 step 在阶段间累计，保证 SwanLab x 轴单调
        self._metric_step_offset = self._elapsed_epochs
        self._best_metric = None
        self._best_epoch = None

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
        """构建单个阶段的回调列表（含阶段独立的早停与检查点）。"""
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

    # ==================== 指标记录 / 收尾 ====================

    def _init_metric_logger(self) -> None:
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

    def _finish(self) -> None:
        """收尾：汇总各阶段最佳指标并结束指标记录。"""
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


__all__ = ["MultiTrainer", "StageConfig", "StageComponents"]
