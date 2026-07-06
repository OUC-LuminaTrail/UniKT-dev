"""训练回调系统

提供训练过程中的回调机制，包括早停、检查点、内存管理等。
"""

from abc import ABC
from collections.abc import Callable
from typing import TypeVar

from ..config import EarlyStopping
from ..core import get_logger
from .checkpoint import CheckpointManager

logger = get_logger(__name__)

TCallback = TypeVar("TCallback", bound="Callback")


class Callback(ABC):
    """回调基类。

    定义了训练过程中的回调接口。子类可以实现特定的回调逻辑。
    """

    def on_train_begin(self, epochs: int, **kwargs):
        """训练开始时调用。"""
        pass

    def on_train_end(self, **kwargs):
        """训练结束时调用。"""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        """Epoch 开始时调用。"""
        pass

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Epoch 结束时调用。"""
        pass

    def on_phase_begin(self, epoch: int, phase: str, **kwargs):
        """Phase（train/val）开始时调用。"""
        pass

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Phase（train/val）结束时调用。"""
        pass

    def on_batch_begin(self, epoch: int, batch_idx: int, phase: str, **kwargs):
        """Batch 开始时调用。"""
        pass

    def on_batch_end(
        self, epoch: int, batch_idx: int, phase: str, loss: float, **kwargs
    ):
        """Batch 结束时调用。"""
        pass

    def should_stop(self, **kwargs) -> bool:
        """检查是否应该停止训练。"""
        return False


class FunctionCallback(Callback):
    """函数式回调包装器。

    使用字典将事件名称映射到函数或函数列表。
    """

    def __init__(self, handlers: dict[str, Callable | list[Callable]]):
        self._handlers: dict[str, list[Callable]] = {}
        for name, funcs in handlers.items():
            if funcs is None:
                continue
            if isinstance(funcs, list):
                self._handlers[name] = funcs
            else:
                self._handlers[name] = [funcs]

    def _call(self, name: str, *args, **kwargs) -> None:
        for func in self._handlers.get(name, []):
            func(*args, **kwargs)

    def on_train_begin(self, epochs: int, **kwargs):
        self._call("on_train_begin", epochs)

    def on_train_end(self, **kwargs):
        self._call("on_train_end")

    def on_epoch_begin(self, epoch: int, **kwargs):
        self._call("on_epoch_begin", epoch)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        self._call("on_epoch_end", epoch, train_loss, val_loss)

    def on_phase_begin(self, epoch: int, phase: str, **kwargs):
        self._call("on_phase_begin", epoch, phase)

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        self._call("on_phase_end", epoch, phase, loss, metrics)

    def on_batch_begin(self, epoch: int, batch_idx: int, phase: str, **kwargs):
        self._call("on_batch_begin", epoch, batch_idx, phase)

    def on_batch_end(
        self, epoch: int, batch_idx: int, phase: str, loss: float, **kwargs
    ):
        self._call("on_batch_end", epoch, batch_idx, phase, loss)

    def should_stop(self, **kwargs) -> bool:
        results = [func() for func in self._handlers.get("should_stop", [])]
        return any(bool(result) for result in results)


class CallbackManager:
    """回调管理器。

    管理多个回调对象，按顺序触发它们的回调方法。
    """

    def __init__(self, callbacks: list[Callback]):
        """初始化回调管理器。

        Args:
            callbacks: 回调对象列表
        """
        self.callbacks = [cb for cb in callbacks if cb is not None]

    def trigger(self, method_name: str, *args, **kwargs):
        """触发所有回调的指定方法。

        Args:
            method_name: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
        """
        for callback in self.callbacks:
            getattr(callback, method_name)(*args, **kwargs)

    def get_callback(self, callback_type: type[TCallback]) -> TCallback | None:
        """获取指定类型的第一个回调。"""
        for cb in self.callbacks:
            if isinstance(cb, callback_type):
                return cb
        return None

    def should_stop(self, **kwargs) -> bool:
        """检查是否应该停止训练。

        Returns:
            是否应该停止训练
        """
        return any(cb.should_stop(**kwargs) for cb in self.callbacks)

    # 便捷方法
    def on_train_begin(self, epochs: int, **kwargs):
        """训练开始。"""
        self.trigger("on_train_begin", epochs, **kwargs)

    def on_train_end(self, **kwargs):
        """训练结束。"""
        self.trigger("on_train_end", **kwargs)

    def on_epoch_begin(self, epoch: int, **kwargs):
        """Epoch 开始。"""
        self.trigger("on_epoch_begin", epoch, **kwargs)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """Epoch 结束。"""
        self.trigger("on_epoch_end", epoch, train_loss, val_loss, **kwargs)

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Phase 结束。"""
        self.trigger("on_phase_end", epoch, phase, loss, metrics, **kwargs)

    def on_phase_begin(self, epoch: int, phase: str, **kwargs):
        """Phase 开始。"""
        self.trigger("on_phase_begin", epoch, phase, **kwargs)

    def on_batch_begin(self, epoch: int, batch_idx: int, phase: str, **kwargs):
        """Batch 开始。"""
        self.trigger("on_batch_begin", epoch, batch_idx, phase, **kwargs)

    def on_batch_end(
        self, epoch: int, batch_idx: int, phase: str, loss: float, **kwargs
    ):
        """Batch 结束。"""
        self.trigger("on_batch_end", epoch, batch_idx, phase, loss, **kwargs)


class EarlyStoppingCallback(Callback):
    """早停回调。"""

    def __init__(
        self,
        *,
        early_stopping: EarlyStopping,
        stage: str | None = None,
    ):
        """初始化早停回调。

        Args:
            early_stopping: 早停对象
            stage: 多阶段训练的阶段名（用于区分记录的 series），单阶段为 None
        """
        self.early_stopping = early_stopping
        self.cfg = early_stopping.cfg
        self.stage = stage

        self._stop = False

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """在验证阶段结束时执行早停检查。"""
        if phase != "val":
            return
        current = self._select_monitor_value(metrics, loss)
        self.step(current, epoch, metrics)
        trainer = kwargs.get("trainer")
        metric_logger = getattr(trainer, "metric_logger", None) if trainer else None
        if metric_logger is not None:
            metric_logger.log_early_stopping(
                phase="val",
                best_score=self.early_stopping.best_score,
                num_bad_epochs=self.early_stopping.num_bad_epochs,
                best_metrics=self.early_stopping.best_metrics,
                step=getattr(trainer, "_global_step", epoch),
                epoch=epoch,
                stage=self.stage,
            )

    def step(self, current: float, epoch: int, metrics: dict | None = None) -> bool:
        """执行早停检查。

        Args:
            current: 当前指标值
            epoch: 当前 epoch
            metrics: 完整指标字典（可选）

        Returns:
            是否应该停止训练
        """
        self._stop = self.early_stopping.step(current, epoch, metrics)
        return self._stop

    def _select_monitor_value(self, metrics: dict, val_loss: float | None) -> float:
        name = (self.cfg.monitor or "auc").lower()
        value = None
        if name == "loss":
            value = float(val_loss) if val_loss is not None else None
        elif name in metrics:
            value = metrics[name]

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

    def should_stop(self, **kwargs) -> bool:
        """检查是否应该停止训练。"""
        return self._stop


class CheckpointCallback(Callback):
    """检查点保存回调。"""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        *,
        early_stopping: EarlyStopping | None = None,
        last_filename: str = "last_checkpoint.pth",
        best_filename: str | None = "best_model.pth",
        keep_best_state: bool = True,
        monitor: str | None = None,
        mode: str | None = None,
    ):
        """初始化检查点回调。

        Args:
            checkpoint_manager: 检查点管理器
            early_stopping: 早停对象（可选）
            last_filename: 最后检查点文件名
            best_filename: 最佳模型文件名（None 表示不保存最佳）
            keep_best_state: 是否缓存最佳模型 state_dict（多阶段训练可用）
            monitor: 选择最佳模型时监控的指标（可选）。默认与 ``early_stopping``
                一致；显式传入可解耦“保存最佳模型”与“早停”所监控的指标。
            mode: 最佳指标方向 ``'max'``/``'min'``（可选，默认随 monitor 或
                early_stopping）。
        """
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        self.last_filename = last_filename
        self.best_filename = best_filename
        self.keep_best_state = keep_best_state
        self._monitor_override = monitor.lower() if monitor else None
        self._mode_override = mode.lower() if mode else None

        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.best_model_state: dict | None = None

    def on_train_begin(self, epochs: int, **kwargs):
        self.best_metric = None
        self.best_epoch = None
        self.best_model_state = None

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """在验证阶段结束时保存最佳模型。"""
        trainer = kwargs.get("trainer")
        if trainer is None or phase != "val" or self.best_filename is None:
            return

        current = self._select_monitor_value(metrics, loss)
        if not self._is_better_metric(current):
            return

        self.best_metric = current
        self.best_epoch = epoch
        snapshot = self.checkpoint_manager.save_weights(
            trainer.model, self.best_filename
        )
        self.best_model_state = snapshot if self.keep_best_state else None

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """每个 epoch 结束时保存 last checkpoint。"""
        trainer = kwargs.get("trainer")
        if trainer is None:
            return
        self.checkpoint_manager.save_checkpoint(
            epoch,
            trainer.model,
            trainer.opt,
            trainer.lr_scheduler,
            early_stopping_state=self._get_early_stopping_state(),
            filename=self.last_filename,
        )

    def _monitor_name(self) -> str:
        if self._monitor_override:
            return self._monitor_override
        if self.early_stopping is None:
            return "auc"
        return (self.early_stopping.cfg.monitor or "auc").lower()

    def _select_monitor_value(self, metrics: dict, val_loss: float | None) -> float:
        name = self._monitor_name()
        value = None
        if name == "loss":
            value = float(val_loss) if val_loss is not None else None
        elif name in metrics:
            value = metrics[name]

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

    def _is_better_metric(self, current: float) -> bool:
        if self.best_metric is None:
            return True
        if self._mode_override:
            mode = self._mode_override
        elif self.early_stopping is not None:
            mode = self.early_stopping.cfg.mode
        elif self._monitor_name() in ["rmse", "loss"]:
            mode = "min"
        else:
            mode = "max"
        return (
            current > self.best_metric if mode == "max" else current < self.best_metric
        )

    def _get_early_stopping_state(self) -> dict | None:
        if self.early_stopping is None:
            return None
        state = {
            "best_score": self.early_stopping.best_score,
            "best_epoch": self.early_stopping.best_epoch,
            "num_bad_epochs": self.early_stopping.num_bad_epochs,
        }
        if self.early_stopping.best_metrics is not None:
            state["best_metrics"] = self.early_stopping.best_metrics.copy()
        return state


class MemoryCleanupCallback(Callback):
    """内存清理回调。"""

    def __init__(self, cleanup_interval: int = 5):
        """初始化内存清理回调。

        Args:
            cleanup_interval: 清理间隔（epoch 数）
        """
        self.cleanup_interval = cleanup_interval

    def on_phase_end(
        self, epoch: int, phase: str, loss: float, metrics: dict, **kwargs
    ):
        """Phase 结束时清理内存。"""
        if epoch % self.cleanup_interval == 0:
            self._cleanup_memory(phase)

    def _cleanup_memory(self, phase: str):
        """清理内存。"""
        import gc

        import torch

        if torch.cuda.is_available():
            try:
                before = torch.cuda.memory_allocated() / 1024**3  # GB
                torch.cuda.empty_cache()
                after = torch.cuda.memory_allocated() / 1024**3
                if before - after > 0.1:  # 超过100MB
                    logger.debug(
                        f"[{phase}] Cleaned up memory: {before:.2f}GB -> {after:.2f}GB"
                    )
            except Exception as e:
                logger.warning(f"Memory cleanup failed: {e}")

        # Python垃圾回收
        gc.collect()


class TestEvaluationCallback(Callback):
    """训练结束后执行测试集评估。"""

    def __init__(self, *, use_best_model: bool = True):
        self.use_best_model = use_best_model

    def on_train_end(self, **kwargs):
        trainer = kwargs.get("trainer")
        if trainer is None:
            return
        trainer._evaluate_on_test_set(use_best_model=self.use_best_model)


__all__ = [
    "Callback",
    "CallbackManager",
    "FunctionCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "MemoryCleanupCallback",
    "TestEvaluationCallback",
]
