"""训练回调系统

提供训练过程中的回调机制，包括早停、检查点、内存管理等。
"""

from abc import ABC

from ..config import EarlyStopping
from ..core import get_logger

logger = get_logger(__name__)


class Callback(ABC):
    """回调基类。

    定义了训练过程中的回调接口。子类可以实现特定的回调逻辑。
    """

    def on_train_begin(self, epochs: int):
        """训练开始时调用。"""
        pass

    def on_train_end(self):
        """训练结束时调用。"""
        pass

    def on_epoch_begin(self, epoch: int):
        """Epoch 开始时调用。"""
        pass

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float):
        """Epoch 结束时调用。"""
        pass

    def on_phase_begin(self, epoch: int, phase: str):
        """Phase（train/val）开始时调用。"""
        pass

    def on_phase_end(self, epoch: int, phase: str, loss: float, metrics: dict):
        """Phase（train/val）结束时调用。"""
        pass

    def on_batch_begin(self, epoch: int, batch_idx: int, phase: str):
        """Batch 开始时调用。"""
        pass

    def on_batch_end(self, epoch: int, batch_idx: int, phase: str, loss: float):
        """Batch 结束时调用。"""
        pass

    def should_stop(self) -> bool:
        """检查是否应该停止训练。"""
        return False


class CallbackManager:
    """回调管理器。

    管理多个回调对象，按顺序触发它们的回调方法。
    """

    def __init__(self, callbacks: list):
        """初始化回调管理器。

        Args:
            callbacks: 回调对象列表
        """
        self.callbacks = callbacks
        self._stop_training = False

    def trigger(self, method_name: str, *args, **kwargs):
        """触发所有回调的指定方法。

        Args:
            method_name: 方法名
            *args: 位置参数
            **kwargs: 关键字参数
        """
        for callback in self.callbacks:
            getattr(callback, method_name)(*args, **kwargs)

    def should_stop(self) -> bool:
        """检查是否应该停止训练。

        Returns:
            是否应该停止训练
        """
        return any(cb.should_stop() for cb in self.callbacks)

    # 便捷方法
    def on_train_begin(self, epochs: int):
        """训练开始。"""
        self.trigger("on_train_begin", epochs)

    def on_epoch_begin(self, epoch: int):
        """Epoch 开始。"""
        self.trigger("on_epoch_begin", epoch)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float):
        """Epoch 结束。"""
        self.trigger("on_epoch_end", epoch, train_loss, val_loss)

    def on_phase_end(self, epoch: int, phase: str, loss: float, metrics: dict):
        """Phase 结束。"""
        self.trigger("on_phase_end", epoch, phase, loss, metrics)

    def on_phase_begin(self, epoch: int, phase: str):
        """Phase 开始。"""
        self.trigger("on_phase_begin", epoch, phase)

    def on_batch_begin(self, epoch: int, batch_idx: int, phase: str):
        """Batch 开始。"""
        self.trigger("on_batch_begin", epoch, batch_idx, phase)

    def on_batch_end(self, epoch: int, batch_idx: int, phase: str, loss: float):
        """Batch 结束。"""
        self.trigger("on_batch_end", epoch, batch_idx, phase, loss)


class EarlyStoppingCallback(Callback):
    """早停回调。"""

    def __init__(
        self,
        *,
        early_stopping: EarlyStopping,
    ):
        """初始化早停回调。

        Args:
            early_stopping: 早停对象
        """
        self.early_stopping = early_stopping
        self.cfg = early_stopping.cfg

        self._stop = False

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float):
        """Epoch 结束时检查早停条件。"""
        # 从 metrics 中选择监控值（需要在调用时传入）
        # 这里简化处理，假设 val_loss 相关
        pass

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

    def should_stop(self) -> bool:
        """检查是否应该停止训练。"""
        return self._stop


class CheckpointCallback(Callback):
    """检查点保存回调。"""

    def __init__(self, log_dir: str, save_best: bool = True):
        """初始化检查点回调。

        Args:
            log_dir: 日志目录
            save_best: 是否保存最佳模型
        """
        self.log_dir = log_dir
        self.save_best = save_best
        self.best_metric = None
        self.checkpoint_manager = None

    def on_train_begin(self, epochs: int):
        """训练开始时初始化检查点管理器。"""
        from .checkpoint import CheckpointManager

        self.checkpoint_manager = CheckpointManager(self.log_dir)

    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float):
        """Epoch 结束时保存检查点。"""
        # 这个方法需要在实际使用时与模型和优化器绑定
        pass


class MemoryCleanupCallback(Callback):
    """内存清理回调。"""

    def __init__(self, cleanup_interval: int = 5):
        """初始化内存清理回调。

        Args:
            cleanup_interval: 清理间隔（epoch 数）
        """
        self.cleanup_interval = cleanup_interval

    def on_phase_end(self, epoch: int, phase: str, loss: float, metrics: dict):
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
                        f"[{phase}] GPU内存清理: {before:.2f}GB -> {after:.2f}GB"
                    )
            except Exception as e:
                logger.warning(f"GPU内存清理失败: {e}")

        # Python垃圾回收
        gc.collect()


__all__ = [
    "Callback",
    "CallbackManager",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "MemoryCleanupCallback",
]
