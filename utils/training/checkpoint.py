"""检查点管理模块

提供模型检查点的保存和加载功能。
"""

import os

import torch

from ..core import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """检查点管理器。

    职责：
    1. 保存模型检查点
    2. 加载模型检查点
    3. 管理检查点文件

    Example:
        >>> ckpt_mgr = CheckpointManager(log_dir="./runs/exp1")
        >>> ckpt_mgr.save_checkpoint(
        ...     epoch=10,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     filename="checkpoint.pth"
        ... )
    """

    def __init__(self, log_dir: str):
        """初始化检查点管理器。

        Args:
            log_dir: 日志目录，用于保存检查点
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None = None,
        additional_state: dict | None = None,
        early_stopping_state: dict | None = None,
        filename: str = "checkpoint.pth",
    ):
        """保存完整检查点。

        Args:
            epoch: 当前 epoch
            model: PyTorch 模型
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
            additional_state: 额外的状态信息（可选）
            early_stopping_state: 早停状态（可选）
            filename: 文件名
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if early_stopping_state is not None:
            state["early_stopping_state"] = early_stopping_state

        if additional_state:
            state.update(additional_state)

        filepath = os.path.join(self.log_dir, filename)
        torch.save(state, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def save_weights(self, model: torch.nn.Module, filename: str = "model.pth"):
        """仅保存模型权重。

        Args:
            model: PyTorch 模型
            filename: 文件名
        """
        filepath = os.path.join(self.log_dir, filename)
        torch.save(model.state_dict(), filepath)
        logger.info(f"Model weights saved to {filepath}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: object | None = None,
        early_stopping: object | None = None,
        device: torch.device = None,
    ) -> dict:
        """加载检查点。

        Args:
            checkpoint_path: 检查点文件路径
            model: PyTorch 模型
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
            early_stopping: 早停对象（可选）
            device: 计算设备（可选）

        Returns:
            检查点字典
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        if device is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path)

        # 加载模型状态
        model.load_state_dict(checkpoint["model_state_dict"])

        # 加载优化器状态
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 加载调度器状态
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 加载早停状态
        if early_stopping is not None and "early_stopping_state" in checkpoint:
            es_state = checkpoint["early_stopping_state"]
            early_stopping.best_score = es_state.get("best_score")
            early_stopping.best_epoch = es_state.get("best_epoch")
            early_stopping.num_bad_epochs = es_state.get("num_bad_epochs", 0)

        logger.info("Checkpoint loaded successfully")
        return checkpoint


__all__ = ["CheckpointManager"]
