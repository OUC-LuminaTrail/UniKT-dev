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

    @staticmethod
    def load_weights(
        path: str,
        model: torch.nn.Module,
        device: torch.device | None = None,
    ) -> dict | None:
        """Load model weights from a checkpoint file.

        Handles both formats:
        - Plain state_dict (saved by ``save_weights``)
        - Full checkpoint dict with ``"model_state_dict"`` key (saved by ``save_checkpoint``)

        Args:
            path: Path to the checkpoint file.
            model: PyTorch model to load weights into.
            device: Target device for ``map_location`` (optional).

        Returns:
            The full checkpoint dict if the file was a full checkpoint,
            ``None`` if it was a plain state_dict.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_kw = {"map_location": device} if device is not None else {}
        logger.info(f"Loading model weights from {path}...")
        raw = torch.load(path, **map_kw)

        if isinstance(raw, dict) and "model_state_dict" in raw:
            model.load_state_dict(raw["model_state_dict"])
            logger.info(
                f"Loaded from full checkpoint (epoch {raw.get('epoch', 'unknown')})"
            )
            return raw

        model.load_state_dict(raw)
        logger.debug("Loaded from plain state_dict")
        return None

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
            early_stopping.best_metrics = es_state.get("best_metrics")

        logger.info("Checkpoint loaded successfully")
        return checkpoint


__all__ = ["CheckpointManager"]
