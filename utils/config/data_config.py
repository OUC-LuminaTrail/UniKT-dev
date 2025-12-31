"""数据配置模块

提供 DataLoader 配置和优化函数，从 dataloader_config.py 迁移。
"""

import os
from dataclasses import dataclass
from typing import Literal, Union, Optional
from ..core import get_logger

logger = get_logger(__name__)

# 类型定义
NumWorkersType = Union[int, Literal["auto"]]


@dataclass
class DataLoaderConfig:
    """DataLoader 配置类。

    Attributes:
        num_workers: 数据加载的工作进程数。
                     "auto" 表示自动设置为 CPU 核心数和 8 的较小值。
                     0 表示禁用多进程。
        pin_memory: 是否将张量固定在 CUDA 内存中（仅 CUDA 时有效）。
        prefetch_factor: 每个工作进程的预取批次数（仅 num_workers > 0 时有效）。
        persistent_workers: 是否在工作进程之间保持工作进程的存活（PyTorch >= 1.7）。
    """

    num_workers: NumWorkersType = "auto"
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    def get_num_workers(self, max_limit: int = 8) -> int:
        """获取实际的 num_workers 值。

        Args:
            max_limit: 最大工作进程数限制

        Returns:
            实际的 num_workers 值
        """
        if self.num_workers == "auto":
            cpu_count = os.cpu_count() or 1
            return min(cpu_count, max_limit)
        return self.num_workers


def optimize_dataloader(
    loader,
    config: Optional[DataLoaderConfig] = None,
    device=None,
) -> None:
    """优化 DataLoader 的性能参数。

    此函数直接修改传入的 DataLoader 对象，而不是返回新对象。

    Args:
        loader: 要优化的 DataLoader 对象
        config: DataLoader 配置（默认使用 DataLoaderConfig()）
        device: 计算设备（用于确定 pin_memory）

    支持的 DataLoader 属性:
        - num_workers: 数据加载进程数
        - pin_memory: 固定内存（CUDA 加速）
        - prefetch_factor: 预取因子
        - persistent_workers: 保持工作进程存活
    """
    # 使用默认配置
    if config is None:
        config = DataLoaderConfig()

    # 确定 pin_memory
    is_cuda = device is not None and device.type == "cuda"
    pin_memory = config.pin_memory and is_cuda

    # 获取 num_workers
    num_workers = config.get_num_workers()

    # 应用配置
    if hasattr(loader, "num_workers"):
        loader.num_workers = num_workers

    if hasattr(loader, "pin_memory"):
        loader.pin_memory = pin_memory

    if num_workers > 0:
        if hasattr(loader, "prefetch_factor"):
            loader.prefetch_factor = config.prefetch_factor

    # 打印优化信息（仅在首次优化时）
    if not hasattr(loader, "_optimized_config"):
        logger.debug(
            f"DataLoader optimized: num_workers={num_workers}, pin_memory={pin_memory}"
        )
        loader._optimized_config = True  # 标记已优化


def create_optimized_dataloader(
    dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    config: Optional[DataLoaderConfig] = None,
    device=None,
    **kwargs,
):
    """创建已优化的 DataLoader。

    这是一个便捷函数，结合了 DataLoader 创建和优化。

    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        config: DataLoader 配置
        device: 计算设备
        **kwargs: 其他传递给 DataLoader 的参数

    Returns:
        优化后的 DataLoader
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )

    optimize_dataloader(loader, config, device)

    return loader


@dataclass
class KFoldDataLoaderConfig:
    """K 折交叉验证专用的 DataLoader 配置。

    Attributes:
        train_config: 训练集 DataLoader 配置
        val_config: 验证集 DataLoader 配置
        shared_config: 共享的配置参数（优先级较低）
    """

    train_config: Optional[DataLoaderConfig] = None
    val_config: Optional[DataLoaderConfig] = None
    shared_config: Optional[DataLoaderConfig] = None

    def __post_init__(self):
        """初始化默认配置。"""
        if self.train_config is None:
            self.train_config = DataLoaderConfig(num_workers="auto", pin_memory=True)

        if self.val_config is None:
            # 验证集通常不需要那么多 workers
            self.val_config = DataLoaderConfig(
                num_workers=0,  # 验证集不需要多进程
                pin_memory=True,
            )

        if self.shared_config is None:
            self.shared_config = DataLoaderConfig()


def optimize_kfold_dataloaders(
    train_loader,
    val_loader,
    config: Optional[KFoldDataLoaderConfig] = None,
    device=None,
) -> None:
    """优化 K 折交叉验证使用的 DataLoader 对。

    Args:
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
        config: K 折配置
        device: 计算设备
    """
    if config is None:
        config = KFoldDataLoaderConfig()

    # 优化训练集
    optimize_dataloader(train_loader, config.train_config, device)

    # 优化验证集
    optimize_dataloader(val_loader, config.val_config, device)


__all__ = [
    "DataLoaderConfig",
    "KFoldDataLoaderConfig",
    "optimize_dataloader",
    "create_optimized_dataloader",
    "optimize_kfold_dataloaders",
]
