"""数据配置模块

提供 DataLoader 配置和优化函数，从 dataloader_config.py 迁移。
"""

import os
from dataclasses import dataclass
from typing import Literal

from ..core import get_logger

logger = get_logger(__name__)

# 类型定义
NumWorkersType = int | Literal["auto"]


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


def create_optimized_dataloader(
    dataset,
    batch_size: int = 128,
    shuffle: bool = True,
    config: DataLoaderConfig | None = None,
    device=None,
    **kwargs,
):
    """创建已优化的 DataLoader。

    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        config: DataLoader 配置（默认使用 DataLoaderConfig()）
        device: 计算设备（用于确定 pin_memory）
        **kwargs: 其他传递给 DataLoader 的参数（优先级高于 config）

    Returns:
        优化后的 DataLoader

    Example:
        >>> from utils.config import DataLoaderConfig, create_optimized_dataloader
        >>> config = DataLoaderConfig(num_workers=4, pin_memory=True)
        >>> loader = create_optimized_dataloader(
        ...     dataset,
        ...     batch_size=64,
        ...     shuffle=True,
        ...     config=config,
        ...     device=torch.device("cuda")
        ... )
    """
    from torch.utils.data import DataLoader

    # 使用默认配置
    if config is None:
        config = DataLoaderConfig()

    # 必须指定设备信息
    assert device is not None, (
        "Device information is required to determine pin_memory setting."
    )

    # 确定 pin_memory
    is_cuda = device.type == "cuda"
    pin_memory = config.pin_memory and is_cuda

    # 获取 num_workers
    num_workers = config.get_num_workers()

    # 准备 DataLoader 参数
    # kwargs 中的参数优先级高于 config
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": config.prefetch_factor if num_workers > 0 else None,
        "persistent_workers": config.persistent_workers if num_workers > 0 else False,
    }

    # 移除无效参数（prefetch_factor 仅在 num_workers > 0 时有效）
    if loader_kwargs["prefetch_factor"] is None:
        del loader_kwargs["prefetch_factor"]

    # 用 kwargs 覆盖默认参数
    loader_kwargs.update(kwargs)

    # 创建 DataLoader
    loader = DataLoader(dataset, **loader_kwargs)

    # 记录优化信息
    logger.debug(
        f"Created optimized DataLoader: num_workers={loader_kwargs.get('num_workers')}, "
        f"pin_memory={loader_kwargs.get('pin_memory')}, "
        f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'N/A')}, "
        f"persistent_workers={loader_kwargs.get('persistent_workers', 'N/A')}"
    )

    return loader


__all__ = [
    "DataLoaderConfig",
    "create_optimized_dataloader",
]
