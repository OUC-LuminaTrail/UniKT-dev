"""随机种子设置模块

提供统一的随机种子设置功能，确保实验可复现。
"""

from typing import Optional


def seed_everything(seed: Optional[int], deterministic: bool = True) -> Optional[int]:
    """设置随机种子以确保结果可复现。

    此函数会设置 Python、NumPy 和 PyTorch 的随机种子，
    并可选地启用确定性模式。

    Args:
        seed: 随机种子值。如果为 None，则不设置任何种子。
        deterministic: 是否启用确定性模式。启用后会导致性能下降，
            但能确保完全可复现的结果。默认为 True。

    Returns:
        设置的种子值，如果 seed 为 None 则返回 None

    Example:
        >>> seed_everything(42)  # 设置种子为 42
        >>> seed_everything(None)  # 不设置种子
    """
    import os
    import random

    import numpy as np
    import torch

    if seed is None:
        return None

    # 设置环境变量
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 设置各个库的随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 设置 CUDA 相关的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 启用确定性模式
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ 支持 use_deterministic_algorithms
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # 旧版本 PyTorch 不支持 warn_only 参数
                torch.use_deterministic_algorithms(True)

        # 设置 CUDA 工作空间配置
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return seed
