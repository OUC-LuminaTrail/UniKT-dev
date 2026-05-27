"""DKVMN 模型模块

包含DKVMN (Dynamic Key-Value Memory Networks) 模型的实现和训练相关组件。
"""

from .DKVMN_model import DKVMN
from .DKVMN_trainer import DKVMNTrainer

__all__ = ["DKVMN", "DKVMNTrainer"]
