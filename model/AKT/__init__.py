"""AKT 模型模块

包含AKT (Attentive Knowledge Tracing) 模型的实现和训练相关组件。
"""

from .AKT_model import AKT
from .AKT_trainer import AKTTrainer

__all__ = ["AKT", "AKTTrainer"]
