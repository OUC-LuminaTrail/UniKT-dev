"""extraKT 模型模块

包含extraKT (Extra Knowledge Tracing) 模型的实现和训练相关组件。
"""

from .extraKT_model import extraKT
from .extraKT_trainer import extraKTTrainer

__all__ = ["extraKT", "extraKTTrainer"]
