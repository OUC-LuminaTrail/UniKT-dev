"""
ABKT (Ability-Boosted Knowledge Tracing) 模型

两阶段 Boosting 知识追踪模型:
- Stage 1: K_CMF (Knowledge Module) - 基于协同矩阵分解的知识追踪
- Stage 2: GMF (Ability Module) - 基于图矩阵分解的能力建模

参考论文: ABKT: A Novel Ability-Boosted Knowledge Tracing Model
"""

from .ABKT_data import ABKTModelData
from .ABKT_model import GMF, IRT_2, K_CMF
from .ABKT_trainer import ABKTModelParams, ABKTTrainer

__all__ = [
    "ABKTModelData",
    "ABKTTrainer",
    "ABKTModelParams",
    "K_CMF",
    "GMF",
    "IRT_2",
]
