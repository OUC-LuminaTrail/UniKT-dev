"""
SGKT (Session Graph-based Knowledge Tracing) model.

Implements dual graph architecture:
- HRG (Heterogeneous Relation Graph) with GCNConv
- SG (Session Graph) with GatedGraphConv
"""

from .SGKT_data import SGKTModelData, SGKTDataset
from .SGKT_model import SGKT
from .SGKT_trainer import SGKTTrainer, SGKTModelParams

__all__ = [
    "SGKTModelData",
    "SGKTDataset",
    "SGKT",
    "SGKTTrainer",
    "SGKTModelParams",
]
