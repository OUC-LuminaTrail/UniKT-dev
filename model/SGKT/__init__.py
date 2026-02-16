"""
SGKT (Session Graph-based Knowledge Tracing) model.

Implements dual graph architecture:
- HRG (Heterogeneous Relation Graph) with GCNConv
- SG (Session Graph) with GatedGraphConv
"""

from .SGKT_data import SGKTDataset, SGKTModelData
from .SGKT_model import SGKT
from .SGKT_trainer import SGKTModelParams, SGKTTrainer

__all__ = [
    "SGKTModelData",
    "SGKTDataset",
    "SGKT",
    "SGKTTrainer",
    "SGKTModelParams",
]
