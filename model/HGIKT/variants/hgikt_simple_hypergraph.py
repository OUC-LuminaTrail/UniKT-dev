"""HGIKT variant with simple hypergraph (no difficulty weighting).

Model structure is unchanged - only data preparation is modified.
The hypergraph is built without difficulty clustering and edge weights.
"""

from typing import Any

from model.HGIKT.HGIKT_model import HGIKT
from utils.core import register_model


@register_model("HGIKT_SimpleHypergraph")
class HGIKT_SimpleHypergraph(HGIKT):
    """HGIKT variant with simple hypergraph (no difficulty weighting).

    This is a passthrough model - the ablation is achieved entirely
    through data preparation (HGIKTSimpleHypergraphData).

    The model structure is identical to HGIKT; only the input hypergraph
    is modified (no difficulty clustering or edge weights).
    """

    def __init__(
        self,
        args: Any,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(args, data_metadata, hetero_metadata, **kwargs)
