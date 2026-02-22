"""HGIKT variant without template edges.

Model structure is unchanged - only data preparation is modified.
The model itself is identical to HGIKT, but the hetero_graph passed to it
doesn't include template edges.
"""

from typing import Any

from model.HGIKT.HGIKT_model import HGIKT
from utils.core import register_model


@register_model("HGIKT_NoTemplateEdges")
class HGIKT_NoTemplateEdges(HGIKT):
    """HGIKT variant with question-template edges removed.

    This is a passthrough model - the ablation is achieved entirely
    through data preparation (HGIKTNoTemplateEdgesData).

    The model structure is identical to HGIKT; only the input graph
    is modified (no template edges).
    """

    def __init__(
        self,
        args: Any,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        **kwargs: Any,
    ) -> None:
        # Initialize parent - all model components are the same
        super().__init__(args, data_metadata, hetero_metadata, **kwargs)

    # No need to override forward - model structure is the same
    # The ablation is achieved by building hetero_graph without template edges
