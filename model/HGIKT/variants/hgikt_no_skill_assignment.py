"""HGIKT variant without skill-assignment edges.

Model structure is unchanged - only data preparation is modified.
The hetero_graph passed to it doesn't include skill-assignment edges.
"""

from typing import Any

from model.HGIKT.HGIKT_model import HGIKT
from utils.core import register_model


@register_model("HGIKT_NoSkillAssignment")
class HGIKT_NoSkillAssignment(HGIKT):
    """HGIKT variant with skill-assignment edges removed.

    This is a passthrough model - the ablation is achieved entirely
    through data preparation (HGIKTNoSkillAssignmentData).

    The model structure is identical to HGIKT; only the input graph
    is modified (no skill-assignment edges).
    """

    def __init__(
        self,
        args: Any,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(args, data_metadata, hetero_metadata, **kwargs)
