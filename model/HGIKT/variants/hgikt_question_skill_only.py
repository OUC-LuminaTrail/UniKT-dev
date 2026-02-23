"""HGIKT variant with question-skill edges only in heterogeneous graph.

Model structure is unchanged - only data preparation is modified.
The hetero_graph passed to it only contains question-skill edges.
"""

from typing import Any

from model.HGIKT.HGIKT_model import HGIKT
from utils.core import register_model


@register_model("HGIKT_QuestionSkillOnly")
class HGIKT_QuestionSkillOnly(HGIKT):
    """HGIKT variant with only question-skill edges in heterogeneous graph.

    This is a passthrough model - the ablation is achieved entirely
    through data preparation (HGIKTQuestionSkillOnlyData).

    The model structure is identical to HGIKT; only the input graph
    is modified (question-skill edges only).
    """

    def __init__(
        self,
        args: Any,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(args, data_metadata, hetero_metadata, **kwargs)
