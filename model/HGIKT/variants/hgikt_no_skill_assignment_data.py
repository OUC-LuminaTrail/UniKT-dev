"""Data preparation for HGIKT_NoSkillAssignment variant.

Removes skill-assignment edges from the heterogeneous graph.
"""

from typing_extensions import override

from model.HGIKT.HGIKT_data import HGIKTDataset, HGIKTModelData
from utils.core import get_logger

logger = get_logger(__name__)


class HGIKTNoSkillAssignmentData(HGIKTModelData):
    """Data preparation without skill-assignment edges.

    Builds the heterogeneous graph without the ("skill", "related_to", "assignment")
    edge type, effectively removing the skill-assignment relationship from HGIKT.
    """

    @override
    def prepare_data(self, args):
        """Prepare HGIKT data without skill-assignment edges.

        The hetero_graph is built without the skill-assignment edges.
        All other components remain the same.
        """
        fold_idx = args.fold if args.fold >= 0 else None

        user_sequence, user_response, user_mask, _ = self.build_sequence_data(
            args.max_seq_len
        )

        import torch

        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

        # === MODIFIED: Build hetero graph WITHOUT skill-assignment edges ===
        hetero_graph = self.build_hetero_graph(
            [
                ("question", "has", "skill"),
                ("question", "belongs_to", "template"),
            ]
        )

        logger.info(
            f"Hetero graph built without skill-assignment edges. "
            f"Node types: {hetero_graph.node_types}, "
            f"Edge types: {hetero_graph.edge_types}"
        )

        skill_hypergraph = self.build_difficulty_weighted_hypergraph(
            ("question", "has", "skill"),
            num_difficulty_clusters=getattr(args, "num_difficulty_clusters", 3),
        )

        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            logger.info(
                f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
            )
            train_data, val_data = self.split_kfold_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            train_data, val_data = self.split_data(
                user_sequence, user_response, user_mask
            )

        train_dataset = HGIKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = HGIKTDataset(val_data[0], val_data[1], val_data[2])

        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "skill_hypergraph": skill_hypergraph,
            "hetero_graph": hetero_graph,
            "question_skill_matrix": question_skill_matrix,
        }
