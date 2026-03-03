"""Data preparation for HGIKT_NoTemplateEdges variant.

Removes question-template edges from the heterogeneous graph.
"""

from typing_extensions import override

from model.HGIKT.HGIKT_data import HGIKTDataset, HGIKTModelData
from utils.core import get_logger

logger = get_logger(__name__)


class HGIKTNoTemplateEdgesData(HGIKTModelData):
    """Data preparation without question-template edges.

    Builds the heterogeneous graph without the ("question", "belongs_to", "template")
    edge type, effectively removing the template component from HGIKT.
    """

    @override
    def prepare_data(self, args):
        """Prepare HGIKT data without template edges.

        The hetero_graph is built without the question-template edges.
        All other components remain the same.
        """
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        # Build user sequence data
        user_sequence, user_response, user_mask, _ = self.build_sequence_data(
            args.max_seq_len
        )

        # Build question-skill relationship matrix
        import torch

        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

        # === MODIFIED: Build hetero graph WITHOUT template edges ===
        hetero_graph = self.build_hetero_graph(
            [
                ("question", "has", "skill"),
                ("skill", "related_to", "assignment"),
                # SKIPPED: ("question", "belongs_to", "template")
            ]
        )

        logger.info(
            f"Hetero graph built without template edges. "
            f"Node types: {hetero_graph.node_types}, "
            f"Edge types: {hetero_graph.edge_types}"
        )

        # Hypergraph unchanged
        skill_hypergraph = self.build_difficulty_weighted_hypergraph(
            ("question", "has", "skill"),
            num_difficulty_clusters=getattr(args, "num_difficulty_clusters", 3),
        )

        # Split data
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            logger.info(
                f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
            )
            train_data, val_data, test_data = self.split_kfold_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            raise ValueError(
                "K-fold cross-validation is required for HGIKT_NoTemplateEdges variant."
            )

        # Create datasets
        train_dataset = HGIKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = HGIKTDataset(val_data[0], val_data[1], val_data[2])
        test_dataset = HGIKTDataset(test_data[0], test_data[1], test_data[2])

        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "skill_hypergraph": skill_hypergraph,
            "hetero_graph": hetero_graph,
            "question_skill_matrix": question_skill_matrix,
        }
