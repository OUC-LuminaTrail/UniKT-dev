"""Data preparation for HGIKT_QuestionSkillOnly variant.

Heterogeneous graph only contains question-skill edges.
"""

import torch
from typing_extensions import override

from model.HGIKT.HGIKT_data import HGIKTDataset, HGIKTModelData
from utils.core import get_logger

logger = get_logger(__name__)


class HGIKTQuestionSkillOnlyData(HGIKTModelData):
    """Data preparation with only question-skill edges in heterogeneous graph.

    Builds the heterogeneous graph with only the ("question", "has", "skill")
    edge type, keeping only the question-skill relationship.
    """

    @override
    def prepare_data(self, args):
        """Prepare HGIKT data with question-skill edges only.

        The hetero_graph is built with only question-skill edges.
        All other components remain the same.
        """
        fold_idx = args.fold if args.fold >= 0 else None

        user_sequence, user_response, user_mask, _ = self.load_sequence_data()



        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

        # === MODIFIED: Build hetero graph with only question-skill edges ===
        hetero_graph = self.build_hetero_graph(
            [
                ("question", "has", "skill"),
            ]
        )

        logger.info(
            f"Hetero graph built with question-skill edges only. "
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
            train_data, val_data, test_data = self.split_kfold_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            raise ValueError(
                "K-fold cross-validation fold index must be specified for HGIKTQuestionSkillOnlyData."
            )

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
