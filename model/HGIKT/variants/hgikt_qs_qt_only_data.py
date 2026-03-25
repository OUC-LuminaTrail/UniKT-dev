"""Data preparation for HGIKT_QS_QT_Only variant."""

from typing_extensions import override
import torch

from model.HGIKT.HGIKT_data import HGIKTDataset, HGIKTModelData
from utils.core import get_logger

logger = get_logger(__name__)


class HGIKTQSQTOnlyData(HGIKTModelData):
    """Data preparation with only QS and QT edges in heterogeneous graph."""

    @override
    def prepare_data(self, args):
        fold_idx = args.fold if args.fold >= 0 else None
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        min_seq_len = self.data_src.get_metadata("min_seq_len")

        user_sequence, user_response, user_mask, _ = self.build_sequence_data(
            max_seq_len, min_seq_len
        )

        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

        # Build hetero graph with only QS and QT edges
        hetero_graph = self.build_hetero_graph(
            [
                ("question", "has", "skill"),
                ("question", "belongs_to", "template"),
            ]
        )

        logger.info(
            f"Hetero graph for QS_QT_Only: Node types: {hetero_graph.node_types}, Edge types: {hetero_graph.edge_types}"
        )

        # Build default hypergraph (though it may be bypassed in model, we keep it for pipeline consistency)
        skill_hypergraph = self.build_difficulty_weighted_hypergraph(
            ("question", "has", "skill"),
            num_difficulty_clusters=getattr(args, "num_difficulty_clusters", 3),
        )

        if fold_idx is not None:
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
