"""Data preparation for HGIKT_Hyper_Only_Unweighted variant."""

from typing_extensions import override
import torch

from model.HGIKT.HGIKT_data import HGIKTDataset, HGIKTModelData
from utils.core import get_logger

logger = get_logger(__name__)


class HGIKTHyperOnlyUnweightedData(HGIKTModelData):
    """Data preparation with simple hypergraph (no difficulty weighting)."""

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

        # Full hetero graph (may be bypassed in model, but we build it for pipeline consistency)
        hetero_graph = self.build_hetero_graph(
            [
                ("question", "has", "skill"),
                ("skill", "related_to", "assignment"),
                ("question", "belongs_to", "template"),
            ]
        )

        # Build simple hypergraph (no difficulty weighting)
        skill_hypergraph = self.build_hyper_graph(("question", "has", "skill"))

        logger.info(
            f"Simple hypergraph built for HyperOnlyUnweighted. Vertices: {skill_hypergraph.num_v}, Hyperedges: {skill_hypergraph.num_e}"
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
