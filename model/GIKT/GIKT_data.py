import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.net_data import GraphModelData

logger = get_logger(__name__)


class GIKTDataset(Dataset):
    def __init__(self, sequences, responses, masks):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class GIKTModelData(GraphModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        r"""
        准备GIKT模型所需的数据
        """
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        min_seq_len = self.data_src.get_metadata("min_seq_len")

        # 构建用户答题序列
        user_sequence, user_response, user_mask, _ = self.build_sequence_data(
            max_seq_len, min_seq_len
        )

        # 构建异构图
        graph = self.build_hetero_graph(
            [
                (
                    "question",
                    "has",
                    "skill",
                )
            ]
        )

        # 构建问题-技能关联矩阵，并转换为torch张量
        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

        # 划分训练集和验证集
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

        # 构建模型数据集
        train_dataset = GIKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = GIKTDataset(val_data[0], val_data[1], val_data[2])

        return train_dataset, val_dataset, graph, question_skill_matrix
