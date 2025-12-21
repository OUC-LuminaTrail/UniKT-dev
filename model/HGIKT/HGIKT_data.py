import torch
from utils.data_process import DataSource
from utils.net_data import GraphModelData
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from typing_extensions import override


class HGIKTDataset(Dataset):
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


class HGIKTModelData(GraphModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        r"""
        准备HGIKT模型所需的数据
        """
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        min_seq_len = self.data_src.get_metadata("min_seq_len")

        # 构建用户答题序列
        user_sequence, user_response, user_mask, _ = self.build_sequence_data(
            max_seq_len, min_seq_len
        )

        # 构建问题-技能关联矩阵，并转换为torch张量
        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

        # 构建基于问题模板超图
        template_hypergraph = self.build_hyper_graph(
            ("question", "belongs_to", "template")
        )

        # 构建异构图
        hetero_graph = self.build_hetero_graph(
            [
                # 问题技能图
                (
                    "question",
                    "has",
                    "skill",
                ),
                # 技能作业图
                (
                    "skill",
                    "related_to",
                    "assignment",
                ),
                # 题目模板图
                (
                    "question",
                    "belongs_to",
                    "template",
                ),
            ]
        )

        # 划分训练集和验证集
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            print(f"Using K-fold cross-validation: fold {fold_idx}/{kfold_n_splits}")
            train_data, val_data = self.split_kfold_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            train_data, val_data = self.split_data(
                user_sequence, user_response, user_mask
            )

        # 构建模型数据集
        train_dataset = HGIKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = HGIKTDataset(val_data[0], val_data[1], val_data[2])

        # 构建数据加载器
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        # 返回数据
        return_data = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "template_hypergraph": template_hypergraph,
            "hetero_graph": hetero_graph,
            "question_skill_matrix": question_skill_matrix,
        }

        return return_data
