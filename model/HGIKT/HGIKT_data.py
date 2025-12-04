import torch
import numpy as np
from dhg import Hypergraph
from utility.data_process.data_utility import DataSource, ModelData
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


class HGIKTModelData(ModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        r"""
        准备HGIKT模型所需的数据
        """
        fold_idx = args.fold if args.fold > 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        min_seq_len = self.data_src.get_metadata("min_seq_len")

        # 构建用户答题序列
        user_sequence, user_response, user_mask, _ = self.build_sequence_data(
            max_seq_len, min_seq_len
        )

        # 构建题目-知识点二值关联矩阵（题目行 - 知识点列）
        # 使用工具库的 build_data_matrix 直接从已处理数据生成二值矩阵，值为 0/1
        H = self.build_data_matrix(("question", "has", "skill"), value_type="binary")

        # 使用 dhg 框架构建超图
        hypergraph = self.build_dhg_hypergraph(H)

        # 构建异构图
        hetero_graph = self.build_hetero_graph(
            [
                (
                    "question",
                    "has",
                    "skill",
                )
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
            train_data, val_data = self.get_kfold_split_data(
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
        return train_dataloader, val_dataloader, hypergraph, hetero_graph

    def build_dhg_hypergraph(self, H):
        """使用 dhg 框架从关联矩阵 H 构建超图
        
        其中：
            - H 是关联矩阵（题目 × 知识点）
            - 每个知识点作为一个超边，连接拥有该知识点的所有题目
            - W_e 默认为单位矩阵（超边权重为1）

        Args:
            H: 关联矩阵，形状为 (num_questions, num_skills)
               H[i,j] = 1 表示题目 i 包含知识点 j

        Returns:
            dhg.Hypergraph: dhg 框架的超图对象
        """
        from tqdm import tqdm

        num_questions = H.shape[0]  # 顶点数（题目数）

        # 将关联矩阵转换为超边列表
        rows, cols = np.nonzero(H)

        # 按列（知识点）分组，每个知识点对应一个超边
        # 使用字典收集每个超边（知识点）包含的顶点（题目）
        edge_dict = {}
        for question_idx, skill_idx in tqdm(zip(rows, cols), total=len(rows), desc="Building hyperedges"):
            if skill_idx not in edge_dict:
                edge_dict[skill_idx] = []
            edge_dict[skill_idx].append(int(question_idx))
        
        # 转换为超边列表（过滤空超边）
        e_list = [vertices for vertices in edge_dict.values() if len(vertices) > 0]

        # 使用 dhg 框架创建超图
        hypergraph = Hypergraph(num_v=num_questions, e_list=e_list)

        print("Hypergraph constructed:")
        print(f"  - Number of vertices (questions): {hypergraph.num_v}")
        print(f"  - Number of hyperedges (skills): {hypergraph.num_e}")

        return hypergraph
