import torch
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

        # 知识点超图：支持普通超图、难度加权超图和对比学习超图
        # 注意：难度加权超图和对比学习超图可以共存！
        use_contrastive = getattr(args, 'use_contrastive_hypergraph', False)
        use_difficulty = getattr(args, 'use_difficulty_weighted_hypergraph', False)
        
        # 初始化变量
        pos_hypergraph = None
        neg_hypergraph = None
        pos_edge_weights = None
        neg_edge_weights = None
        
        # 1. 构建主超图（难度加权 或 普通）
        if use_difficulty:
            # 使用难度加权超图作为主超图
            skill_hypergraph, edge_weights = self.build_difficulty_weighted_hypergraph(
                ("question", "has", "skill"),
                num_difficulty_clusters=getattr(args, 'num_difficulty_clusters', 3)
            )
        else:
            # 使用普通超图作为主超图
            skill_hypergraph = self.build_hypergraph(("question", "has", "skill"))
            edge_weights = None
        
        # 2. 如果启用对比学习，额外构建正负超图
        if use_contrastive:
            # 构建对比学习超图（正负超边）
            (
                pos_hypergraph, 
                neg_hypergraph, 
                pos_edge_weights, 
                neg_edge_weights
            ) = self.build_contrastive_hypergraph(
                ("question", "has", "skill"),
                easy_threshold=getattr(args, 'contrastive_easy_threshold', 0.3),
                hard_threshold=getattr(args, 'contrastive_hard_threshold', 0.6),
                min_samples=getattr(args, 'contrastive_min_samples', 5)
            )
            
            # 如果同时启用了难度加权，打印组合模式提示
            if use_difficulty:
                print("使用组合模式: 难度加权超图 + 对比学习超图")
                print(f"  - 主超图: 难度加权超图（{skill_hypergraph.num_e}个超边）")
                print(f"  - 对比超图: 正超边{pos_hypergraph.num_e}个，负超边{neg_hypergraph.num_e}个")

        # 构建异构图
        hetero_graph = self.build_hetero_graph(
            [
                (
                    "question",
                    "has",
                    "skill",
                ),
                (
                    "skill",
                    "related_to",
                    "assignment",
                ),
                (
                    "assignment",
                    "contains",
                    "question",
                ),
                (
                    "question",
                    "belongs_to",
                    "template",
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
        
        # 返回数据时包含边权重信息和对比超图
        return_data = {
            'train_dataloader': train_dataloader,
            'val_dataloader': val_dataloader,
            'skill_hypergraph': skill_hypergraph,
            'hetero_graph': hetero_graph,
            'edge_weights': edge_weights,
            'pos_hypergraph': pos_hypergraph,
            'neg_hypergraph': neg_hypergraph,
            'pos_edge_weights': pos_edge_weights,
            'neg_edge_weights': neg_edge_weights
        }
        
        return return_data
