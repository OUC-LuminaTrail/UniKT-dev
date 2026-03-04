"""GKT (Graph-based Knowledge Tracing) 数据处理模块"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData

logger = get_logger(__name__)


class GKTDataset(Dataset):
    """GKT 数据集

    处理输入数据，将其转换为模型所需的格式。

    Args:
        sequences: 概念ID序列
        responses: 响应序列
        masks: 掩码序列
        late_group_ids: 题目级分组ID（可选，窗口测试时使用）
        true_labels: 真实标签序列（可选，窗口测试时使用）

    Returns:
        训练/验证模式 (无 late_group_ids/true_labels):
            (sequence, response, mask) 元组
        窗口测试模式 (有 late_group_ids/true_labels):
            (sequence, response, mask, late_group_id, true_labels) 元组
    """

    def __init__(
        self, sequences, responses, masks, late_group_ids=None, true_labels=None
    ):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.late_group_ids = late_group_ids
        self.true_labels = true_labels

        # 判断是否为窗口测试模式
        self._is_window_mode = late_group_ids is not None and true_labels is not None

    def __len__(self) -> int:
        """返回数据集长度

        Returns:
            数据集的样本数量
        """
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)

        if self._is_window_mode:
            late_group_id = torch.tensor(self.late_group_ids[idx], dtype=torch.long)
            true_labels = torch.tensor(self.true_labels[idx], dtype=torch.long)
            return sequence, response, mask, late_group_id, true_labels

        return sequence, response, mask


class GKTModelData(SkillModelData):
    """GKT 模型数据加载器

    负责处理数据的预处理和加载，为GKT模型提供训练和验证数据。
    继承自 SkillModelData，使用技能序列构建方式。

    Args:
        data_src: 数据源实例，包含原始数据
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args: Any) -> tuple:
        """准备训练和验证数据

        Args:
            args: 模型参数配置

        Returns:
            训练数据集、验证数据集和窗口验证数据集
        """
        fold_idx = args.fold if args.fold >= 0 else None

        # 构建用户答题序列
        user_sequence, user_response, user_mask, _ = self.build_sequence_data(
            args.max_seq_len
        )

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
            train_data, val_data, _ = self.split_kfold_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            raise ValueError("K-fold cross-validation is not enabled.")

        # 构建 windowlate 评估数据
        window_test_data = self.create_windowlate_iterable_dataset(args.max_seq_len)

        # 构建模型数据集
        train_dataset = GKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = GKTDataset(val_data[0], val_data[1], val_data[2])
        test_dataset = DataLoader(
            window_test_data, batch_size=args.batch_size, shuffle=False
        )

        logger.debug(
            f"GKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    def build_transition_graph(self) -> torch.Tensor:
        """构建基于概念转移的图

        基于数据集中概念序列的转移模式构建邻接矩阵

        Returns:
            图邻接矩阵，形状为 [num_skills, num_skills]
        """
        data = self.data_src.get_sequence_data()
        num_skills = self.data_src.get_metadata("num_skills")

        # 构建转移矩阵
        graph = np.zeros((num_skills, num_skills))

        for row in data.itertuples():
            # 获取问题到技能的映射
            question_data = self.data_src.get_question_data()
            q_to_skills = (
                question_data.groupby("question")["skill"].apply(list).to_dict()
            )

            # 获取用户的技能序列
            skills = q_to_skills.get(row.question, [])
            for skill in skills:
                if skill != -1:
                    # 简单转移计数
                    pass  # TODO: 实现完整的转移图构建逻辑

        # 对角线置零
        np.fill_diagonal(graph, 0)

        # 行归一化
        rowsum = np.array(graph.sum(1))

        def inv(x):
            return 1.0 / x if x != 0 else 0.0

        inv_func = np.vectorize(inv)
        r_inv = inv_func(rowsum).flatten()
        r_mat_inv = np.diag(r_inv)
        graph = r_mat_inv.dot(graph)

        return torch.from_numpy(graph).float()
