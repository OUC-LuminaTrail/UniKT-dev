from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData

logger = get_logger(__name__)


class DKTDataset(Dataset):
    """DKT 数据集

    处理输入数据，将其转换为模型所需的格式。

    Args:
        sequences: 概念ID序列
        responses: 响应序列
        masks: 掩码序列
    """

    def __init__(self, sequences, responses, masks):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks

    def __len__(self) -> int:
        """返回数据集长度

        Returns:
            数据集的样本数量
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取单个样本

        Args:
            idx: 样本索引

        Returns:
            包含 (sequence, response, mask) 的元组
            - sequence: 概念ID序列
            - response: 响应序列
            - mask: 掩码序列
        """
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)

        return sequence, response, mask


class DKTWindowDataset(Dataset):
    """DKT 窗口测试数据集。

    在普通序列数据的基础上，额外返回 late_group_id，
    用于 windowlate 指标按组聚合。
    """

    def __init__(self, sequences, responses, masks, late_group_ids):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.late_group_ids = late_group_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        late_group_id = torch.tensor(self.late_group_ids[idx], dtype=torch.long)
        return sequence, response, mask, late_group_id


class DKTModelData(SkillModelData):
    """DKT 模型数据加载器

    负责处理数据的预处理和加载，为DKT模型提供训练和验证数据。
    继承自SkillModelData，使用技能序列构建方式。

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
        window_test_data = self.build_windowlate_data(args.max_seq_len)

        # 构建模型数据集
        train_dataset = DKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = DKTDataset(val_data[0], val_data[1], val_data[2])
        test_dataset = DKTWindowDataset(
            window_test_data[0],
            window_test_data[1],
            window_test_data[2],
            window_test_data[4],
        )

        logger.debug(
            f"DKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
