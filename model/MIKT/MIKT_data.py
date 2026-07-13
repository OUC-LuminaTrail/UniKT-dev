"""MIKT 模型数据处理模块"""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class MIKTDataset(Dataset):
    """MIKT 数据集

    返回 (sequence, response, mask) 元组。
    """

    def __init__(self, sequences, responses, masks):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.bool),
        )

    def __len__(self):
        return len(self.sequences)


class MIKTModelData(QuestionModelData):
    """MIKT 模型数据加载器

    使用问题级序列数据，构建问题-技能关联矩阵供模型使用。
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练和验证数据

        Args:
            rc: RunConfig (OmegaConf DictConfig)

        Returns:
            (train_dataset, val_dataset, test_dataset, question_skill_matrix)
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        user_sequence, user_response, user_mask, _ = self.load_sequence_data()

        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

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
            raise ValueError("fold_idx must be specified for K-fold cross-validation")

        train_dataset = MIKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = MIKTDataset(val_data[0], val_data[1], val_data[2])
        test_dataset = MIKTDataset(test_data[0], test_data[1], test_data[2])

        logger.debug(
            f"MIKT data prepared: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset, question_skill_matrix
