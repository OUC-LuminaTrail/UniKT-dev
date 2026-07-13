"""QIKT 模型数据处理模块"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class QIKTDataset(Dataset):
    """QIKT 数据集

    提供问题序列、响应序列、掩码和多概念技能序列。

    Args:
        sequences: 问题ID序列 [N, S]
        responses: 响应序列 [N, S]
        masks: 掩码序列 [N, S]
        skills: 多概念技能ID序列 [N, S, max_concepts]，-1 表示填充
    """

    def __init__(self, sequences, responses, masks, skills):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.skills = skills

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.skills[idx], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class QIKTModelData(QuestionModelData):
    """QIKT 模型数据加载器

    从问题序列数据和问题-技能关系表中构建多概念技能序列。
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.max_concepts = None

    @override
    def prepare_data(self, rc):
        """准备训练和验证数据

        Args:
            rc: RunConfig (OmegaConf DictConfig)

        Returns:
            (train_dataset, val_dataset, test_dataset) 元组
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        user_sequence, user_response, user_mask, _ = self.load_sequence_data()

        q_matrix = self.build_relationship_matrix(("question", "has", "skill"))

        self.max_concepts = int(q_matrix.sum(axis=1).max())
        user_skills = self._build_skill_sequences(user_sequence, q_matrix)

        logger.info(
            f"QIKT data prepared: max_concepts={self.max_concepts}, "
            f"num_questions={q_matrix.shape[0]}, num_skills={q_matrix.shape[1]}"
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
                user_sequence,
                user_response,
                user_mask,
                user_skills,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("K-fold cross-validation is not enabled.")

        train_dataset = QIKTDataset(*train_data)
        val_dataset = QIKTDataset(*val_data)
        test_dataset = QIKTDataset(*test_data)

        logger.info(
            f"QIKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset

    def _build_skill_sequences(self, user_sequence, q_matrix):
        """从 Q-matrix 构建多概念技能序列

        为每个问题查找其关联的技能，构建 [N, S, max_concepts] 的技能序列。

        Args:
            user_sequence: 问题ID序列 [N, S]
            q_matrix: 问题-技能关联矩阵 [num_questions, num_skills]

        Returns:
            多概念技能序列 [N, S, max_concepts]，-1 表示填充
        """
        num_questions = q_matrix.shape[0]

        # Lookup table: question_id -> [skill_id_0, skill_id_1, ...]
        question_skills = np.full(
            (num_questions, self.max_concepts), -1, dtype=np.int64
        )
        for q_id in range(num_questions):
            related_skills = np.where(q_matrix[q_id] > 0)[0]
            for k in range(min(len(related_skills), self.max_concepts)):
                question_skills[q_id, k] = related_skills[k]

        return question_skills[user_sequence]
