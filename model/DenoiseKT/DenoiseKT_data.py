from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class DenoiseKTDataset(Dataset):
    """DenoiseKT 问题级数据集。"""

    def __init__(self, questions, responses, masks):
        self.questions = questions
        self.responses = responses
        self.masks = masks

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        question = torch.tensor(self.questions[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        return question, response, mask


class DenoiseKTModelData(QuestionModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    def build_question_concepts(self) -> torch.Tensor:
        """构建题目-概念查表 ``[num_questions, max_concepts]``（long，-1 填充）。"""
        qs_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )  # [num_q, num_c]
        num_q = qs_matrix.shape[0]
        skill_counts = qs_matrix.sum(axis=1)
        max_concepts = int(skill_counts.max()) if num_q > 0 else 1
        max_concepts = max(max_concepts, 1)

        question_concepts = np.full((num_q, max_concepts), -1, dtype=np.int64)
        for q in range(num_q):
            skills = np.where(qs_matrix[q] == 1)[0]
            k = min(len(skills), max_concepts)
            if k > 0:
                question_concepts[q, :k] = skills[:k]

        logger.info(
            f"DenoiseKT question_concepts: num_q={num_q}, max_concepts={max_concepts}"
        )
        return torch.from_numpy(question_concepts)

    def build_question_graph(self) -> torch.Tensor:
        """构建题目-概念邻接矩阵（行归一化），嵌入 ``[num_q, num_q]`` 供 GCN 使用。"""
        num_q = self.data_src.get_metadata("num_questions")
        num_c = self.data_src.get_metadata("num_skills")
        if num_c > num_q:
            raise ValueError(
                f"DenoiseKT requires num_skills ({num_c}) <= num_questions ({num_q}) "
                f"to embed the question-concept matrix into [num_q, num_q]."
            )

        qs_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        ).astype(np.float32)  # [num_q, num_c]

        # Nonzero positions (q, c): question q contains concept c.
        q_idx, c_idx = np.nonzero(qs_matrix)
        # Row-normalized: each nonzero entry = 1 / (concept count of that question).
        row_sum = qs_matrix.sum(axis=1)
        row_sum[row_sum == 0] = 1.0
        values = 1.0 / row_sum[q_idx]

        indices = torch.from_numpy(np.stack([q_idx, c_idx])).long()
        sparse_adj = torch.sparse_coo_tensor(
            indices, torch.from_numpy(values).float(), size=(num_q, num_q)
        ).coalesce()

        logger.info(
            f"DenoiseKT question-concept graph: {sparse_adj.shape[0]}x{sparse_adj.shape[1]}, "
            f"nnz={sparse_adj._nnz()}"
        )
        return sparse_adj

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练、验证、测试数据，以及题目-概念查表和题目图。"""
        fold_idx = rc.data.fold
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        if not (0 <= fold_idx < kfold_n_splits):
            raise ValueError(
                f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
            )
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        user_question, user_response, user_mask, _ = self.load_sequence_data()

        train_data, val_data, test_data = self.split_kfold_data(
            user_question, user_response, user_mask, fold_idx=fold_idx
        )

        train_dataset = DenoiseKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = DenoiseKTDataset(val_data[0], val_data[1], val_data[2])
        test_dataset = DenoiseKTDataset(test_data[0], test_data[1], test_data[2])

        question_concepts = self.build_question_concepts()
        question_graph = self.build_question_graph()

        logger.info(
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
            f"Test: {len(test_dataset)}"
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            question_concepts,
            question_graph,
        )
