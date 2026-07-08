"""DKT2 数据准备：Question 粒度序列 + 多概念折叠。"""

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class DKT2ModelData(QuestionModelData):
    """DKT2 数据类。

    序列粒度为 Question。多概念题目按其概念集合整体折叠为单一新概念：对
    question-skill 二部矩阵去重，每一个独特的概念组合分配一个 id。这样
    每个题目恰好对应一个概念，便于 DKT 式的“输出全概念向量再按下一概念
    收集”预测。概念 id 采用 1-index（0 留给 padding）。
    """

    def __init__(self, data_src):
        super().__init__(data_src)

    def prepare_data(self, args):
        user_sequence, user_response, user_mask, _ = self.load_sequence_data()
        num_questions = self.data_src.get_metadata("num_questions")

        # 多概念折叠：Q-矩阵每行是一个题目的概念集合，去重后每行对应一个新概念。
        q_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )
        collapsed = np.unique(q_matrix, axis=0, return_inverse=True)[
            1
        ]  # [num_questions]
        num_concepts = int(collapsed.max()) + 1
        num_skills = num_concepts + 1  # 1..num_concepts 为真实概念，0 为 padding
        logger.info(
            f"DKT2 folded {q_matrix.shape[1]} base skills into {num_concepts} concepts"
        )

        question_to_skill = collapsed + 1  # [num_questions]，1-index

        # 折叠后的概念序列；padding 位置（mask=0）置 0
        skills_sequence = question_to_skill[user_sequence]
        skills_sequence = skills_sequence * user_mask

        train_data, val_data, test_data = self.split_kfold_data(
            user_sequence,
            user_response,
            user_mask,
            skills_sequence,
            fold_idx=args.fold,
        )
        train_dataset = DKT2Dataset(*train_data)
        val_dataset = DKT2Dataset(*val_data)
        test_dataset = DKT2Dataset(*test_data)

        logger.info(
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset, num_skills, num_questions


class DKT2Dataset(Dataset):
    """每条样本返回 question/response/mask/skill 四元组（已折叠的概念序列）。"""

    def __init__(self, questions, responses, masks, skills):
        self.questions = torch.from_numpy(np.asarray(questions)).long()
        self.responses = torch.from_numpy(np.asarray(responses)).long()
        self.masks = torch.from_numpy(np.asarray(masks)).bool()
        self.skills = torch.from_numpy(np.asarray(skills)).long()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "questions": self.questions[idx],
            "responses": self.responses[idx],
            "masks": self.masks[idx],
            "skills": self.skills[idx],
        }
