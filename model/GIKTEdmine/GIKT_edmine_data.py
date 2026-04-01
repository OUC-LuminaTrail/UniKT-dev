import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class GIKTEdmineDataset(Dataset):
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


class GIKTEdmineModelData(QuestionModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        r"""
        准备GIKT模型所需的数据
        """
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        # 构建用户答题序列
        user_sequence, user_response, user_mask, _ = self.load_sequence_data()

        # 构建问题-技能关联矩阵
        q_table = self.build_relationship_matrix(("question", "has", "skill"))
        # 转换为GIKT所需的邻接表格式
        question2concept = self.q2c_from_q_table(q_table)
        concept2question = self.c2q_from_q_table(q_table)
        num_max_concept = int(q_table.sum(axis=1).max())
        num_q, num_c = q_table.shape[0], q_table.shape[1]
        question_neighbors, concept_neighbors = self.gen_gikt_graph(
            question2concept,
            concept2question,
            num_max_concept,
            min(20, int(num_q / num_c)),
        )
        q_table = torch.tensor(q_table, dtype=torch.long)
        question_neighbors = torch.tensor(question_neighbors, dtype=torch.long)
        concept_neighbors = torch.tensor(concept_neighbors, dtype=torch.long)

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
            train_data, val_data, test_data = self.split_kfold_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            raise ValueError("fold_idx must be specified for K-fold cross-validation.")

        # 构建模型数据集
        train_dataset = GIKTEdmineDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = GIKTEdmineDataset(val_data[0], val_data[1], val_data[2])
        test_dataset = GIKTEdmineDataset(test_data[0], test_data[1], test_data[2])

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            question_neighbors,
            concept_neighbors,
            q_table,
        )

    def c2q_from_q_table(self, q_table: np.ndarray) -> dict[int, list[int]]:
        """
        Converts a question-concept matrix (q_table) into a dictionary mapping each concept to its associated questions.
        :param q_table: A 2D NumPy array representing the question-concept relationship, where rows correspond to questions and columns correspond to concepts. A value of 1 indicates a relationship between a question and a concept.
        :return: A dictionary where each key is a concept ID, and the value is a list of question IDs linked to that concept.
        """
        self.check_q_table(q_table)
        return {
            i: np.argwhere(q_table[:, i] == 1).reshape(-1).tolist()
            for i in range(q_table.shape[1])
        }

    def q2c_from_q_table(self, q_table: np.ndarray) -> dict[int, list[int]]:
        """
        Converts a question-concept matrix (q_table) into a dictionary mapping each question to its associated concepts.
        :param q_table: A 2D NumPy array representing the question-concept relationship, where rows correspond to questions and columns correspond to concepts. A value of 1 indicates a relationship between a question and a concept.
        :return: A dictionary where each key is a question ID, and the value is a list of concept IDs linked to that concept.
        """
        self.check_q_table(q_table)
        return {
            i: np.argwhere(q_table[i] == 1).reshape(-1).tolist()
            for i in range(q_table.shape[0])
        }

    def check_q_table(self, q_table: np.ndarray):
        # Check if q_table is a 2D NumPy array
        if q_table.size == 0 or q_table.ndim != 2:
            raise IndexError("Input q_table must be a 2D NumPy array.")

        # Check if q_table contains only 0s and 1s
        if not np.all(np.isin(q_table, [0, 1])):
            raise ValueError("Input q_table must contain only 0s and 1s.")

        rows_check = np.any(q_table == 1, axis=1)
        cols_check = np.any(q_table == 1, axis=0)

        if not (np.all(rows_check) and np.all(cols_check)):
            raise ValueError(
                "Each row and column of the input q_table has at least one value of 1."
            )

    def gen_gikt_graph(
        self, question2concept, concept2question, q_neighbor_size, c_neighbor_size
    ):
        num_question = len(question2concept)
        num_concept = len(concept2question)
        q_neighbors = np.zeros([num_question, q_neighbor_size], dtype=np.int32)
        c_neighbors = np.zeros([num_concept, c_neighbor_size], dtype=np.int32)
        for q_id, neighbors in question2concept.items():
            if len(neighbors) >= q_neighbor_size:
                q_neighbors[q_id] = np.random.choice(
                    neighbors, q_neighbor_size, replace=False
                )
            else:
                q_neighbors[q_id] = np.random.choice(
                    neighbors, q_neighbor_size, replace=True
                )
        for c_id, neighbors in concept2question.items():
            if len(neighbors) >= c_neighbor_size:
                c_neighbors[c_id] = np.random.choice(
                    neighbors, c_neighbor_size, replace=False
                )
            else:
                c_neighbors[c_id] = np.random.choice(
                    neighbors, c_neighbor_size, replace=True
                )
        return q_neighbors, c_neighbors
