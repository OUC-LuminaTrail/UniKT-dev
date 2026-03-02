import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process.data_source import DataSource
from utils.model_data import QuestionModelData


class SQGKTDataset(Dataset):
    def __init__(self, users, sequences, responses, masks):
        self.users = users
        self.sequences = sequences
        self.responses = responses
        self.masks = masks

    def __getitem__(self, index):
        return (
            torch.tensor(self.users[index], dtype=torch.long),
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.bool),  # 掩码为布尔类型
        )

    def __len__(self):
        return len(self.sequences)


class SQGKTModelData(QuestionModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    @override
    def prepare_data(self, args):
        r"""
        准备SQGKT模型所需的数据
        """
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        # 构建用户答题序列
        user_sequence, user_response, user_mask, user_id_sequence = (
            self.build_sequence_data(args.max_seq_len)
        )

        # 构建问题-技能关联矩阵
        qs_table = self.build_relationship_matrix(("question", "has", "skill"))

        # 问题-技能图的邻接表
        # q_neighbors_qs[question_id] -> 该问题的技能邻居ID列表
        # c_neighbors_qs[skill_id] -> 该技能的问题邻居ID列表
        qs_q_neighbor_size = max(1, int(getattr(args, "qs_question_neighbors", 5)))
        qs_c_neighbor_size = max(1, int(getattr(args, "qs_skill_neighbors", 10)))
        self.logger.info(
            f"[SQGKT] QS neighbors: q->s={qs_q_neighbor_size}, s->q={qs_c_neighbor_size}"
        )
        q_neighbors_qs, c_neighbors_qs = self.build_graph_neighbors(
            qs_table,
            q_neighbor_size=qs_q_neighbor_size,
            c_neighbor_size=qs_c_neighbor_size,
        )
        qs_table = torch.tensor(qs_table, dtype=torch.long)
        q_neighbors_qs = torch.tensor(q_neighbors_qs, dtype=torch.long)
        c_neighbors_qs = torch.tensor(c_neighbors_qs, dtype=torch.long)

        # 构建用户-问题邻接表
        uq_matrix = self.build_relationship_matrix(("user", "answers", "question"))

        # 用户-问题图的邻接表
        # u_neighbors_uq[user_id] -> 该用户的问题邻居ID列表
        # q_neighbors_uq[question_id] -> 该问题的用户邻居ID列表
        uq_u_neighbor_size = max(1, int(getattr(args, "uq_user_neighbors", 5)))
        uq_q_neighbor_size = max(1, int(getattr(args, "uq_question_neighbors", 5)))
        self.logger.info(
            f"[SQGKT] UQ neighbors: u->q={uq_u_neighbor_size}, q->u={uq_q_neighbor_size}"
        )
        u_neighbors_uq, q_neighbors_uq = self.build_graph_neighbors(
            uq_matrix,
            q_neighbor_size=uq_u_neighbor_size,
            c_neighbor_size=uq_q_neighbor_size,
        )
        u_neighbors_uq = torch.tensor(u_neighbors_uq, dtype=torch.long)
        q_neighbors_uq = torch.tensor(q_neighbors_uq, dtype=torch.long)

        uq_table = self.build_uq_table_with_factors()
        uq_table = torch.tensor(uq_table, dtype=torch.float32)

        # 划分训练集和验证集
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"Fold index {fold_idx} is out of range for {kfold_n_splits} folds."
                )
            train_data, val_data = self.split_kfold_data(
                user_id_sequence,
                user_sequence,
                user_response,
                user_mask,
                fold_idx=fold_idx,
            )
        else:
            train_data, val_data = self.split_data(
                user_id_sequence, user_sequence, user_response, user_mask
            )

        # 构建模型数据集
        train_dataset = SQGKTDataset(
            train_data[0], train_data[1], train_data[2], train_data[3]
        )
        val_dataset = SQGKTDataset(val_data[0], val_data[1], val_data[2], val_data[3])

        return (
            train_dataset,
            val_dataset,
            qs_table,
            q_neighbors_qs,
            c_neighbors_qs,
            uq_table,
            u_neighbors_uq,
            q_neighbors_uq,
        )

    def build_graph_neighbors(self, adj_matrix, q_neighbor_size, c_neighbor_size):
        """
        从邻接矩阵构建邻居表。

        Args:
            adj_matrix: shape [num_row, num_col] 的0/1矩阵
            q_neighbor_size: 每行节点的邻居采样数
            c_neighbor_size: 每列节点的邻居采样数

        Returns:
            row_neighbors: [num_row, q_neighbor_size] 每行的列邻居
            col_neighbors: [num_col, c_neighbor_size] 每列的行邻居
        """
        num_row, num_col = adj_matrix.shape
        row_neighbors = np.zeros([num_row, q_neighbor_size], dtype=np.int32)
        col_neighbors = np.zeros([num_col, c_neighbor_size], dtype=np.int32)

        # 构建行 -> 列邻居
        for row_id in range(num_row):
            neighbors = np.argwhere(adj_matrix[row_id] > 0).reshape(-1)
            if len(neighbors) == 0:
                continue
            if len(neighbors) >= q_neighbor_size:
                row_neighbors[row_id] = np.random.choice(
                    neighbors, q_neighbor_size, replace=False
                )
            else:
                row_neighbors[row_id] = np.random.choice(
                    neighbors, q_neighbor_size, replace=True
                )

        # 构建列 -> 行邻居
        for col_id in range(num_col):
            neighbors = np.argwhere(adj_matrix[:, col_id] > 0).reshape(-1)
            if len(neighbors) == 0:
                continue
            if len(neighbors) >= c_neighbor_size:
                col_neighbors[col_id] = np.random.choice(
                    neighbors, c_neighbor_size, replace=False
                )
            else:
                col_neighbors[col_id] = np.random.choice(
                    neighbors, c_neighbor_size, replace=True
                )

        return row_neighbors, col_neighbors

    def build_uq_table_with_factors(self):
        """
        构建三维的用户-问题表，包含三个因子：
        - ability_factor: 用户学习能力因子（用户的平均正确率）
        - attempt_factor_g: 尝试因子（基于问题的尝试次数统计）
        - hint_factor_g: 提示因子（基于问题的提示次数统计）

        返回:
            uq_table: 形状 (num_user, num_question, 3) 的三维张量
        """
        self.logger.info("Building user-question table with factors...")
        from scipy.stats import poisson

        # 获取数据
        data = self.data_src.sequence_data
        num_users = self.data_src.get_metadata("num_users")
        num_questions = self.data_src.get_metadata("num_questions")

        # 创建三维表
        uq_table = np.zeros([num_users, num_questions, 3], dtype=np.float32)

        # 参数设置
        alpha = 0.3
        eta = 10
        beta = 0.7

        # 1. 计算学习能力因子
        self.logger.info("Computing learning ability factors...")

        # 用户ID和问题ID
        user_ids = data["user"].values
        question_ids = data["question"].values

        # 计算每个用户的平均正确率
        user_ability_series = data.groupby("user")["label"].mean()
        # 映射回每个交互，得到能力因子
        ability_factors = data["user"].map(user_ability_series).fillna(0.5).values

        # 2. 计算问题的尝试次数和提示次数统计
        if "attempt_count" in data.columns and "hint_count" in data.columns:
            self.logger.info("Computing attempt and hint factors...")

            # 计算每个问题的平均尝试次数和提示次数
            question_attempt_mean = data.groupby("question")["attempt_count"].mean()
            question_hint_mean = data.groupby("question")["hint_count"].mean()

            # 映射到每个交互
            mean_attempts = data["question"].map(question_attempt_mean).fillna(1).values
            mean_hints = data["question"].map(question_hint_mean).fillna(0).values

            # 获取每个交互的尝试次数和提示次数
            attempt_counts = data["attempt_count"].fillna(1).values
            hint_counts = data["hint_count"].fillna(0).values

            # 计算 attempt_factor
            attempt_factor = poisson.cdf(attempt_counts - 1, mean_attempts)
            attempt_factor_g = alpha + (1 - alpha) / (
                1 + np.exp(eta * (attempt_factor - beta))
            )

            # 计算 hint_factor
            hint_factor = np.zeros_like(hint_counts, dtype=np.float32)
            mask_hint = mean_hints > 0
            if np.any(mask_hint):
                hint_factor[mask_hint] = poisson.cdf(
                    hint_counts[mask_hint] - 1, mean_hints[mask_hint]
                )

            hint_factor_g = alpha + (1 - alpha) / (
                1 + np.exp(eta * (hint_factor - beta))
            )

            # 存储到三维表中
            uq_table[user_ids, question_ids, 0] = ability_factors
            uq_table[user_ids, question_ids, 1] = attempt_factor_g
            uq_table[user_ids, question_ids, 2] = hint_factor_g
        else:
            raise ValueError(
                "Data must contain 'attempt_count' and 'hint_count' columns to compute factors."
            )

        return uq_table
