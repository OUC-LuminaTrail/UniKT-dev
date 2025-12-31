import torch
import numpy as np
from tqdm import tqdm
from utils.data_process.data_source import DataSource
from utils.net_data import GraphModelData
from utils.core import get_logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from typing_extensions import override


class SQGKTDataset(Dataset):
    def __init__(self, sequences, responses, masks, user_ids):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.user_ids = user_ids

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.long),
            torch.tensor(self.user_ids[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class SQGKTModelData(GraphModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    def sample_fixed_neighbors(
        self, matrix, neighbor_size, axis=1, desc="Sampling neighbors"
    ):
        """
        高效地从关系矩阵中采样固定数量的邻居

        参数:
            matrix: 关系矩阵 (二值或计数矩阵)
            neighbor_size: 固定的邻居数量
            axis: 采样维度 (0表示按列采样, 1表示按行采样)
            desc: 进度条描述信息

        返回:
            neighbors: 形状为 (num_nodes, neighbor_size) 的邻居索引数组
        """
        if axis == 0:
            matrix = matrix.T

        num_nodes = matrix.shape[0]
        neighbors = np.zeros((num_nodes, neighbor_size), dtype=np.int32)

        # 使用进度条显示采样进度
        for i in tqdm(range(num_nodes), desc=desc, unit="nodes"):
            # 找到非零邻居的索引
            neighbor_indices = np.where(matrix[i] > 0)[0]
            num_neighbors = len(neighbor_indices)

            if num_neighbors == 0:
                # 无邻居，保持全0
                continue
            elif num_neighbors >= neighbor_size:
                # 邻居数足够，无放回采样
                neighbors[i] = np.random.choice(
                    neighbor_indices, neighbor_size, replace=False
                )
            else:
                # 邻居数不足，有放回采样
                neighbors[i] = np.random.choice(
                    neighbor_indices, neighbor_size, replace=True
                )

        return neighbors

    def build_uq_table_with_factors(self):
        """
        构建三维的用户-问题表，包含三个因子：
        - ability_factor: 用户能力因子（用户的平均正确率）
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

        # 参数设置（参考原始SQGKT实现）
        k = 0.3
        d = 0.7
        b = 10

        # 1. 计算用户能力因子（每个用户的平均正确率）
        self.logger.info("Computing ability factors...")

        # 确保 user 和 question 是整数
        user_ids = data["user"].astype(int).values
        question_ids = data["question"].astype(int).values

        # 计算每个用户的平均正确率
        user_ability_series = data.groupby("user")["label"].mean()
        # 映射回每个交互，填充默认值 0.5
        ability_factors = data["user"].map(user_ability_series).fillna(0.5).values

        # 2. 计算问题的尝试次数和提示次数统计
        if "attempt_count" in data.columns and "hint_count" in data.columns:
            self.logger.info("Computing attempt and hint factors from data...")

            # 计算每个问题的平均尝试次数和提示次数
            question_attempt_mean = data.groupby("question")["attempt_count"].mean()
            question_hint_mean = data.groupby("question")["hint_count"].mean()

            # 映射到每个交互
            mean_attempts = data["question"].map(question_attempt_mean).fillna(1).values
            mean_hints = data["question"].map(question_hint_mean).fillna(0).values

            attempt_counts = data["attempt_count"].fillna(1).values
            hint_counts = data["hint_count"].fillna(0).values

            # 计算 attempt_factor
            attempt_factor = 1 - poisson.cdf(attempt_counts - 1, mean_attempts)
            attempt_factor_g = k + (1 - k) / (1 + np.exp(-d * (attempt_factor - b)))

            # 计算 hint_factor
            hint_factor = np.zeros_like(hint_counts, dtype=np.float32)
            mask_hint = mean_hints > 0
            if np.any(mask_hint):
                hint_factor[mask_hint] = 1 - poisson.cdf(
                    hint_counts[mask_hint] - 1, mean_hints[mask_hint]
                )

            hint_factor_g = k + (1 - k) / (1 + np.exp(-d * (hint_factor - b)))

            # 存储到三维表中
            uq_table[user_ids, question_ids, 0] = ability_factors
            uq_table[user_ids, question_ids, 1] = attempt_factor_g
            uq_table[user_ids, question_ids, 2] = hint_factor_g

        else:
            self.logger.warning(
                "Using simplified factors (attempt_count and hint_count not available)..."
            )
            # 简化版本：只使用能力因子和基于问题难度的估计
            # 计算每个问题的平均正确率（作为难度的反向指标）
            question_difficulty_series = data.groupby("question")["label"].mean()

            # 映射到每个交互
            difficulties = (
                data["question"].map(question_difficulty_series).fillna(0.5).values
            )

            attempt_factor_g = 1.0 - difficulties
            hint_factor_g = 1.0 - difficulties

            # 存储到三维表中
            uq_table[user_ids, question_ids, 0] = ability_factors
            uq_table[user_ids, question_ids, 1] = attempt_factor_g
            uq_table[user_ids, question_ids, 2] = hint_factor_g

        self.logger.info(f"UQ table shape: {uq_table.shape}")
        return uq_table

    def generate_question_skill_neighbors(
        self, qs_matrix, q_neighbor_size=4, s_neighbor_size=10
    ):
        """
        生成SQGKT问题-技能图的固定大小邻居数组

        参数:
            qs_matrix: 问题-技能关系矩阵，形状 (num_question, num_skill)
            q_neighbor_size: 每个问题的固定邻居数量
            s_neighbor_size: 每个技能的固定邻居数量

        返回:
            q_neighbors: 形状 (num_question, q_neighbor_size) 的问题邻居数组
            s_neighbors: 形状 (num_skill, s_neighbor_size) 的技能邻居数组
        """
        self.logger.info("Generating question-skill graph neighbors...")
        # 问题的邻居是技能 (按行采样)
        q_neighbors = self.sample_fixed_neighbors(
            qs_matrix,
            q_neighbor_size,
            axis=1,
            desc="Sampling question->skill neighbors",
        )

        # 技能的邻居是问题 (按列采样，即转置后按行采样)
        s_neighbors = self.sample_fixed_neighbors(
            qs_matrix,
            s_neighbor_size,
            axis=0,
            desc="Sampling skill->question neighbors",
        )

        return q_neighbors, s_neighbors

    def generate_user_question_neighbors(
        self, uq_matrix, u_neighbor_size=5, q_neighbor_size=5
    ):
        """
        生成SQGKT用户-问题图的固定大小邻居数组

        参数:
            uq_matrix: 用户-问题关系矩阵，形状 (num_user, num_question)
            u_neighbor_size: 每个用户的固定邻居数量
            q_neighbor_size: 每个问题的固定邻居数量

        返回:
            u_neighbors: 形状 (num_user, u_neighbor_size) 的用户邻居数组
            q_neighbors: 形状 (num_question, q_neighbor_size) 的问题邻居数组
        """
        self.logger.info("Generating user-question graph neighbors...")
        # 用户的邻居是问题 (按行采样)
        u_neighbors = self.sample_fixed_neighbors(
            uq_matrix, u_neighbor_size, axis=1, desc="Sampling user->question neighbors"
        )

        # 问题的邻居是用户 (按列采样，即转置后按行采样)
        q_neighbors = self.sample_fixed_neighbors(
            uq_matrix, q_neighbor_size, axis=0, desc="Sampling question->user neighbors"
        )

        return u_neighbors, q_neighbors

    @override
    def prepare_data(self, args):
        r"""
        准备SQGKT模型所需的数据
        """
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        min_seq_len = self.data_src.get_metadata("min_seq_len")

        # 构建用户答题序列
        user_sequence, user_response, user_mask, user_id_sequence = (
            self.build_sequence_data(max_seq_len, min_seq_len)
        )

        # 构建学生-问题矩阵和问题-技能矩阵
        uq_matrix_2d = self.build_relationship_matrix(("user", "answers", "question"))
        qs_matrix = self.build_relationship_matrix(("question", "has", "skill"))

        # 构建三维的用户-问题表（包含三个因子）
        uq_table = self.build_uq_table_with_factors()

        # 生成固定大小的邻居数组（使用2D矩阵）
        qs_q_neighbors, qs_s_neighbors = self.generate_question_skill_neighbors(
            qs_matrix,
            q_neighbor_size=args.qs_question_neighbors,
            s_neighbor_size=args.qs_skill_neighbors,
        )

        uq_u_neighbors, uq_q_neighbors = self.generate_user_question_neighbors(
            uq_matrix_2d,
            u_neighbor_size=args.uq_user_neighbors,
            q_neighbor_size=args.uq_question_neighbors,
        )

        # 划分训练集和验证集
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"Fold index {fold_idx} is out of range for {kfold_n_splits} folds."
                )
            train_data, val_data = self.split_kfold_data(
                user_sequence,
                user_response,
                user_mask,
                user_id_sequence,
                fold_idx=fold_idx,
            )
        else:
            train_data, val_data = self.split_data(
                user_sequence, user_response, user_mask, user_id_sequence
            )

        # 构建模型数据集
        train_dataset = SQGKTDataset(
            train_data[0], train_data[1], train_data[2], train_data[3]
        )
        val_dataset = SQGKTDataset(val_data[0], val_data[1], val_data[2], val_data[3])

        # 构建数据加载器
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
        return (
            train_dataloader,
            val_dataloader,
            uq_table,
            qs_matrix,
            qs_q_neighbors,
            qs_s_neighbors,
            uq_u_neighbors,
            uq_q_neighbors,
        )
