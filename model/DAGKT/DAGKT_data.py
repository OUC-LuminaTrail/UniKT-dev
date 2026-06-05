import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class DAGKTDataset(Dataset):
    """DAGKT 数据集，返回 (sequence, response, mask, attempt_counts) 四元组。"""

    def __init__(self, sequences, responses, masks, attempt_counts):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.responses = torch.as_tensor(responses, dtype=torch.long)
        self.masks = torch.as_tensor(masks, dtype=torch.long)
        self.attempt_counts = torch.as_tensor(attempt_counts, dtype=torch.float)

    def __getitem__(self, index):
        return (
            self.sequences[index],
            self.responses[index],
            self.masks[index],
            self.attempt_counts[index],
        )

    def __len__(self):
        return len(self.sequences)


class DAGKTModelData(QuestionModelData):
    """DAGKT 模型数据准备类。

    在 GIKT 数据基础上增加：
    - 题目正确率（difficulty_rates）: 从训练集统计每题的正确率
    - 学生尝试次数（attempt_counts）: 直接使用数据集中的 attempt_count 字段
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    def load_sequence_data(self):
        """加载用户答题序列，额外加载 attempt_count 字段。

        Returns:
            (user_sequence, user_response, user_mask, user_id_sequence, user_attempt)
        """
        import numpy as np

        self.logger.info("Building response sequences from split data...")

        data = self.data_src.get_split_question_sequence_data().to_pandas()
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_users = data["user"].nunique()

        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)
        user_attempt = np.zeros((num_users, max_seq_len), dtype=np.float32)

        user_indices = data["user"].values
        seq_positions = data["seq_pos"].values

        user_sequence[user_indices, seq_positions] = data["question"].values
        user_id_sequence[user_indices, seq_positions] = user_indices
        user_response[user_indices, seq_positions] = data["label"].values
        user_mask[user_indices, seq_positions] = 1

        # 加载 attempt_count 字段
        if "attempt_count" in data.columns:
            user_attempt[user_indices, seq_positions] = data[
                "attempt_count"
            ].values.astype(np.float32)
            self.logger.info("Loaded attempt_count from dataset.")
        else:
            self.logger.warning("attempt_count column not found in data, using zeros.")

        return user_sequence, user_response, user_mask, user_id_sequence, user_attempt

    @override
    def prepare_data(self, args):
        r"""准备 DAGKT 模型所需的数据。"""
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        # 构建用户答题序列（含 attempt_count）
        user_sequence, user_response, user_mask, user_id_sequence, attempt_counts = (
            self.load_sequence_data()
        )

        # 构建异构图（同 GIKT）
        graph = self.build_hetero_graph(
            [
                (
                    "question",
                    "has",
                    "skill",
                )
            ]
        )

        # 构建问题-技能关联矩阵（同 GIKT）
        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

        # 计算题目正确率 [num_questions, 1]
        num_questions = self.data_src.get_metadata("num_questions")
        question_difficulty = self._compute_question_correct_rates(
            num_questions, fold_idx
        )

        # 归一化 attempt_count 到 [0, 1]
        attempt_counts = self._normalize_attempt_counts(attempt_counts)

        # K-fold 划分
        if fold_idx is not None:
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
                attempt_counts,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("fold_idx must be specified for K-fold cross-validation")

        # 构建模型数据集
        train_dataset = DAGKTDataset(*train_data)
        val_dataset = DAGKTDataset(*val_data)
        test_dataset = DAGKTDataset(*test_data)

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            graph,
            question_skill_matrix,
            question_difficulty,
        )

    def _compute_question_correct_rates(
        self, num_questions: int, fold_idx: int | None = None
    ) -> torch.Tensor:
        """仅从训练集统计每题的正确率。

        排除验证集（fold == fold_idx）和测试集（fold == -1），
        只使用训练集数据以避免数据泄露。

        Args:
            num_questions: 题目总数
            fold_idx: 当前验证集的 fold 索引（可选）

        Returns:
            题目正确率张量 [num_questions, 1]
        """
        import tqdm.auto as tqdm

        data = self.data_src.get_sequence_data().to_pandas()

        # 仅使用训练集数据：排除验证集（fold_idx）和测试集（fold == -1）
        if "fold" in data.columns:
            mask = data["fold"] != -1  # 排除测试集
            if fold_idx is not None:
                mask &= data["fold"] != fold_idx  # 排除验证集
            data = data[mask]
            self.logger.info(
                f"Computing correct rates from training data only "
                f"(excluded fold {fold_idx} and test fold -1)."
            )

        # 统计每题正确率
        question_stats = data.groupby("question")["label"].mean()

        # 构建正确率数组
        correct_rates = torch.ones(num_questions, 1) * 0.5  # 默认 0.5
        for qid, rate in tqdm.tqdm(
            question_stats.items(),
            total=len(question_stats),
            desc="Computing question correct rates",
        ):
            if qid < num_questions:
                correct_rates[qid] = rate

        self.logger.info(
            f"Question correct rates: min={correct_rates.min().item():.4f}, "
            f"max={correct_rates.max().item():.4f}, "
            f"mean={correct_rates.mean().item():.4f}"
        )

        return correct_rates

    def _normalize_attempt_counts(self, attempt_counts: np.ndarray) -> np.ndarray:
        """归一化 attempt_count 到 [0, 1]。

        Args:
            attempt_counts: 原始尝试次数 [num_users, max_seq_len]

        Returns:
            归一化的尝试次数 [num_users, max_seq_len]
        """
        max_attempts = attempt_counts.max()
        if max_attempts > 0:
            attempt_counts = attempt_counts / max_attempts

        self.logger.info(
            f"Attempt counts: max_raw={max_attempts}, "
            f"normalized_max={attempt_counts.max():.4f}"
        )

        return attempt_counts
