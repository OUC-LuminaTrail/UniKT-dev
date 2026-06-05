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
    - 题目正确率（difficulty_rates）: 从数据中统计每题的正确率
    - 学生尝试次数（attempt_counts）: 对每个学生统计每题的累计尝试次数
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        r"""准备 DAGKT 模型所需的数据。"""
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        # 构建用户答题序列
        user_sequence, user_response, user_mask, user_id_sequence = (
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

        # 计算学生尝试次数 [num_users, max_seq_len]
        attempt_counts = self._compute_attempt_counts(user_sequence)

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
        """从数据中统计每题的正确率。

        Args:
            num_questions: 题目总数
            fold_idx: 要排除的 fold 索引（可选）

        Returns:
            题目正确率张量 [num_questions, 1]
        """
        import tqdm.auto as tqdm

        data = self.data_src.get_sequence_data().to_pandas()

        # 排除指定 fold 的数据
        if fold_idx is not None and "fold" in data.columns:
            data = data[data["fold"] != fold_idx]
            self.logger.info(
                f"Excluding fold {fold_idx} from correct rate calculation."
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

    def _compute_attempt_counts(self, user_sequence: np.ndarray) -> np.ndarray:
        """计算每个学生对每题的累计尝试次数。

        对每个学生遍历序列，统计在当前时间步之前已经做过同一题的次数。
        结果归一化到 [0, 1]。

        Args:
            user_sequence: 用户问题序列 [num_users, max_seq_len]

        Returns:
            归一化的尝试次数 [num_users, max_seq_len]
        """
        num_users, seq_len = user_sequence.shape
        attempt_counts = np.zeros((num_users, seq_len), dtype=np.float32)

        for u in range(num_users):
            question_seen: dict[int, int] = {}
            for t in range(seq_len):
                q = user_sequence[u, t]
                if q == 0:  # padding 位置跳过
                    continue
                attempt_counts[u, t] = question_seen.get(q, 0)
                question_seen[q] = question_seen.get(q, 0) + 1

        # 归一化到 [0, 1]
        max_attempts = attempt_counts.max()
        if max_attempts > 0:
            attempt_counts = attempt_counts / max_attempts

        self.logger.info(
            f"Attempt counts: max_raw={max_attempts}, "
            f"normalized_max={attempt_counts.max():.4f}"
        )

        return attempt_counts

    def split_kfold_data(
        self, user_sequence, user_response, user_mask, attempt_counts, fold_idx
    ):
        """K-fold 划分数据，返回包含 attempt_counts 的数据元组。"""
        from sklearn.model_selection import KFold

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        kf = KFold(n_splits=kfold_n_splits, shuffle=False)

        indices = list(kf.split(user_sequence))
        train_val_idx, test_idx = indices[fold_idx]

        # 从训练+验证集中再划分训练集和验证集
        # 使用训练+验证集的前 80% 作为训练集，后 20% 作为验证集
        n_train_val = len(train_val_idx)
        n_train = int(n_train_val * 0.8)

        # 确保不重复随机划分：使用固定的训练/验证切分
        train_idx = train_val_idx[:n_train]
        val_idx = train_val_idx[n_train:]

        train_data = (
            user_sequence[train_idx],
            user_response[train_idx],
            user_mask[train_idx],
            attempt_counts[train_idx],
        )
        val_data = (
            user_sequence[val_idx],
            user_response[val_idx],
            user_mask[val_idx],
            attempt_counts[val_idx],
        )
        test_data = (
            user_sequence[test_idx],
            user_response[test_idx],
            user_mask[test_idx],
            attempt_counts[test_idx],
        )

        return train_data, val_data, test_data
