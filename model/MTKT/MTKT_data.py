"""MTKT 模型数据处理模块

负责加载技能序列数据并计算时间间隔特征 (rgap, sgap, pcount)。
"""

import math

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData

logger = get_logger(__name__)


class MTKTDataset(Dataset):
    """MTKT 数据集

    训练/验证模式返回 7-元组:
        (sequence, response, mask, question, rgap, sgap, pcount)
    """

    def __init__(self, sequences, responses, masks, questions, rgaps, sgaps, pcounts):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.questions = questions
        self.rgaps = rgaps
        self.sgaps = sgaps
        self.pcounts = pcounts

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.questions[idx], dtype=torch.long),
            torch.tensor(self.rgaps[idx], dtype=torch.long),
            torch.tensor(self.sgaps[idx], dtype=torch.long),
            torch.tensor(self.pcounts[idx], dtype=torch.long),
        )


def _compute_time_gaps(
    skills: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int,
    num_rgap: int,
    num_sgap: int,
    num_pcount: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算单个用户序列的时间间隔特征

    Args:
        skills: 技能 ID 数组, shape (max_seq_len,)
        timestamps: 时间戳数组 (毫秒), shape (max_seq_len,)
        seq_len: 有效序列长度
        num_rgap: 复习间隔桶数量
        num_sgap: 连续间隔桶数量
        num_pcount: 练习次数桶数量

    Returns:
        rgap: 复习间隔索引, shape (max_seq_len,)
        sgap: 连续间隔索引, shape (max_seq_len,)
        pcount: 练习次数索引, shape (max_seq_len,)
    """
    max_len = len(skills)
    rgap = np.zeros(max_len, dtype=np.int64)
    sgap = np.full(max_len, num_sgap - 1, dtype=np.int64)
    pcount = np.zeros(max_len, dtype=np.int64)

    # Track last timestamp per skill and cumulative count
    last_ts: dict[int, float] = {}
    skill_count: dict[int, int] = {}

    def _log2_bucket(val):
        """Discretize via round(log2(val+1))."""
        return round(math.log(val + 1, 2))

    for t in range(seq_len):
        s = int(skills[t])

        # rgap: time-based log2 discretization of elapsed minutes since last same skill
        if s in last_ts and timestamps[t] > 0 and last_ts[s] > 0:
            diff_min = (timestamps[t] - last_ts[s]) / 60000.0
            if diff_min >= 0:
                bucket = _log2_bucket(diff_min) + 1
                rgap[t] = min(bucket, num_rgap - 1)

        # sgap: time-based log2 discretization of minutes since previous interaction
        if t > 0 and timestamps[t] > 0 and timestamps[t - 1] > 0:
            diff_ms = timestamps[t] - timestamps[t - 1]
            diff_min = diff_ms / 60000.0
            if diff_min >= 0:
                bucket = _log2_bucket(diff_min) + 1
                sgap[t] = min(bucket, num_sgap - 1)

        # pcount: log2 discretization of prior practice count for this skill
        if s in skill_count:
            pc = skill_count[s]
            bucket = _log2_bucket(pc)
            pcount[t] = min(bucket, num_pcount - 1)

        # Update state
        last_ts[s] = timestamps[t]
        skill_count[s] = skill_count.get(s, 0) + 1

    return rgap, sgap, pcount


class MTKTModelData(SkillModelData):
    """MTKT 模型数据加载器

    继承 SkillModelData, 在标准技能序列基础上额外计算时间间隔特征。
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    def _build_time_gaps(
        self,
        user_sequence: np.ndarray,
        user_mask: np.ndarray,
        user_timestamp: np.ndarray,
        num_rgap: int,
        num_sgap: int,
        num_pcount: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """为所有用户计算时间间隔特征

        Args:
            user_sequence: 技能 ID, shape (num_users, max_seq_len)
            user_mask: 有效掩码, shape (num_users, max_seq_len)
            user_timestamp: 时间戳, shape (num_users, max_seq_len)
            num_rgap/num_sgap/num_pcount: 桶数量

        Returns:
            user_rgap, user_sgap, user_pcount: 各 shape (num_users, max_seq_len)
        """
        num_users: int = user_sequence.shape[0]
        max_seq_len: int = user_sequence.shape[1]

        user_rgap = np.zeros((num_users, max_seq_len), dtype=np.int64)
        user_sgap = np.full((num_users, max_seq_len), num_sgap - 1, dtype=np.int64)
        user_pcount = np.zeros((num_users, max_seq_len), dtype=np.int64)

        for i in range(num_users):
            mask_row = user_mask[i]
            valid_positions = np.where(mask_row > 0)[0]
            if len(valid_positions) == 0:
                continue
            seq_len = int(valid_positions[-1]) + 1

            r, s, p = _compute_time_gaps(
                user_sequence[i],
                user_timestamp[i],
                seq_len,
                num_rgap,
                num_sgap,
                num_pcount,
            )
            user_rgap[i] = r
            user_sgap[i] = s
            user_pcount[i] = p

        return user_rgap, user_sgap, user_pcount

    def _load_timestamps(self) -> np.ndarray:
        """从原始数据加载用户时间戳数组

        Returns:
            user_timestamp: shape (num_users, max_seq_len)
        """
        data = self.data_src.get_split_skill_sequence_data().to_pandas()
        max_seq_len: int = self.data_src.get_metadata("max_seq_len")
        num_users: int = int(data["user"].nunique())

        user_timestamp = np.zeros((num_users, max_seq_len), dtype=np.float64)
        user_indices = data["user"].values
        seq_positions = data["seq_pos"].values
        user_timestamp[user_indices, seq_positions] = data["timestamp"].values
        return user_timestamp

    @override
    def prepare_data(self, args) -> tuple:
        """准备训练和验证数据

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        fold_idx = args.fold if args.fold >= 0 else None
        num_rgap = args.num_rgap
        num_sgap = args.num_sgap
        num_pcount = args.num_pcount

        # 1. 构建标准技能序列
        user_sequence, user_response, user_mask, _, user_question = (
            self.build_sequence_data()
        )

        # 2. 加载时间戳
        user_timestamp = self._load_timestamps()

        # 3. 计算时间间隔特征
        logger.info("Computing time gap features (rgap, sgap, pcount)...")
        user_rgap, user_sgap, user_pcount = self._build_time_gaps(
            user_sequence,
            user_mask,
            user_timestamp,
            num_rgap,
            num_sgap,
            num_pcount,
        )

        # 4. K-fold 划分
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            logger.info(
                f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
            )
            (
                train_data,
                val_data,
                _,
            ) = self.split_kfold_data(
                user_sequence,
                user_response,
                user_mask,
                user_question,
                user_rgap,
                user_sgap,
                user_pcount,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("K-fold cross-validation is not enabled.")

        # 5. 构建 Dataset
        train_dataset = MTKTDataset(
            train_data[0],
            train_data[1],
            train_data[2],
            train_data[3],
            train_data[4],
            train_data[5],
            train_data[6],
        )
        val_dataset = MTKTDataset(
            val_data[0],
            val_data[1],
            val_data[2],
            val_data[3],
            val_data[4],
            val_data[5],
            val_data[6],
        )

        # 6. Windowlate 测试集
        test_dataset = self.create_windowlate_iterable_dataset(args.max_seq_len)

        logger.info(
            f"MTKT data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )

        return train_dataset, val_dataset, test_dataset
