import os

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import BaseModelData


class WindowlateIterableDataset(IterableDataset):
    """Stream windowlate samples from parquet without materializing all samples in memory."""

    def __init__(
        self,
        parquet_path: str,
        max_seq_len: int,
        batch_read_rows: int = 200_000,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.max_seq_len = max_seq_len
        self.batch_read_rows = batch_read_rows
        self._num_samples = self._compute_num_samples()

    def _compute_num_samples(self) -> int:
        import polars as pl

        stats = (
            pl.scan_parquet(self.parquet_path)
            .select(pl.col("sample_id").n_unique().alias("num_samples"))
            .collect(engine="streaming")
        )
        return int(stats["num_samples"][0])

    def __len__(self) -> int:
        return self._num_samples

    def _build_sample_tensors(
        self,
        positions: np.ndarray,
        skills: np.ndarray,
        responses: np.ndarray,
        masks: np.ndarray,
        group_ids: np.ndarray,
        true_labels: np.ndarray,
    ):
        sequence = np.zeros(self.max_seq_len, dtype=np.int64)
        response = np.zeros(self.max_seq_len, dtype=np.int64)
        mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        late_group_id = np.full(self.max_seq_len, -1, dtype=np.int64)
        labels = np.zeros(self.max_seq_len, dtype=np.int64)

        sequence[positions] = skills
        response[positions] = responses
        mask[positions] = masks.astype(np.bool_)
        late_group_id[positions] = group_ids
        labels[positions] = true_labels

        return (
            torch.from_numpy(sequence),
            torch.from_numpy(response),
            torch.from_numpy(mask),
            torch.from_numpy(late_group_id),
            torch.from_numpy(labels),
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise RuntimeError(
                "WindowlateIterableDataset does not support num_workers > 1. "
                "Please use DataLoader(..., num_workers=0)."
            )

        parquet_file = pq.ParquetFile(self.parquet_path)
        columns = [
            "sample_id",
            "position",
            "skill",
            "response",
            "mask",
            "group_id",
            "true_label",
        ]

        current_sample_id = None
        pos_buf = []
        skill_buf = []
        resp_buf = []
        mask_buf = []
        group_buf = []
        label_buf = []

        def _flush_current():
            if current_sample_id is None:
                return None
            return self._build_sample_tensors(
                positions=np.asarray(pos_buf, dtype=np.int64),
                skills=np.asarray(skill_buf, dtype=np.int64),
                responses=np.asarray(resp_buf, dtype=np.int64),
                masks=np.asarray(mask_buf, dtype=np.int8),
                group_ids=np.asarray(group_buf, dtype=np.int64),
                true_labels=np.asarray(label_buf, dtype=np.int64),
            )

        for record_batch in parquet_file.iter_batches(
            batch_size=self.batch_read_rows, columns=columns
        ):
            batch = record_batch.to_pydict()
            sample_ids = np.asarray(batch["sample_id"], dtype=np.int64)
            positions = np.asarray(batch["position"], dtype=np.int64)
            skills = np.asarray(batch["skill"], dtype=np.int64)
            responses = np.asarray(batch["response"], dtype=np.int64)
            masks = np.asarray(batch["mask"], dtype=np.int8)
            group_ids = np.asarray(batch["group_id"], dtype=np.int64)
            true_labels = np.asarray(batch["true_label"], dtype=np.int64)

            if sample_ids.size == 0:
                continue

            boundaries = np.flatnonzero(sample_ids[1:] != sample_ids[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [sample_ids.size]))

            for start, end in zip(starts, ends, strict=False):
                sid = int(sample_ids[start])
                if current_sample_id is not None and sid < current_sample_id:
                    raise ValueError(
                        "windowlate parquet must be sorted by sample_id. "
                        f"Found out-of-order sample_id {sid} after {current_sample_id}."
                    )
                if current_sample_id is None:
                    current_sample_id = sid
                elif sid != current_sample_id:
                    sample = _flush_current()
                    if sample is not None:
                        yield sample
                    current_sample_id = sid
                    pos_buf.clear()
                    skill_buf.clear()
                    resp_buf.clear()
                    mask_buf.clear()
                    group_buf.clear()
                    label_buf.clear()

                pos_buf.extend(positions[start:end].tolist())
                skill_buf.extend(skills[start:end].tolist())
                resp_buf.extend(responses[start:end].tolist())
                mask_buf.extend(masks[start:end].tolist())
                group_buf.extend(group_ids[start:end].tolist())
                label_buf.extend(true_labels[start:end].tolist())

        sample = _flush_current()
        if sample is not None:
            yield sample


class SkillModelData(BaseModelData):
    """
    技能序列数据基类

    用于构建基于技能（skill/concept）的知识追踪模型数据
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    def build_sequence_data(self, max_seq_len: int):
        r"""
        构建用户答题序列，将问题ID映射到技能ID，并展开多知识点

        参数:
            max_seq_len: 最大序列长度

        返回:
            user_sequence: 用户技能ID序列，shape为(num_users, max_seq_len)
            user_response: 用户响应序列，shape为(num_users, max_seq_len)
            user_mask: 用户掩码序列，shape为(num_users, max_seq_len)
            user_id_sequence: 用户ID序列，shape为(num_users, max_seq_len)
        """
        import numpy as np
        from tqdm import tqdm

        data = self.data_src.get_sequence_data().to_pandas()
        question_data = self.data_src.get_question_data().to_pandas()
        num_users = self.data_src.get_metadata("num_users")

        # 构建问题ID到技能ID列表的映射
        # 在数据预处理中，question_data已经将多知识点展开为多个技能ID
        # 例如：question_id=1, skill=10; question_id=1, skill=20
        question_to_skills = {}
        for row in question_data.itertuples():
            qid = row.question
            sid = row.skill
            if qid not in question_to_skills:
                question_to_skills[qid] = []
            question_to_skills[qid].append(sid)

        # 初始化序列数组
        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)

        # 用户序列长度计数
        num_sequence = [0] * num_users

        # 按用户分组构建序列，展开多知识点
        for row in tqdm(
            data.itertuples(), total=data.shape[0], desc="Building skill sequences"
        ):
            user_idx = row.user
            question_idx = row.question
            label = row.label

            # 获取该问题对应的所有技能
            skills = question_to_skills.get(question_idx, [0])

            # 展开多知识点：一个问题对应多个技能时，展开成多个独立的交互
            for skill_idx in skills:
                # 如果当前用户的序列长度未达到最大长度，则添加数据
                if num_sequence[user_idx] < max_seq_len:
                    user_sequence[user_idx, num_sequence[user_idx]] = skill_idx
                    user_id_sequence[user_idx, num_sequence[user_idx]] = user_idx
                    user_response[user_idx, num_sequence[user_idx]] = label
                    user_mask[user_idx, num_sequence[user_idx]] = 1
                    num_sequence[user_idx] += 1
                else:
                    # 如果序列已满，跳过剩余的技能
                    break

        self.logger.debug(
            f"Built skill sequences for {num_users} users, max_len={max_seq_len}"
        )
        self.logger.debug(
            f"Multi-skill expansion applied: {len(question_to_skills)} questions mapped to skills"
        )

        return user_sequence, user_response, user_mask, user_id_sequence

    def load_windowlate_data(self, max_seq_len: int):
        r"""
        加载用于 windowlateauc_mean 评估的样本。

        从预处理的 Parquet 文件加载滑动窗口数据，并转换为 numpy 数组。

        参数:
            max_seq_len: 最大序列长度（窗口大小）

        返回:
            user_sequence: 技能序列，shape=(num_samples, max_seq_len)
            user_response: 响应序列，shape=(num_samples, max_seq_len)
            user_mask: 预测掩码，shape=(num_samples, max_seq_len)，1 表示需要预测
            user_id_sequence: 用户ID序列，shape=(num_samples, max_seq_len)
            late_group_id: 题目级分组ID，shape=(num_samples, max_seq_len)
            user_true_labels: 真实标签序列，shape=(num_samples, max_seq_len)
        """
        import numpy as np
        import polars as pl

        # 从预处理文件加载长格式数据
        data = self.data_src.get_windowlate_data()

        if data is None:
            raise ValueError(
                "No windowlate data available. Please re-run preprocessing with K-fold labels."
            )

        required_cols = [
            "sample_id",
            "position",
            "skill",
            "response",
            "mask",
            "user_id",
            "group_id",
            "true_label",
        ]
        lazy_data = data.select(required_cols)
        stats = lazy_data.select(
            [
                pl.col("sample_id").n_unique().alias("num_samples"),
                pl.col("sample_id").max().alias("max_sample_id"),
            ]
        ).collect(engine="streaming")
        num_samples = int(stats["num_samples"][0])

        if num_samples == 0:
            raise ValueError(
                "No windowlate data available. Please re-run preprocessing with K-fold labels."
            )

        # 初始化数组
        user_sequence = np.zeros((num_samples, max_seq_len), dtype=np.int32)
        user_response = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_mask = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_id_sequence = np.zeros((num_samples, max_seq_len), dtype=np.int32)
        late_group_id = np.full((num_samples, max_seq_len), -1, dtype=np.int64)
        user_true_labels = np.zeros((num_samples, max_seq_len), dtype=np.int8)

        sample_pos = lazy_data.select(["sample_id", "position"]).collect(
            engine="streaming"
        )
        sample_ids = sample_pos["sample_id"].to_numpy()
        positions = sample_pos["position"].to_numpy()

        user_sequence[sample_ids, positions] = (
            lazy_data.select("skill").collect(engine="streaming")["skill"].to_numpy()
        )
        user_response[sample_ids, positions] = (
            lazy_data.select("response")
            .collect(engine="streaming")["response"]
            .to_numpy()
        )
        user_mask[sample_ids, positions] = (
            lazy_data.select("mask").collect(engine="streaming")["mask"].to_numpy()
        )
        user_id_sequence[sample_ids, positions] = (
            lazy_data.select("user_id")
            .collect(engine="streaming")["user_id"]
            .to_numpy()
        )
        late_group_id[sample_ids, positions] = (
            lazy_data.select("group_id")
            .collect(engine="streaming")["group_id"]
            .to_numpy()
        )
        user_true_labels[sample_ids, positions] = (
            lazy_data.select("true_label")
            .collect(engine="streaming")["true_label"]
            .to_numpy()
        )

        self.logger.debug(
            f"Loaded windowlate data: samples={num_samples}, max_seq_len={max_seq_len}"
        )

        return (
            user_sequence,
            user_response,
            user_mask,
            user_id_sequence,
            late_group_id,
            user_true_labels,
        )

    def create_windowlate_iterable_dataset(
        self, max_seq_len: int, batch_read_rows: int = 200_000
    ) -> WindowlateIterableDataset:
        parquet_path = os.path.join(
            self.data_src.data_folder, f"{self.data_src.dataset}_windowlate.parquet"
        )

        return WindowlateIterableDataset(
            parquet_path=parquet_path,
            max_seq_len=max_seq_len,
            batch_read_rows=batch_read_rows,
        )
