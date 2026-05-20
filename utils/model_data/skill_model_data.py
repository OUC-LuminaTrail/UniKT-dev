import os
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import BaseModelData


class WindowlateIterableDataset(IterableDataset):
    """Stream windowlate samples from parquet."""

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
        self._num_samples = None
        self._num_row_groups = None

    def _init_metadata(self) -> None:
        """延迟初始化元数据"""
        if self._num_samples is None:
            import polars as pl

            stats = (
                pl.scan_parquet(self.parquet_path)
                .select(pl.col("sample_id").n_unique().alias("num_samples"))
                .collect(engine="streaming")
            )
            self._num_samples = int(stats["num_samples"][0])
            parquet_file = pq.ParquetFile(self.parquet_path)
            self._num_row_groups = parquet_file.num_row_groups

    def __len__(self) -> int:
        self._init_metadata()
        return self._num_samples

    def _build_single_tensor(
        self, sample: dict[str, np.ndarray]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """构建单个样本张量，返回 (sequence, response, mask, late_group_id, label, question)"""
        positions = sample["position"]

        sequence = np.zeros(self.max_seq_len, dtype=np.int64)
        response = np.zeros(self.max_seq_len, dtype=np.int64)
        mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        late_group_id = np.full(self.max_seq_len, -1, dtype=np.int64)
        label = np.zeros(self.max_seq_len, dtype=np.int64)
        question = np.zeros(self.max_seq_len, dtype=np.int64)

        sequence[positions] = sample["skill"]
        response[positions] = sample["response"]
        mask[positions] = sample["mask"].astype(np.bool_)
        late_group_id[positions] = sample["group_id"]
        label[positions] = sample["true_label"]
        question[positions] = sample["question"]

        return (
            torch.from_numpy(sequence),
            torch.from_numpy(response),
            torch.from_numpy(mask),
            torch.from_numpy(late_group_id),
            torch.from_numpy(label),
            torch.from_numpy(question),
        )

    def _read_batch_arrays(self, table: pa.Table) -> dict[str, np.ndarray]:
        """读取 Table 为 numpy 数组"""
        return {
            "sample_id": table.column("sample_id").to_numpy(),
            "position": table.column("position").to_numpy(),
            "skill": table.column("skill").to_numpy(),
            "question": table.column("question").to_numpy(),
            "response": table.column("response").to_numpy(),
            "mask": table.column("mask").to_numpy(),
            "group_id": table.column("group_id").to_numpy(),
            "true_label": table.column("true_label").to_numpy(),
        }

    def _iter_row_groups(
        self, row_group_indices: list[int]
    ) -> Iterator[dict[str, np.ndarray]]:
        """迭代指定的 row groups，返回批量数据"""
        parquet_file = pq.ParquetFile(self.parquet_path)

        for rg_idx in row_group_indices:
            table = parquet_file.read_row_group(rg_idx)
            yield self._read_batch_arrays(table)

    def _process_batch(
        self, batch: dict[str, np.ndarray]
    ) -> Iterator[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]:
        """处理一个批量数据，逐个 yield 样本"""
        sample_ids = batch["sample_id"]
        if sample_ids.size == 0:
            return

        boundaries = np.flatnonzero(sample_ids[1:] != sample_ids[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [sample_ids.size]))

        sample_data = {
            "position": None,
            "skill": None,
            "question": None,
            "response": None,
            "mask": None,
            "group_id": None,
            "true_label": None,
        }

        for start, end in zip(starts, ends, strict=False):
            sample_data["position"] = batch["position"][start:end]
            sample_data["skill"] = batch["skill"][start:end]
            sample_data["question"] = batch["question"][start:end]
            sample_data["response"] = batch["response"][start:end]
            sample_data["mask"] = batch["mask"][start:end]
            sample_data["group_id"] = batch["group_id"][start:end]
            sample_data["true_label"] = batch["true_label"][start:end]

            yield self._build_single_tensor(sample_data)

    def __iter__(self):
        self._init_metadata()
        worker_info = get_worker_info()

        if worker_info is not None and worker_info.num_workers > 0:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            all_row_groups = list(range(self._num_row_groups))
            row_group_indices = all_row_groups[worker_id::num_workers]
        else:
            row_group_indices = list(range(self._num_row_groups))

        for batch in self._iter_row_groups(row_group_indices):
            yield from self._process_batch(batch)


class SkillModelData(BaseModelData):
    """
    技能序列数据基类

    用于构建基于技能（skill/concept）的知识追踪模型数据
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    def _get_kfold_data(self):
        r"""重写：从技能序列数据获取 K-fold 标签。"""
        return self.data_src.get_split_skill_sequence_data()

    def build_sequence_data(self):
        r"""
        从切分后的技能序列数据加载用户技能序列

        返回:
            user_sequence: 用户技能ID序列，shape为(num_split_users, max_seq_len)
            user_response: 用户响应序列，shape为(num_split_users, max_seq_len)
            user_mask: 用户掩码序列，shape为(num_split_users, max_seq_len)
            user_id_sequence: 用户ID序列，shape为(num_split_users, max_seq_len)
            user_question: 用户题目ID序列，shape为(num_split_users, max_seq_len)
        """
        import numpy as np

        self.logger.info("Building skill sequences from split data...")

        data = self.data_src.get_split_skill_sequence_data().to_pandas()
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_users = data["user"].nunique()

        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)
        user_question = np.zeros((num_users, max_seq_len), dtype=int)

        user_indices = data["user"].values
        seq_positions = data["seq_pos"].values

        user_sequence[user_indices, seq_positions] = data["skill"].values
        user_id_sequence[user_indices, seq_positions] = user_indices
        user_response[user_indices, seq_positions] = data["label"].values
        user_mask[user_indices, seq_positions] = 1
        user_question[user_indices, seq_positions] = data["question"].values

        self.logger.debug(
            f"Built split skill sequences for {num_users} split users, max_len={max_seq_len}"
        )

        return user_sequence, user_response, user_mask, user_id_sequence, user_question

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
            user_question: 题目ID序列，shape=(num_samples, max_seq_len)
        """
        import numpy as np
        import polars as pl

        data = self.data_src.get_windowlate_data()

        if data is None:
            raise ValueError(
                "No windowlate data available. Please re-run preprocessing with K-fold labels."
            )

        required_cols = [
            "sample_id",
            "position",
            "skill",
            "question",
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

        user_sequence = np.zeros((num_samples, max_seq_len), dtype=np.int32)
        user_response = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_mask = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_id_sequence = np.zeros((num_samples, max_seq_len), dtype=np.int32)
        late_group_id = np.full((num_samples, max_seq_len), -1, dtype=np.int64)
        user_true_labels = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_question = np.zeros((num_samples, max_seq_len), dtype=np.int32)

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
        user_question[sample_ids, positions] = (
            lazy_data.select("question")
            .collect(engine="streaming")["question"]
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
            user_question,
        )

    def create_windowlate_iterable_dataset(
        self,
        max_seq_len: int,
        batch_read_rows: int = 200_000,
    ) -> WindowlateIterableDataset:
        parquet_path = os.path.join(
            self.data_src.data_folder, f"{self.data_src.dataset}_windowlate.parquet"
        )

        return WindowlateIterableDataset(
            parquet_path=parquet_path,
            max_seq_len=max_seq_len,
            batch_read_rows=batch_read_rows,
        )
