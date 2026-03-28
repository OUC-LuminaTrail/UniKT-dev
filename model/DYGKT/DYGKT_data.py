"""Data pipeline for DYGKT.

This module reconstructs per-interaction neighborhood features that match the
original DyGKT implementation semantics while keeping kt-exp-graph interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


@dataclass
class _InteractionRecord:
    user_id: int
    seq_len: int
    question_seq: list[int]
    correctness_seq: list[int]
    time_seq: list[int]


class DYGKTDataset(Dataset):
    """Per-interaction dataset with prebuilt user/question histories.

    Notes
    -----
    - Interaction index starts from 1 to reserve 0 as padding index.
    - Historical index lists store interaction indices and are padded with 0.
    - For source(user) node, neighbors are historical questions.
    - For destination(question) node, neighbors are historical users.
    """

    def __init__(
        self,
        dataset_config: dict[str, Any],
        data_all: list[dict[str, Any]],
        q_table: np.ndarray,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset_config = dataset_config
        self.data_all = data_all
        self.q_table = q_table
        self.device = device

        self.num_neighbor = int(self.dataset_config["num_neighbor"])

        self.dataset_converted: dict[str, list[Any]] = {
            "idx": [],
            "user": [],
            "question": [],
            "idx_in_seq": [],
            "time": [],
            "correctness": [],
            "user_his_seq": [],
            "user_his_snq_seq": [],
            "user_his_snd_seq": [],
            "user_his_snk_seq": [],
            "que_his_seq": [],
            "que_his_qn_seq": [],
        }

        self.base_tensors: dict[str, torch.Tensor] = {}
        self.lookup_tensors: dict[str, torch.Tensor] = {}

        self.process_dataset()

    def __len__(self) -> int:
        return len(self.dataset_converted["idx"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}

        # Base scalar fields for the current interaction.
        for key in ["idx", "user", "question", "idx_in_seq"]:
            result[key] = self.base_tensors[key][index]
        result["time"] = self.base_tensors["time"][index].float()
        result["correctness"] = self.base_tensors["correctness"][index].float()

        user_his_idx = self.dataset_converted["user_his_seq"][index]
        que_his_idx = self.dataset_converted["que_his_seq"][index]

        user_pad = user_his_idx + [0] * (self.num_neighbor - len(user_his_idx))
        que_pad = que_his_idx + [0] * (self.num_neighbor - len(que_his_idx))

        user_his_idx_t = torch.tensor(user_pad, dtype=torch.long)
        que_his_idx_t = torch.tensor(que_pad, dtype=torch.long)

        user_his_last_idx = torch.tensor(len(user_his_idx), dtype=torch.long)
        que_his_last_idx = torch.tensor(len(que_his_idx), dtype=torch.long)

        # Original compatibility fields.
        result["user_his_time_seq"] = self.lookup_tensors["time"][user_his_idx_t].float()
        result["user_his_correctness_seq"] = self.lookup_tensors["correctness"][user_his_idx_t].float()
        result["user_his_last_idx"] = user_his_last_idx

        result["que_his_time_seq"] = self.lookup_tensors["time"][que_his_idx_t].float()
        result["que_his_correctness_seq"] = self.lookup_tensors["correctness"][que_his_idx_t].float()
        result["que_his_last_idx"] = que_his_last_idx

        for key in ["user_his_snq_seq", "user_his_snd_seq", "user_his_snk_seq", "que_his_qn_seq"]:
            key_data = self.dataset_converted[key][index]
            padded = key_data + [0] * (self.num_neighbor - len(key_data))
            result[key] = torch.tensor(padded, dtype=torch.long)

        # DyGKT-native fields.
        # Source=user, so user history neighbors are question nodes.
        result["src_neighbor_node_ids"] = self.lookup_tensors["question"][user_his_idx_t].long()
        result["src_neighbor_times"] = self.lookup_tensors["time"][user_his_idx_t].float()
        result["src_neighbor_edge_feats"] = self.lookup_tensors["correctness"][user_his_idx_t].float()
        result["src_neighbor_len"] = user_his_last_idx

        # Destination=question, so question history neighbors are user nodes.
        result["dst_neighbor_node_ids"] = self.lookup_tensors["user"][que_his_idx_t].long()
        result["dst_neighbor_times"] = self.lookup_tensors["time"][que_his_idx_t].float()
        result["dst_neighbor_edge_feats"] = self.lookup_tensors["correctness"][que_his_idx_t].float()
        result["dst_neighbor_len"] = que_his_last_idx

        return result

    def process_dataset(self) -> None:
        self.convert_dataset()
        self.dataset2tensor()

    def convert_dataset(self) -> None:
        """Convert per-user sequences into per-interaction records."""
        que_sim_by_concept = ((self.q_table @ self.q_table.T) > 0).astype(int)

        num_question = int(self.dataset_config["num_question"])
        num_neighbor = int(self.dataset_config["num_neighbor"])

        logger.info("DYGKT: building interaction records...")

        # Reserve 0 as pad index.
        n = 1
        que_his_seqs: dict[int, list[tuple[int, int]]] = {}

        for user_data in self.data_all:
            user_id = num_question + int(user_data["user_id"])
            seq_len = int(user_data["seq_len"])
            question_seq = user_data["question_seq"][:seq_len]
            correctness_seq = user_data["correctness_seq"][:seq_len]
            time_seq = user_data["time_seq"][:seq_len]

            for i, (q_id, t, c) in enumerate(zip(question_seq, time_seq, correctness_seq)):
                q_id = int(q_id)
                t = int(t)
                c = int(c)

                if q_id not in que_his_seqs:
                    que_his_seqs[q_id] = []
                que_his_seqs[q_id].append((n, t))

                self.dataset_converted["idx"].append(n)
                self.dataset_converted["user"].append(user_id)
                self.dataset_converted["question"].append(q_id)
                self.dataset_converted["idx_in_seq"].append(i)
                self.dataset_converted["time"].append(t)
                self.dataset_converted["correctness"].append(c)

                # Keep exactly the source behavior from original code.
                user_his_seq = (
                    list(range(n - i, n))
                    if i < num_neighbor
                    else list(range(n - num_neighbor, n))
                )
                self.dataset_converted["user_his_seq"].append(user_his_seq)

                question_seq_ = question_seq[
                    0 if (i <= num_neighbor) else (i - num_neighbor) : i
                ]
                user_his_snd_seq = [int(q == q_id) for q in question_seq_]
                user_his_snk_seq = [int(que_sim_by_concept[q, q_id]) for q in question_seq_]

                self.dataset_converted["user_his_snq_seq"].append(user_his_snd_seq)
                self.dataset_converted["user_his_snd_seq"].append(user_his_snd_seq)
                self.dataset_converted["user_his_snk_seq"].append(user_his_snk_seq)

                self.dataset_converted["que_his_seq"].append([])
                self.dataset_converted["que_his_qn_seq"].append([])

                n += 1

        logger.info("DYGKT: building question histories...")

        for i, current_idx in enumerate(self.dataset_converted["idx"]):
            q_id = self.dataset_converted["question"][i]
            t = self.dataset_converted["time"][i]

            que_his_seq = [
                x[0]
                for x in sorted(
                    [y for y in que_his_seqs[q_id] if y[1] < t],
                    key=lambda z: z[1],
                )
            ]
            if len(que_his_seq) >= num_neighbor:
                que_his_seq = que_his_seq[-num_neighbor:]

            self.dataset_converted["que_his_seq"][i] = que_his_seq

    def dataset2tensor(self) -> None:
        self.base_tensors = {
            "idx": torch.tensor(self.dataset_converted["idx"], dtype=torch.long),
            "user": torch.tensor(self.dataset_converted["user"], dtype=torch.long),
            "question": torch.tensor(self.dataset_converted["question"], dtype=torch.long),
            "idx_in_seq": torch.tensor(self.dataset_converted["idx_in_seq"], dtype=torch.long),
            "time": torch.tensor(self.dataset_converted["time"], dtype=torch.long),
            "correctness": torch.tensor(self.dataset_converted["correctness"], dtype=torch.long),
        }

        # Build lookup tensors with index 0 reserved for padding.
        max_idx = int(self.base_tensors["idx"].max().item()) if len(self) > 0 else 0
        self.lookup_tensors = {
            "user": torch.zeros(max_idx + 1, dtype=torch.long),
            "question": torch.zeros(max_idx + 1, dtype=torch.long),
            "time": torch.zeros(max_idx + 1, dtype=torch.long),
            "correctness": torch.zeros(max_idx + 1, dtype=torch.long),
        }

        idx = self.base_tensors["idx"]
        self.lookup_tensors["user"][idx] = self.base_tensors["user"]
        self.lookup_tensors["question"][idx] = self.base_tensors["question"]
        self.lookup_tensors["time"][idx] = self.base_tensors["time"]
        self.lookup_tensors["correctness"][idx] = self.base_tensors["correctness"]


class DYGKTModelData(QuestionModelData):
    """Data adapter for DYGKT in kt-exp-graph."""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args: Any):
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        num_neighbor = int(getattr(args, "num_neighbor", 50))

        q_table = self.build_relationship_matrix(("question", "has", "skill"))
        if q_table is None or q_table.size == 0:
            raise ValueError("DYGKT requires a non-empty question-skill matrix.")

        num_questions = int(q_table.shape[0])

        dataset_config = {
            "num_question": num_questions,
            "num_neighbor": num_neighbor,
        }

        question_sequences, user_responses, user_masks, user_id_sequences = self.load_sequence_data()
        time_sequences = self._load_time_sequences(question_sequences.shape)

        if fold_idx is not None:
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(f"fold_idx {fold_idx} out of range [0, {kfold_n_splits})")

            logger.info(f"K-fold: fold {fold_idx + 1}/{kfold_n_splits}")
            train_data, val_data, test_data = self.split_kfold_data(
                question_sequences,
                user_responses,
                user_masks,
                time_sequences,
                user_id_sequences,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("fold_idx must be specified")

        train_records = self._build_interaction_records(*train_data)
        val_records = self._build_interaction_records(*val_data)
        test_records = self._build_interaction_records(*test_data)

        train_dataset = DYGKTDataset(dataset_config, train_records, q_table)
        val_dataset = DYGKTDataset(dataset_config, val_records, q_table)
        test_dataset = DYGKTDataset(dataset_config, test_records, q_table)

        logger.info(
            "Train: %s, Val: %s, Test: %s",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )

        question_skill_ids = self._build_primary_skill_ids(q_table)
        num_users = int(self.data_src.get_metadata("num_split_question_users"))

        model_metadata = {
            "num_questions": num_questions,
            "num_users": num_users,
            "question_skill_ids": question_skill_ids,
            "num_neighbor": num_neighbor,
        }

        return train_dataset, val_dataset, test_dataset, model_metadata

    def _build_primary_skill_ids(self, q_table: np.ndarray) -> np.ndarray:
        """Build one skill id per question from the question-skill matrix."""
        # For multi-skill questions, use the first active skill index.
        has_skill = q_table > 0
        first_skill = has_skill.argmax(axis=1).astype(np.int64)
        # Questions with no linked skill default to 0.
        no_skill_mask = has_skill.sum(axis=1) == 0
        first_skill[no_skill_mask] = 0
        return first_skill

    def _load_time_sequences(self, target_shape: tuple[int, int]) -> np.ndarray:
        """Load timestamp matrix aligned with split sequences."""
        num_users, max_seq_len = target_shape
        timestamps = np.zeros((num_users, max_seq_len), dtype=np.int64)

        split_data = self.data_src.get_split_question_sequence_data().to_pandas()
        time_col = None
        for candidate in ["timestamp", "startTime", "start_time", "time"]:
            if candidate in split_data.columns:
                time_col = candidate
                break

        if time_col is None:
            logger.warning("No time column found in split data, rebuilding from sequence data.")
            reconstructed = self._rebuild_split_timestamps_from_sequence(target_shape)
            if reconstructed is not None:
                return reconstructed

            logger.warning("Timestamp reconstruction failed, using synthetic hourly timestamps.")
            for u in range(num_users):
                timestamps[u, :] = np.arange(max_seq_len, dtype=np.int64) * 3600
            return timestamps

        ts_series = split_data[time_col]
        if np.issubdtype(ts_series.dtype, np.datetime64):
            ts = (ts_series.astype("int64") // 10**9).to_numpy(dtype=np.int64)
        else:
            import pandas as pd

            ts_numeric = pd.to_numeric(ts_series, errors="coerce")
            nan_count = int(ts_numeric.isna().sum())
            if nan_count > 0:
                logger.warning(
                    "Time column '%s' has %s invalid values; fill with 0.",
                    time_col,
                    nan_count,
                )
                ts_numeric = ts_numeric.fillna(0)
            ts = ts_numeric.to_numpy(dtype=np.int64)

        users = split_data["user"].to_numpy(dtype=np.int64)
        seq_pos = split_data["seq_pos"].to_numpy(dtype=np.int64)

        valid = (
            (users >= 0)
            & (users < num_users)
            & (seq_pos >= 0)
            & (seq_pos < max_seq_len)
        )
        timestamps[users[valid], seq_pos[valid]] = ts[valid]
        return timestamps

    def _rebuild_split_timestamps_from_sequence(
        self,
        target_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Rebuild split timestamps from sequence data when split file has no time."""
        try:
            import polars as pl
        except Exception as e:
            logger.warning("Polars unavailable for timestamp reconstruction: %s", e)
            return None

        num_users, max_seq_len = target_shape
        min_seq_len = int(self.data_src.get_metadata("min_seq_len"))

        sequence_data = self.data_src.get_sequence_data()
        if "timestamp" not in sequence_data.columns:
            logger.warning("Sequence data has no timestamp column.")
            return None

        data = sequence_data.with_columns(
            pl.int_range(pl.len()).over("user").alias("seq_pos")
        ).join(
            sequence_data.group_by("user").agg(pl.len().alias("seq_len")),
            on="user",
            how="left",
        )

        data = data.with_columns((pl.col("seq_pos") // max_seq_len).alias("split_idx")).with_columns(
            pl.when(pl.col("seq_pos") + max_seq_len >= pl.col("seq_len"))
            .then(pl.col("seq_len") - pl.col("split_idx") * max_seq_len)
            .otherwise(max_seq_len)
            .alias("split_len")
        )

        valid_splits = (
            data.filter(pl.col("split_len") >= min_seq_len)
            .select(["user", "split_idx"])
            .unique()
            .with_row_index("new_user_id")
            .sort(["user", "split_idx"])
        )

        split_data = data.join(valid_splits, on=["user", "split_idx"], how="inner")
        split_data = split_data.with_columns(
            [
                pl.col("new_user_id").cast(pl.Int32).alias("user"),
                (pl.col("seq_pos") % max_seq_len).alias("seq_pos"),
            ]
        ).select(["user", "seq_pos", "timestamp"])

        ts_matrix = np.zeros((num_users, max_seq_len), dtype=np.int64)
        users = split_data["user"].to_numpy()
        seq_pos = split_data["seq_pos"].to_numpy()
        ts = split_data["timestamp"].to_numpy()

        valid = (
            (users >= 0)
            & (users < num_users)
            & (seq_pos >= 0)
            & (seq_pos < max_seq_len)
        )
        ts_matrix[users[valid], seq_pos[valid]] = ts[valid].astype(np.int64)

        if int((ts_matrix != 0).sum()) == 0:
            logger.warning("Rebuilt timestamps are all zeros; fallback required.")
            return None

        return ts_matrix

    def _build_interaction_records(
        self,
        question_sequences: np.ndarray,
        user_responses: np.ndarray,
        user_masks: np.ndarray,
        time_sequences: np.ndarray,
        user_id_sequences: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Convert sequence arrays to DyGKT interaction records."""
        records: list[dict[str, Any]] = []

        for idx, (q_seq, r_seq, mask_seq, t_seq, uid_seq) in enumerate(
            zip(
                question_sequences,
                user_responses,
                user_masks,
                time_sequences,
                user_id_sequences,
            )
        ):
            seq_len = int(np.asarray(mask_seq).sum())
            if seq_len <= 0:
                continue

            valid_uid = np.asarray(uid_seq)[:seq_len]
            if valid_uid.size > 0:
                user_id = int(valid_uid[0])
            else:
                user_id = idx

            records.append(
                {
                    "user_id": user_id,
                    "seq_len": seq_len,
                    "question_seq": np.asarray(q_seq)[:seq_len].astype(np.int64).tolist(),
                    "correctness_seq": np.asarray(r_seq)[:seq_len].astype(np.int64).tolist(),
                    "time_seq": np.asarray(t_seq)[:seq_len].astype(np.int64).tolist(),
                }
            )

        return records
