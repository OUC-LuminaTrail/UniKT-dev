"""Data pipeline for DYGKT.

This module reconstructs per-interaction neighborhood features that match the
original DyGKT implementation semantics while keeping kt-exp-graph interfaces.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


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
        target_user_ids: set[int] | None = None,
        device: str | None = None,
        que_sim_matrix: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.dataset_config = dataset_config
        self.data_all = data_all
        self.q_table = q_table
        self.target_user_ids = target_user_ids
        self.device = device
        self.que_sim_matrix = que_sim_matrix  # Pre-computed similarity matrix

        self.num_neighbor = int(self.dataset_config["num_neighbor"])
        # Compatibility fields are expensive and not used by current DYGKT model/trainer.
        self.compat_fields = bool(self.dataset_config.get("compat_fields", False))

        self.dataset_converted: dict[str, list[Any]] = {
            "idx": [],
            "user": [],
            "question": [],
            "question_raw": [],
            "idx_in_seq": [],
            "time": [],
            "correctness": [],
            "user_his_seq": [],
            "que_his_seq": [],
        }
        if self.compat_fields:
            self.dataset_converted.update(
                {
                    "user_his_snq_seq": [],
                    "user_his_snd_seq": [],
                    "user_his_snk_seq": [],
                    "que_his_qn_seq": [],
                }
            )

        self.base_tensors: dict[str, torch.Tensor] = {}
        self.lookup_tensors: dict[str, torch.Tensor] = {}
        self.base_float_tensors: dict[str, torch.Tensor] = {}
        self.lookup_float_tensors: dict[str, torch.Tensor] = {}
        self.history_index_tensors: dict[str, torch.Tensor] = {}
        self.history_feature_tensors: dict[str, torch.Tensor] = {}
        self.history_len_tensors: dict[str, torch.Tensor] = {}
        self.target_positions: list[int] = []
        self.target_positions_tensor = torch.empty(0, dtype=torch.long)

        self.process_dataset()

    def __len__(self) -> int:
        return len(self.target_positions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = self.get_batch(torch.tensor([index], dtype=torch.long))
        return {k: v[0] for k, v in batch.items()}

    def collate_indices(self, batch_indices: list[int]) -> dict[str, torch.Tensor]:
        """Collate a batch of dataset positions into one vectorized dictionary batch."""
        return self.get_batch(batch_indices)

    def get_batch(self, batch_positions: list[int] | torch.Tensor) -> dict[str, torch.Tensor]:
        """Build one batch from dataset positions using vectorized tensor gather."""
        if not torch.is_tensor(batch_positions):
            batch_positions = torch.tensor(batch_positions, dtype=torch.long)
        else:
            batch_positions = batch_positions.long()

        data_indices = self.target_positions_tensor[batch_positions]
        return self._gather_batch_by_data_indices(data_indices)

    def _gather_batch_by_data_indices(self, data_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Gather all required fields for a batch from underlying tensor storages."""
        result: dict[str, torch.Tensor] = {}

        # Base scalar fields for current interactions.
        for key in ["idx", "user", "question", "idx_in_seq"]:
            result[key] = self.base_tensors[key][data_indices]
        result["time"] = self.base_float_tensors["time"][data_indices]
        result["correctness"] = self.base_float_tensors["correctness"][data_indices]

        user_his_idx_t = self.history_index_tensors["user_his_seq"][data_indices]
        que_his_idx_t = self.history_index_tensors["que_his_seq"][data_indices]

        user_his_last_idx = self.history_len_tensors["user_his_seq"][data_indices]
        que_his_last_idx = self.history_len_tensors["que_his_seq"][data_indices]

        # Original compatibility fields.
        result["user_his_time_seq"] = self.lookup_float_tensors["time"][user_his_idx_t]
        result["user_his_correctness_seq"] = self.lookup_float_tensors["correctness"][user_his_idx_t]
        result["user_his_last_idx"] = user_his_last_idx

        result["que_his_time_seq"] = self.lookup_float_tensors["time"][que_his_idx_t]
        result["que_his_correctness_seq"] = self.lookup_float_tensors["correctness"][que_his_idx_t]
        result["que_his_last_idx"] = que_his_last_idx

        if self.compat_fields:
            for key in ["user_his_snq_seq", "user_his_snd_seq", "user_his_snk_seq", "que_his_qn_seq"]:
                result[key] = self.history_feature_tensors[key][data_indices]

        # DyGKT-native fields.
        # Source=user, so user history neighbors are question nodes.
        result["src_neighbor_node_ids"] = self.lookup_tensors["question"][user_his_idx_t]
        result["src_neighbor_times"] = self.lookup_float_tensors["time"][user_his_idx_t]
        result["src_neighbor_edge_feats"] = self.lookup_float_tensors["correctness"][user_his_idx_t]
        result["src_neighbor_len"] = user_his_last_idx

        # Destination=question, so question history neighbors are user nodes.
        result["dst_neighbor_node_ids"] = self.lookup_tensors["user"][que_his_idx_t]
        result["dst_neighbor_times"] = self.lookup_float_tensors["time"][que_his_idx_t]
        result["dst_neighbor_edge_feats"] = self.lookup_float_tensors["correctness"][que_his_idx_t]
        result["dst_neighbor_len"] = que_his_last_idx

        return result

    def process_dataset(self) -> None:
        self.convert_dataset()
        self.dataset2tensor()

    def convert_dataset(self) -> None:
        """Convert per-user sequences into per-interaction records.
        
        Optimized version with vectorized operations for better performance.
        """
        # Use precomputed similarity matrix when available. For large datasets,
        # fallback to local on-the-fly similarity to avoid building huge NxN matrices.
        use_precomputed_similarity = self.que_sim_matrix is not None
        if use_precomputed_similarity:
            que_sim_by_concept = self.que_sim_matrix
            q_table_binary = None
        else:
            que_sim_by_concept = None
            q_table_binary = (self.q_table > 0).astype(np.int8)

        num_question = int(self.dataset_config["num_question"])
        num_neighbor = int(self.dataset_config["num_neighbor"])

        logger.info("DYGKT: building interaction records (optimized)...")

        # Reserve 0 as pad index.
        n = 1
        que_his_seqs: dict[int, list[tuple[int, int]]] = {}

        for user_data in self.data_all:
            # Reserve node id 0 as a global padding id.
            # Question ids are shifted by +1 and user ids start from num_question + 1.
            user_id = num_question + 1 + int(user_data["user_id"])
            seq_len = int(user_data["seq_len"])
            question_seq = user_data["question_seq"][:seq_len]
            correctness_seq = user_data["correctness_seq"][:seq_len]
            time_seq = user_data["time_seq"][:seq_len]

            # 🚀 Optimization: Convert to numpy arrays once per user
            question_seq_np = np.array(question_seq, dtype=np.int32)
            correctness_seq_np = np.array(correctness_seq, dtype=np.int8)
            # Keep 64-bit timestamps to avoid overflow on epoch-millisecond data.
            time_seq_np = np.array(time_seq, dtype=np.int64)

            for i in range(seq_len):
                q_id = int(question_seq_np[i])
                t = int(time_seq_np[i])
                c = int(correctness_seq_np[i])

                if q_id not in que_his_seqs:
                    que_his_seqs[q_id] = []
                que_his_seqs[q_id].append((n, t))

                self.dataset_converted["idx"].append(n)
                self.dataset_converted["user"].append(user_id)
                self.dataset_converted["question"].append(q_id + 1)
                self.dataset_converted["question_raw"].append(q_id)
                self.dataset_converted["idx_in_seq"].append(i)
                self.dataset_converted["time"].append(t)
                self.dataset_converted["correctness"].append(c)

                # 🚀 Optimization: Pre-compute range boundaries
                start_idx = max(1, n - min(i, num_neighbor))
                user_his_seq = list(range(start_idx, n))
                self.dataset_converted["user_his_seq"].append(user_his_seq)

                # 🚀 Optimization: Vectorized operations for similarity calculations
                if self.compat_fields:
                    if i == 0:
                        # Empty history
                        user_his_snd_seq = []
                        user_his_snk_seq = []
                    else:
                        start_pos = max(0, i - num_neighbor)
                        question_window = question_seq_np[start_pos:i]

                        # Vectorized comparison (10-50x faster than list comprehension)
                        user_his_snd_seq = (question_window == q_id).astype(np.int8).tolist()
                        if use_precomputed_similarity:
                            user_his_snk_seq = que_sim_by_concept[question_window, q_id].tolist()
                        else:
                            # Local concept overlap: similar iff shared concept exists.
                            window_concepts = q_table_binary[question_window]
                            current_concepts = q_table_binary[q_id]
                            user_his_snk_seq = ((window_concepts @ current_concepts) > 0).astype(np.int8).tolist()

                    self.dataset_converted["user_his_snq_seq"].append(user_his_snd_seq)
                    self.dataset_converted["user_his_snd_seq"].append(user_his_snd_seq)
                    self.dataset_converted["user_his_snk_seq"].append(user_his_snk_seq)

                self.dataset_converted["que_his_seq"].append([])
                if self.compat_fields:
                    self.dataset_converted["que_his_qn_seq"].append([])

                n += 1

        logger.info("DYGKT: building question histories (optimized)...")

        # Build question histories in O(total_interactions) after per-question sorting.
        # For equal timestamps, keep strict < t behavior by assigning histories before
        # inserting the current same-time group into history.
        total_interactions = len(self.dataset_converted["idx"])
        histories_by_interaction_idx: list[list[int]] = [[] for _ in range(n)]
        for seq_list in que_his_seqs.values():
            if not seq_list:
                continue
            seq_sorted = sorted(seq_list, key=lambda x: x[1])
            recent_hist: list[int] = []
            i = 0
            seq_len = len(seq_sorted)
            while i < seq_len:
                t = seq_sorted[i][1]
                j = i
                while j < seq_len and seq_sorted[j][1] == t:
                    j += 1

                shared_hist = list(recent_hist)
                for k in range(i, j):
                    interaction_idx = seq_sorted[k][0]
                    histories_by_interaction_idx[interaction_idx] = shared_hist

                recent_hist.extend(seq_sorted[k][0] for k in range(i, j))
                if len(recent_hist) > num_neighbor:
                    recent_hist = recent_hist[-num_neighbor:]
                i = j

        for i, interaction_idx in enumerate(self.dataset_converted["idx"]):
            self.dataset_converted["que_his_seq"][i] = histories_by_interaction_idx[interaction_idx]

        if self.target_user_ids is None:
            self.target_positions = list(range(len(self.dataset_converted["idx"])))
        else:
            self.target_positions = [
                i
                for i, user_id in enumerate(self.dataset_converted["user"])
                if user_id in self.target_user_ids
            ]

        self.target_positions_tensor = torch.tensor(self.target_positions, dtype=torch.long)

        logger.info(
            "DYGKT dataset built: total interactions=%s, target interactions=%s",
            len(self.dataset_converted["idx"]),
            len(self.target_positions),
        )

    def dataset2tensor(self) -> None:
        num_records = len(self.dataset_converted["idx"])

        self.base_tensors = {
            "idx": torch.tensor(self.dataset_converted["idx"], dtype=torch.long),
            "user": torch.tensor(self.dataset_converted["user"], dtype=torch.long),
            "question": torch.tensor(self.dataset_converted["question"], dtype=torch.long),
            "idx_in_seq": torch.tensor(self.dataset_converted["idx_in_seq"], dtype=torch.long),
            "time": torch.tensor(self.dataset_converted["time"], dtype=torch.long),
            "correctness": torch.tensor(self.dataset_converted["correctness"], dtype=torch.long),
        }
        self.base_float_tensors = {
            "time": self.base_tensors["time"].float(),
            "correctness": self.base_tensors["correctness"].float(),
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
        self.lookup_float_tensors = {
            "time": self.lookup_tensors["time"].float(),
            "correctness": self.lookup_tensors["correctness"].float(),
        }

        def _pad_sequences(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
            padded = np.zeros((num_records, self.num_neighbor), dtype=np.int64)
            lengths = np.zeros(num_records, dtype=np.int64)
            for i, seq in enumerate(sequences):
                seq_clip = seq[-self.num_neighbor :]
                seq_len = len(seq_clip)
                lengths[i] = seq_len
                if seq_len > 0:
                    padded[i, :seq_len] = seq_clip
            return torch.from_numpy(padded), torch.from_numpy(lengths)

        user_his_padded, user_his_len = _pad_sequences(self.dataset_converted["user_his_seq"])
        que_his_padded, que_his_len = _pad_sequences(self.dataset_converted["que_his_seq"])
        self.history_index_tensors = {
            "user_his_seq": user_his_padded,
            "que_his_seq": que_his_padded,
        }
        self.history_len_tensors = {
            "user_his_seq": user_his_len,
            "que_his_seq": que_his_len,
        }

        if self.compat_fields:
            for key in ["user_his_snq_seq", "user_his_snd_seq", "user_his_snk_seq", "que_his_qn_seq"]:
                padded, _ = _pad_sequences(self.dataset_converted[key])
                self.history_feature_tensors[key] = padded


class DYGKTModelData(QuestionModelData):
    """Data adapter for DYGKT in kt-exp-graph."""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args: Any):
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        num_neighbor = int(getattr(args, "num_neighbor", 50))

        # 🚀 OPTIMIZATION: Try to load from cache first
        use_cache = not bool(getattr(args, "no_cache", False))
        cache_dir = getattr(args, "cache_dir", None)
        if use_cache:
            from .dataset_cache import get_cache_key, load_cached_dataset

            cache_key = get_cache_key(args, fold_idx)
            cached_data = load_cached_dataset(cache_key, cache_dir=cache_dir)
            if cached_data is not None:
                logger.info("✅ Using cached dataset (skip preprocessing)")
                return cached_data
            logger.info("Cache miss, building dataset from scratch...")
        else:
            logger.info("DYGKT dataset cache disabled (--no_cache)")

        q_table = self.build_relationship_matrix(("question", "has", "skill"))
        if q_table is None or q_table.size == 0:
            raise ValueError("DYGKT requires a non-empty question-skill matrix.")

        num_questions = int(q_table.shape[0])

        dataset_config = {
            "num_question": num_questions,
            "num_neighbor": num_neighbor,
            "compat_fields": bool(getattr(args, "compat_fields", False)),
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

        # For large question vocab, full NxN similarity is too expensive.
        max_similarity_matrix_questions = int(
            getattr(args, "max_similarity_matrix_questions", 12000)
        )
        que_sim_matrix = None
        if num_questions <= max_similarity_matrix_questions:
            logger.info(
                "Pre-computing question similarity matrix (num_questions=%s)...",
                num_questions,
            )
            import time

            start_time = time.time()
            
            # 🚀 OPTIMIZATION: Use sparse matrix for 5-10x speedup
            try:
                from scipy.sparse import csr_matrix
                q_table_sparse = csr_matrix(q_table > 0, dtype=np.int8)
                que_sim_sparse = q_table_sparse @ q_table_sparse.T
                que_sim_matrix = (que_sim_sparse > 0).astype(np.int8).toarray()
                logger.info(
                    "Similarity matrix computed (sparse) in %.2fs",
                    time.time() - start_time,
                )
            except ImportError:
                # Fallback to dense computation if scipy not available
                logger.warning("scipy not available, using dense matrix computation")
                que_sim_matrix = ((q_table @ q_table.T) > 0).astype(np.int8)
                logger.info(
                    "Similarity matrix computed (dense) in %.2fs",
                    time.time() - start_time,
                )
        else:
            logger.info(
                "Skip full similarity matrix: num_questions=%s > threshold=%s. "
                "Use local on-the-fly similarity.",
                num_questions,
                max_similarity_matrix_questions,
            )

        train_dataset = DYGKTDataset(dataset_config, train_records, q_table, que_sim_matrix=que_sim_matrix)

        val_history_records = train_records + val_records
        val_target_user_ids = self._extract_user_ids(val_records, num_questions)
        val_dataset = DYGKTDataset(
            dataset_config,
            val_history_records,
            q_table,
            target_user_ids=val_target_user_ids,
            que_sim_matrix=que_sim_matrix,
        )

        test_history_records = train_records + val_records + test_records
        test_target_user_ids = self._extract_user_ids(test_records, num_questions)
        test_dataset = DYGKTDataset(
            dataset_config,
            test_history_records,
            q_table,
            target_user_ids=test_target_user_ids,
            que_sim_matrix=que_sim_matrix,
        )

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
            "question_id_offset": 1,
            "user_id_offset": num_questions + 1,
            "question_skill_ids": question_skill_ids,
            "question_features": q_table.astype(np.float32),
            "num_neighbor": num_neighbor,
        }

        # 🚀 OPTIMIZATION: Save to cache for future runs
        if use_cache:
            from .dataset_cache import save_cached_dataset

            dataset_tuple = (train_dataset, val_dataset, test_dataset, model_metadata)
            save_cached_dataset(cache_key, dataset_tuple, cache_dir=cache_dir)

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

    def _extract_user_ids(self, records: list[dict[str, Any]], num_questions: int) -> set[int]:
        """Extract encoded user ids for selecting target interactions."""
        if not records:
            return set()
        return {num_questions + 1 + int(record["user_id"]) for record in records}

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
