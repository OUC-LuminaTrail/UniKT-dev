"""Question-level data preparation for ReKTP."""

import math
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class ReKTPDataset(Dataset):
    """One position per original question interaction."""

    def __init__(self, questions, responses, times, masks):
        self.questions = torch.from_numpy(np.asarray(questions)).long()
        self.responses = torch.from_numpy(np.asarray(responses)).long()
        self.times = torch.from_numpy(np.asarray(times)).double()
        self.masks = torch.from_numpy(np.asarray(masks)).bool()

    def __getitem__(self, index):
        return (
            self.questions[index],
            self.responses[index],
            self.times[index],
            self.masks[index],
        )

    def __len__(self):
        return len(self.questions)


def build_question_skill_table(data_src: DataSource) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded, permutation-stable KC set for every question."""
    relation = data_src.get_relation("question_skill")
    if isinstance(relation, pl.LazyFrame):
        relation = relation.collect()

    grouped = relation.group_by("question").agg(
        pl.col("skill").unique().sort().alias("skills")
    )
    num_questions = int(data_src.get_metadata("num_questions"))
    num_skills = int(data_src.get_metadata("num_skills"))
    max_skills = max(1, int(grouped.select(pl.col("skills").list.len().max()).item()))

    skill_ids = np.full((num_questions, max_skills), num_skills, dtype=np.int64)
    skill_mask = np.zeros((num_questions, max_skills), dtype=np.bool_)
    for row in grouped.iter_rows(named=True):
        question = int(row["question"])
        skills = np.asarray(row["skills"], dtype=np.int64)
        if question < 0 or question >= num_questions:
            raise ValueError(f"Question id {question} is outside metadata range")
        if skills.size and (skills.min() < 0 or skills.max() >= num_skills):
            raise ValueError(f"Question {question} has an out-of-range skill id")
        skill_ids[question, : skills.size] = skills
        skill_mask[question, : skills.size] = True

    missing = np.flatnonzero(~skill_mask.any(axis=1))
    if missing.size:
        logger.warning("ReKTP found %d questions without a KC relation", missing.size)
    logger.info("ReKTP question-KC table: max_skills_per_question=%d", max_skills)
    return skill_ids, skill_mask


def derive_max_gap_bins(time_seqs: np.ndarray) -> int:
    """Derive the log2-gap bucket count covering the largest time span."""
    max_span = max(1.0, float(time_seqs.max()))
    return max(2, int(math.floor(math.log2(max_span))) + 2)


class ReKTPModelData(QuestionModelData):
    """Prepare original question sequences and a separate question-KC view."""

    @override
    def prepare_data(self, rc: Any):
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        if fold_idx is None:
            raise ValueError("fold_idx must be specified for K-fold cross-validation")

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        if fold_idx >= kfold_n_splits:
            raise ValueError(
                f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
            )

        questions, responses, masks, _ = self.load_sequence_data()
        times = self._build_time_sequences()
        train_data, val_data, test_data = self.split_kfold_data(
            questions, responses, times, masks, fold_idx=fold_idx
        )
        question_skill_ids, question_skill_mask = build_question_skill_table(
            self.data_src
        )
        max_gap_bins = derive_max_gap_bins(times)
        logger.info(
            "Derived max_gap_bins=%d from the largest intra-sequence time span",
            max_gap_bins,
        )
        logger.info("Using K-fold: fold %d/%d", fold_idx + 1, kfold_n_splits)
        return (
            ReKTPDataset(*train_data),
            ReKTPDataset(*val_data),
            ReKTPDataset(*test_data),
            {
                "question_skill_ids": question_skill_ids,
                "question_skill_mask": question_skill_mask,
                "max_gap_bins": max_gap_bins,
            },
        )

    def _build_time_sequences(self) -> np.ndarray:
        """Return per-position interaction times in seconds (float64).

        Real timestamps are used when the split data carries a ``timestamp``
        column (milliseconds converted to seconds). ``assistments09`` stores
        ``order_id`` there instead of wall-clock time, so its
        ``ms_first_response`` dwell times are accumulated like the original
        HawkesKT implementation. Sequences without a usable timestamp fall
        back to position indices.
        """
        q_data = self.data_src.get_split_question_sequence_data()
        num_users = q_data["user"].n_unique()
        max_seq_len = int(self.data_src.get_metadata("max_seq_len"))
        time_seqs = np.zeros((num_users, max_seq_len), dtype=np.float64)
        user_indices = q_data["user"].to_numpy()
        seq_positions = q_data["seq_pos"].to_numpy()
        columns = q_data.columns

        if self.data_src.dataset == "assistments09" and "ms_first_response" in columns:
            dwell = (
                q_data["ms_first_response"].fill_null(0).to_numpy().astype(np.float64)
                / 1000.0
            )
            cumshift = np.zeros(num_users, dtype=np.float64)
            for idx in np.lexsort((seq_positions, user_indices)):
                uid = user_indices[idx]
                pos = seq_positions[idx]
                time_seqs[uid, pos] = cumshift[uid]
                cumshift[uid] += dwell[idx] + 1.0
        elif "timestamp" in columns:
            time_seqs[user_indices, seq_positions] = (
                q_data["timestamp"].fill_null(0).to_numpy().astype(np.float64) / 1000.0
            )
        else:
            logger.warning(
                "No usable timestamp column for dataset %r; ReKTP time gaps "
                "fall back to position indices",
                self.data_src.dataset,
            )
            time_seqs[user_indices, seq_positions] = seq_positions.astype(np.float64)
        return time_seqs


__all__ = [
    "ReKTPDataset",
    "ReKTPModelData",
    "build_question_skill_table",
    "derive_max_gap_bins",
]
