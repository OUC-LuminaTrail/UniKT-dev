"""Question-level data preparation for ReKTP."""

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

    def __init__(self, questions, responses, masks):
        self.questions = torch.from_numpy(np.asarray(questions)).long()
        self.responses = torch.from_numpy(np.asarray(responses)).long()
        self.masks = torch.from_numpy(np.asarray(masks)).bool()

    def __getitem__(self, index):
        return self.questions[index], self.responses[index], self.masks[index]

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


class ReKTPModelData(QuestionModelData):
    """Prepare original question sequences and a separate question-KC view."""

    def __init__(self, data_src: DataSource, cache: bool = False):
        super().__init__(data_src, cache=cache)

    @override
    @QuestionModelData.disk_cache("rektp_data")
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
        train_data, val_data, test_data = self.split_kfold_data(
            questions, responses, masks, fold_idx=fold_idx
        )
        question_skill_ids, question_skill_mask = build_question_skill_table(
            self.data_src
        )

        logger.info("Using K-fold: fold %d/%d", fold_idx + 1, kfold_n_splits)
        return (
            ReKTPDataset(*train_data),
            ReKTPDataset(*val_data),
            ReKTPDataset(*test_data),
            {
                "question_skill_ids": question_skill_ids,
                "question_skill_mask": question_skill_mask,
            },
        )


__all__ = [
    "ReKTPDataset",
    "ReKTPModelData",
    "build_question_skill_table",
]
