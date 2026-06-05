import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class ReKTDataset(Dataset):
    def __init__(self, questions, skills, responses, masks):
        self.questions = questions
        self.skills = skills
        self.responses = responses
        self.masks = masks

    def __getitem__(self, index):
        return (
            torch.tensor(self.questions[index], dtype=torch.long),
            torch.tensor(self.skills[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.questions)


def build_combined_skill_mapping(data_src: DataSource):
    """构建组合技能映射：每个多技能组合视为一个新的技能。"""
    question_data = data_src.get_relation("question_skill")
    if isinstance(question_data, pl.LazyFrame):
        question_data = question_data.collect()

    question_skills_df = question_data.group_by("question").agg(
        pl.col("skill").unique().sort().alias("skills")
    )

    questions = question_skills_df["question"].to_list()
    skill_lists = question_skills_df["skills"].to_list()
    skill_tuples = [tuple(s) for s in skill_lists]

    unique_combos = sorted(set(skill_tuples))
    combo_to_id = {combo: i for i, combo in enumerate(unique_combos)}
    num_combined_skills = len(combo_to_id)

    num_questions = data_src.get_metadata("num_questions")
    question_to_combined = np.zeros(num_questions, dtype=np.int64)
    for q, st in zip(questions, skill_tuples):
        question_to_combined[q] = combo_to_id[st]

    logger.info(
        f"Combined skill mapping: {num_combined_skills} unique combinations "
        f"from {len(skill_tuples)} questions"
    )

    return question_to_combined, num_combined_skills


class ReKTModelData(QuestionModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        user_sequence, user_response, user_mask, _ = self.load_sequence_data()

        question_to_combined, num_combined_skills = build_combined_skill_mapping(
            self.data_src
        )
        user_skill_sequence = question_to_combined[user_sequence]

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
                user_skill_sequence,
                user_response,
                user_mask,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("fold_idx must be specified for K-fold cross-validation")

        train_dataset = ReKTDataset(
            train_data[0], train_data[1], train_data[2], train_data[3]
        )
        val_dataset = ReKTDataset(val_data[0], val_data[1], val_data[2], val_data[3])
        test_dataset = ReKTDataset(
            test_data[0], test_data[1], test_data[2], test_data[3]
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            {"num_combined_skills": num_combined_skills},
        )
