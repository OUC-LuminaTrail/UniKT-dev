import os

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import (
    DataSource,
    build_question_data_from_cleared,
    map_to_continuous_ids,
    restrains_sequence_length,
)

logger = get_logger(__name__)


@register_data_source("assistments12")
class Assistments2012Data(DataSource):
    """ASSISTments 2012 dataset handler."""

    def __init__(self, args):
        super().__init__(
            dataset="assistments12",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments12.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(
            self.data_folder, "raw", "2012-2013-data-with-predictions-4-final.csv"
        )

    @override
    def load_src_data(self):
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        logger.info(f"Loading raw data from: {self.raw_data_path}")
        self.raw_data = pl.read_csv(
            self.raw_data_path,
            encoding="latin1",
            ignore_errors=True,
            try_parse_dates=False,
            null_values=["NA", ""],
        ).lazy()

    @override
    def clear_data(self):
        logger.info("Processing Data...")
        if self.raw_data is None:
            try:
                self.load_src_data()
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Raw data not found. Please fetch the data first."
                )

        data = self.raw_data.drop(
            [
                "problem_log_id",
                "skill",
                # "problem_id",
                # "user_id",
                # "assignment_id",
                "assistment_id",
                # "start_time",
                "end_time",
                "problem_type",
                "original",
                # "correct",
                "bottom_hint",
                # "hint_count",
                "actions",
                # "attempt_count",
                "ms_first_response",
                "tutor_mode",
                "sequence_id",
                "student_class_id",
                "position",
                "type",
                "base_sequence_id",
                # "skill_id",
                "teacher_id",
                "school_id",
                "overlap_time",
                # "template_id",
                "answer_id",
                "answer_text",
                "first_action",
                "problemlogid",
                "Average_confidence(FRUSTRATED)",
                "Average_confidence(CONFUSED)",
                "Average_confidence(CONCENTRATING)",
                "Average_confidence(BORED)",
            ]
        )

        data = data.rename(
            {
                "user_id": "user",
                "problem_id": "question",
                "correct": "label",
                "assignment_id": "assignment",
                "skill_id": "skill",
                "template_id": "template",
            }
        )

        data = data.unique()
        data = data.collect()

        data = data.sort(["user", "start_time"])
        data = data.with_columns([pl.col("user").cast(pl.Int32)])
        data = data.filter(pl.col("skill").is_not_null())
        data = data.filter(pl.col("label").is_in([0, 1]))

        data = restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )

        data = map_to_continuous_ids(
            data, columns=["user", "question", "skill", "assignment", "template"]
        )

        self.cleared_data = data.clone()
        self.sequence_data = data.clone()
        self.question_data = build_question_data_from_cleared(
            self.cleared_data, skill_column="skill", question_column="question"
        )

        self.add_metadatas(
            {
                "num_users": self.cleared_data["user"].n_unique(),
                "num_questions": self.question_data["question"].n_unique(),
                "num_skills": self.question_data["skill"].n_unique(),
                "num_assignments": self.question_data["assignment"].n_unique(),
                "num_templates": self.question_data["template"].n_unique(),
                "max_seq_len": self.args.max_seq_len,
                "min_seq_len": self.args.min_seq_len,
                "columns": data.columns,
            }
        )


__all__ = ["Assistments2012Data"]
