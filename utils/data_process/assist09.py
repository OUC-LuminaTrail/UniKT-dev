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


@register_data_source("assistments09")
class Assistments2009Data(DataSource):
    """Assistments 2009-2010 dataset handler.

    Dataset source: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
    """

    def __init__(self, args):
        super().__init__(
            dataset="assistments09",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments09.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(
            self.data_folder, "raw", "skill_builder_data_corrected_collapsed.csv"
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
            null_values=["NA"],
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

        schema_names = self.raw_data.collect_schema().names()
        data = self.raw_data.drop("") if "" in schema_names else self.raw_data

        data = data.drop(
            [
                # "order_id",
                # "assignment_id",
                # "user_id",
                "assistment_id",
                # "problem_id",
                # "original",
                # "correct",
                # "attempt_count",
                "ms_first_response",
                "tutor_mode",
                "answer_type",
                "sequence_id",
                "student_class_id",
                "position",
                "type",
                "base_sequence_id",
                # "skill_id",
                "skill_name",
                "teacher_id",
                "school_id",
                # "hint_count",
                "hint_total",
                "overlap_time",
                # "template_id",
                "answer_id",
                "answer_text",
                "first_action",
                "bottom_hint",
                "opportunity",
                "opportunity_original",
            ]
        )

        data = data.rename(
            {
                "correct": "label",
                "user_id": "user",
                "problem_id": "question",
                "assignment_id": "assignment",
                "skill_id": "skill",
                "template_id": "template",
            }
        )

        data = data.filter(
            pl.col("user").is_not_null()
            & pl.col("skill").is_not_null()
            & pl.col("label").is_not_null()
        )

        data = data.unique().collect()
        data = data.sort(["user", "order_id"])
        data = restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )
        data = map_to_continuous_ids(
            data, columns=["user", "question", "assignment", "template"]
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


__all__ = ["Assistments2009Data"]
