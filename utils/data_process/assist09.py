import os

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import (
    DataSource,
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
        """Clean data and build question_data and sequence_data."""
        logger.info("Processing ASSISTments 2009 data...")

        # Clean raw data
        cleaned_data = self._clean_raw_data()
        logger.debug(f"Cleaned data shape: {cleaned_data.shape}")

        # Build question ID mapping
        question_map_df = (
            cleaned_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
        )

        # Save question mapping for later use
        question_map = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )
        self._id_mappings["question"] = question_map
        logger.debug(f"Built question ID mapping: {len(question_map)} unique questions")

        # Apply question mapping to cleaned data
        mapped_data = (
            cleaned_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Build question_data
        # First, get unique question-assignment-template combos (preserving all questions)
        question_meta = mapped_data.select(
            ["question", "assignment", "template"]
        ).unique(subset=["question"])
        logger.debug(f"Question metadata shape: {question_meta.shape}")

        # Build base data with question-skill pairs
        base_question_data = mapped_data.select(["question", "skill"]).unique(
            subset=["question", "skill"], keep="first"
        )
        logger.debug(f"Base question_data shape: {base_question_data.shape}")

        # Split multi-skills
        # ASSISTments 09 skill format: "2_37_70" -> ["2", "37", "70"]
        split_skills = (
            base_question_data.with_columns(
                pl.col("skill").str.split("_").alias("skill_parts")
            )
            .explode("skill_parts")
            .with_columns(pl.col("skill_parts").cast(pl.String).alias("skill"))
            .select(["question", "skill"])
            .unique(subset=["question", "skill"], keep="first")
        )
        logger.debug(f"Split skills shape: {split_skills.shape}")

        # Join with question metadata to preserve all questions
        split_question_data = question_meta.join(
            split_skills, on="question", how="left"
        )
        logger.debug(
            f"Split question_data shape (after join): {split_question_data.shape}"
        )

        # Build ID mappings for skill/assignment/template ===
        # Note: question IDs are already mapped, so we don't remap them
        self._build_id_mapping(split_question_data, ["skill", "assignment", "template"])
        logger.debug(
            f"ID mappings: skills={self._get_mapped_count('skill')}, "
            f"assignments={self._get_mapped_count('assignment')}, "
            f"templates={self._get_mapped_count('template')}"
        )

        # Apply ID mappings to question_data
        question_data = self._apply_id_mapping(
            split_question_data, columns=["skill", "assignment", "template"]
        )

        # Build final sequence_data
        sequence_data = mapped_data.select(
            ["user", "question", "label", "attempt_count", "hint_count", "order_id"]
        )

        # Build ID mapping for user in sequence_data
        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])

        # Store processed data in instance variables
        self.question_data = question_data
        self.sequence_data = sequence_data

    def _clean_raw_data(self) -> pl.DataFrame:
        """Clean raw sequence data."""
        if self.raw_data is None:
            self.load_src_data()

        schema_names = self.raw_data.collect_schema().names()
        data = self.raw_data.drop("") if "" in schema_names else self.raw_data

        # Drop unnecessary columns
        data = data.drop(
            [
                "assistment_id",
                "ms_first_response",
                "tutor_mode",
                "answer_type",
                "sequence_id",
                "student_class_id",
                "position",
                "type",
                "base_sequence_id",
                "skill_name",
                "teacher_id",
                "school_id",
                "hint_total",
                "overlap_time",
                "answer_id",
                "answer_text",
                "first_action",
                "bottom_hint",
                "opportunity",
                "opportunity_original",
            ]
        )

        # Rename columns
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

        # Filter invalid data
        data = data.filter(
            pl.col("user").is_not_null()
            & pl.col("skill").is_not_null()
            & pl.col("label").is_not_null()
        )

        data = data.unique().collect()
        data = data.sort(["user", "order_id"])

        # Restrict sequence length
        data = restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )

        return data


__all__ = ["Assistments2009Data"]
