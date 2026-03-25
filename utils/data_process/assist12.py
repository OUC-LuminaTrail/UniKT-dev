import os

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import (
    DataSource,
    exclude_short_sequences,
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
    def transform_data(self):
        """Clean data and build question_data and sequence_data."""
        logger.info("Processing ASSISTments 2012 data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        # Build question ID mapping
        question_map_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
        )

        # Save question ID mapping
        question_map = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )
        self._id_mappings["question"] = question_map
        logger.info(f"Built question ID mapping: {len(question_map)} unique questions")

        # Apply question mapping
        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Build question_data
        question_meta = mapped_data.select(
            ["question", "assignment", "template"]
        ).unique(subset=["question"])
        logger.debug(f"Question metadata shape: {question_meta.shape}")

        # Build base data with question-skill pairs
        base_question_data = mapped_data.select(["question", "skill"]).unique(
            subset=["question", "skill"], keep="first"
        )
        logger.debug(f"Base question_data shape: {base_question_data.shape}")

        # Join with question metadata to preserve all questions
        question_data = question_meta.join(
            base_question_data, on="question", how="left"
        )
        logger.debug(f"question_data shape: {question_data.shape}")

        # Build ID mappings for skill/assignment/template
        self._build_id_mapping(question_data, ["skill", "assignment", "template"])
        logger.info(
            f"ID mappings: skills={self._get_mapped_count('skill')}, "
            f"assignments={self._get_mapped_count('assignment')}, "
            f"templates={self._get_mapped_count('template')}"
        )

        # Apply ID mappings to question_data
        question_data = self._apply_id_mapping(
            question_data, columns=["skill", "assignment", "template"]
        )

        # Build final sequence_data
        sequence_data = mapped_data.select(
            ["user", "question", "label", "attempt_count", "hint_count", "timestamp"]
        )

        # Build ID mapping for user in sequence_data
        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])
        logger.debug(f"Built user ID mapping: {self._get_mapped_count('user')} users")

        # Store processed data in instance variables
        self.question_data = question_data
        self.sequence_data = sequence_data

    def clean_raw_data(self):
        """Clean raw sequence data."""
        if self.raw_data is None:
            self.load_src_data()

        data = self.raw_data.with_row_index("tmp_index")

        # Drop unnecessary columns
        data = data.drop(
            [
                "problem_log_id",
                "skill",
                "assistment_id",
                "end_time",
                "problem_type",
                "original",
                "bottom_hint",
                "actions",
                "tutor_mode",
                "sequence_id",
                "student_class_id",
                "position",
                "type",
                "base_sequence_id",
                "teacher_id",
                "school_id",
                "overlap_time",
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

        # Rename columns
        data = data.rename(
            {
                "user_id": "user",
                "problem_id": "question",
                "correct": "label",
                "assignment_id": "assignment",
                "skill_id": "skill",
                "template_id": "template",
                "start_time": "timestamp",
            }
        )

        # pyKT 对齐：保留原始行顺序索引用于同时间戳稳定排序，
        # 并在清洗阶段要求关键字段非空（含 ms_first_response）。
        data = data.filter(
            pl.col("user").is_not_null()
            & pl.col("question").is_not_null()
            & pl.col("skill").is_not_null()
            & pl.col("timestamp").is_not_null()
            & pl.col("label").is_not_null()
            & pl.col("ms_first_response").is_not_null()
        ).with_columns(pl.col("label").cast(pl.Int32))

        data = data.filter(pl.col("label").is_in([0, 1]))
        data = data.collect()

        # pyKT change2timestamp 等价：按是否含小数秒分别解析时间字符串
        data = data.with_columns(
            [
                pl.when(pl.col("timestamp").str.contains(r"\."))
                .then(
                    pl.col("timestamp").str.strptime(
                        pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False
                    )
                )
                .otherwise(
                    pl.col("timestamp").str.strptime(
                        pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
                    )
                )
                .dt.epoch("ms")
                .alias("timestamp")
            ]
        )

        # 与 pyKT 一致：无法解析的时间戳不参与序列构建
        data = data.filter(pl.col("timestamp").is_not_null())
        data = data.sort(["user", "timestamp", "tmp_index"])
        data = data.with_columns([pl.col("user").cast(pl.Int32)])
        data = data.drop(["tmp_index", "ms_first_response"])

        # Exclude short sequences
        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data


__all__ = ["Assistments2012Data"]
