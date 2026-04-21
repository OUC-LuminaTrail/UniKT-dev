import os

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import (
    DataSource,
    exclude_short_sequences,
)

logger = get_logger(__name__)


@register_data_source("assistments17")
class Assistments2017Data(DataSource):
    """ASSISTments 2017 dataset handler."""

    def __init__(self, args):
        super().__init__(
            dataset="assistments17",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments17.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(
            self.data_folder, "raw", "anonymized_full_release_competition_dataset.csv"
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
        logger.info("Processing ASSISTments 2017 data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        # Build question ID mapping
        question_map_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
        )

        # Save question mapping
        question_map = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )
        self._id_mappings["question"] = question_map
        logger.debug(f"Built question ID mapping: {len(question_map)} unique questions")

        # Apply question mapping
        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Build question_data
        question_meta = mapped_data.select(["question", "assignment"]).unique(
            subset=["question"]
        )
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

        # Build ID mappings for skill/assignment
        self._build_id_mapping(question_data, ["skill", "assignment"])
        logger.debug(
            f"ID mappings: skills={self._get_mapped_count('skill')}, "
            f"assignments={self._get_mapped_count('assignment')}"
        )

        # Apply ID mappings to question_data
        question_data = self._apply_id_mapping(
            question_data, columns=["skill", "assignment"]
        )

        # Build sequence_data
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

        # Drop unnecessary columns
        data = self.raw_data.drop(
            [
                "MiddleSchoolId",
                "InferredGender",
                "SY ASSISTments Usage",
                "AveKnow",
                "AveCarelessness",
                "AveCorrect",
                "NumActions",
                "AveResBored",
                "AveResEngcon",
                "AveResConf",
                "AveResFrust",
                "AveResOfftask",
                "AveResGaming",
                "action_num",
                "problemType",
                "assistmentId",
                "endTime",
                "timeTaken",
                "hint",
                "hintTotal",
                "scaffold",
                "bottomHint",
                "frIsHelpRequest",
                "frPast5HelpRequest",
                "frPast8HelpRequest",
                "stlHintUsed",
                "past8BottomOut",
                "totalFrPercentPastWrong",
                "totalFrPastWrongCount",
                "frPast5WrongCount",
                "frPast8WrongCount",
                "totalFrTimeOnSkill",
                "timeSinceSkill",
                "frWorkingInSchool",
                "totalFrAttempted",
                "totalFrSkillOpportunities",
                "responseIsFillIn",
                "responseIsChosen",
                "endsWithScaffolding",
                "endsWithAutoScaffolding",
                "frTimeTakenOnScaffolding",
                "frTotalSkillOpportunitiesScaffolding",
                "totalFrSkillOpportunitiesByScaffolding",
                "frIsHelpRequestScaffolding",
                "timeGreater5Secprev2wrong",
                "sumRight",
                "helpAccessUnder2Sec",
                "timeGreater10SecAndNextActionRight",
                "consecutiveErrorsInRow",
                "sumTime3SDWhen3RowRight",
                "sumTimePerSkill",
                "totalTimeByPercentCorrectForskill",
                "Prev5count",
                "timeOver80",
                "manywrong",
                "confidence(BORED)",
                "confidence(CONCENTRATING)",
                "confidence(CONFUSED)",
                "confidence(FRUSTRATED)",
                "confidence(OFF TASK)",
                "confidence(GAMING)",
                "RES_BORED",
                "RES_CONCENTRATING",
                "RES_CONFUSED",
                "RES_FRUSTRATED",
                "RES_OFFTASK",
                "RES_GAMING",
                "Ln-1",
                "Ln",
                "MCAS",
                "Enrolled",
                "Selective",
                "isSTEM",
            ]
        )

        # Rename columns
        data = data.rename(
            {
                "studentId": "user",
                "problemId": "question",
                "correct": "label",
                "skill": "skill",
                "assignmentId": "assignment",
                "hintCount": "hint_count",
                "attemptCount": "attempt_count",
                "startTime": "timestamp",
            }
        )

        data = data.unique()
        data = data.collect()

        # Convert Unix seconds to milliseconds for consistency
        data = data.with_columns(
            (pl.col("timestamp") * 1000).cast(pl.Int64).alias("timestamp")
        )

        # Convert to global relative time (dataset-wise earliest timestamp as zero)
        data = data.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        data = data.sort(["user", "timestamp"])
        data = data.with_columns([pl.col("user").cast(pl.Int32)])
        data = data.filter(pl.col("skill").is_not_null())
        data = data.filter(pl.col("label").is_in([0, 1]))

        # Exclude sequences that are too short
        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data


__all__ = ["Assistments2017Data"]
