"""ASSISTments 2017 datasets, built from one raw release at two granularities.

- ``assistments17``: original action-level (one row per student action).
- ``assistments17_per_que``: per-encounter rebuild. Scaffolding substeps
  (``scaffold=1``) reuse the main problem's ``problemId``, so they are shifted
  to independent question ids before aggregation -- otherwise their
  correctness leaks into the main problem's label.
"""

import os

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import (
    DataSource,
    exclude_short_sequences,
)

logger = get_logger(__name__)

_RAW_CSV = "anonymized_full_release_competition_dataset.csv"

# Shifts scaffold=1 actions to independent question ids; max(problemId)=22761.
_SCAFFOLD_OFFSET = 10_000_000


class Assistments2017Base(DataSource):
    """Shared raw loading and transform; subclasses implement clean_raw_data."""

    def __init__(self, args, dataset: str):
        """Initialize the shared AS17 handler."""
        super().__init__(
            dataset=dataset,
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments17.zip",
            seed=args.seed,
        )
        # Both granularities live in the AS17 folder; non-default variants
        # write a separate metadata file so AS17's metadata.json is untouched.
        self.data_folder = os.path.join(self.data_base_path, "assistments17")
        if dataset != "assistments17":
            self.metadata_path = os.path.join(
                self.data_folder, f"{dataset}_metadata.json"
            )
        self.args = args
        self.raw_data_path = os.path.join(self.data_folder, "raw", _RAW_CSV)

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
        logger.info(f"Processing {self.dataset} data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        question_map_df = (
            self.cleaned_raw_data.select("question")
            .unique()
            .sort("question")
            .with_row_index("question_id")
        )

        question_map = dict(
            zip(
                question_map_df["question"].to_list(),
                question_map_df["question_id"].to_list(),
            )
        )
        self._id_mappings["question"] = question_map
        logger.debug(f"Built question ID mapping: {len(question_map)} unique questions")

        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        question_skill = mapped_data.select(["question", "skill"]).unique(
            subset=["question", "skill"], keep="first"
        )
        question_assignment = mapped_data.select(["question", "assignment"]).unique(
            subset=["question", "assignment"]
        )

        self._build_id_mapping(question_skill, ["skill"])
        self._build_id_mapping(question_assignment, ["assignment"])
        logger.debug(
            f"ID mappings: skills={self._get_mapped_count('skill')}, "
            f"assignments={self._get_mapped_count('assignment')}"
        )

        question_skill = self._apply_id_mapping(question_skill, columns=["skill"])
        question_assignment = self._apply_id_mapping(
            question_assignment, columns=["assignment"]
        )

        sequence_data = mapped_data.select(
            [
                "user",
                "question",
                "label",
                "attempt_count",
                "hint_count",
                "ms_first_response",
                "timestamp",
            ]
        )

        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])
        logger.debug(f"Built user ID mapping: {self._get_mapped_count('user')} users")

        self.relation_data = {
            "question_skill": question_skill,
            "question_assignment": question_assignment,
        }
        self.sequence_data = sequence_data


@register_data_source("assistments17")
class Assistments2017Data(Assistments2017Base):
    """Original action-level granularity."""

    def __init__(self, args):
        """Initialize the action-level AS17 handler."""
        super().__init__(args=args, dataset="assistments17")

    def clean_raw_data(self):
        """Clean raw action-level data."""
        if self.raw_data is None:
            self.load_src_data()

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
        ).rename(
            {
                "studentId": "user",
                "problemId": "question",
                "correct": "label",
                "skill": "skill",
                "assignmentId": "assignment",
                "hintCount": "hint_count",
                "attemptCount": "attempt_count",
                "startTime": "timestamp",
                "timeTaken": "ms_first_response",
            }
        )

        data = data.unique().collect()

        data = data.with_columns(
            (pl.col("timestamp") * 1000).cast(pl.Int64).alias("timestamp"),
            (pl.col("ms_first_response") * 1000)
            .cast(pl.Int64)
            .alias("ms_first_response"),
        )
        data = data.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        data = data.sort(["user", "timestamp"])
        data = data.with_columns(pl.col("user").cast(pl.Int32))
        data = data.filter(pl.col("skill").is_not_null())
        data = data.filter(pl.col("label").is_in([0, 1]))

        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data


@register_data_source("assistments17_per_que")
class Assistments2017PerQueData(Assistments2017Base):
    """Per-encounter rebuild with scaffolding split into independent questions."""

    def __init__(self, args):
        """Initialize the per-encounter AS17 handler."""
        super().__init__(args=args, dataset="assistments17_per_que")

    def clean_raw_data(self):
        """Aggregate action-level rows into per-encounter rows."""
        if self.raw_data is None:
            self.load_src_data()

        data = self.raw_data.select(
            [
                "studentId",
                "problemId",
                "assignmentId",
                "action_num",
                "skill",
                "correct",
                "scaffold",
                "hint",
                "startTime",
                "timeTaken",
            ]
        )

        max_pid = data.select(pl.col("problemId").max()).collect().item()
        if max_pid is not None and max_pid >= _SCAFFOLD_OFFSET:
            raise ValueError(
                f"problemId max ({max_pid}) >= SCAFFOLD_OFFSET "
                f"({_SCAFFOLD_OFFSET}); increase the offset"
            )

        logger.info(
            "Rebuilding AS17 to per-encounter granularity (scaffold offset=%d)...",
            _SCAFFOLD_OFFSET,
        )

        data = data.with_columns(
            pl.when(pl.col("scaffold") == 1)
            .then(pl.col("problemId") + _SCAFFOLD_OFFSET)
            .otherwise(pl.col("problemId"))
            .alias("question")
        )

        data = (
            data.sort(["studentId", "question", "assignmentId", "action_num"])
            .group_by(["studentId", "question", "assignmentId"])
            .agg(
                pl.col("correct").filter(pl.col("hint") == 0).first().alias("label"),
                (pl.col("hint") == 0).sum().cast(pl.Int64).alias("attempt_count"),
                (pl.col("hint") == 1).sum().cast(pl.Int64).alias("hint_count"),
                pl.col("timeTaken")
                .filter(pl.col("hint") == 0)
                .first()
                .alias("ms_first_response"),
                pl.col("startTime").min().alias("timestamp"),
                pl.col("skill").first().alias("skill"),
            )
            .filter(pl.col("label").is_in([0, 1]))
        )

        data = data.collect()
        n_scaff_q = (
            data.filter(pl.col("question") >= _SCAFFOLD_OFFSET)
            .select(pl.col("question").n_unique())
            .item()
        )
        logger.info(
            "Per-encounter rebuild: %d encounters, %d scaffolding questions",
            data.height,
            n_scaff_q or 0,
        )

        data = data.rename({"studentId": "user", "assignmentId": "assignment"})

        data = data.with_columns(
            (pl.col("timestamp") * 1000).cast(pl.Int64).alias("timestamp"),
            (pl.col("ms_first_response") * 1000)
            .cast(pl.Int64)
            .alias("ms_first_response"),
        )
        data = data.with_columns(
            (pl.col("timestamp") - pl.col("timestamp").min())
            .cast(pl.Int64)
            .alias("timestamp")
        )

        data = data.sort(["user", "timestamp"])
        data = data.with_columns(pl.col("user").cast(pl.Int32))
        data = data.filter(pl.col("skill").is_not_null())

        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data


__all__ = ["Assistments2017Base", "Assistments2017Data", "Assistments2017PerQueData"]
