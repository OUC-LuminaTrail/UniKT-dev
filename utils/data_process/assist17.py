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
                # "studentId",
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
                # "skill",
                # "problemId",
                "problemType",
                # "assignmentId",
                "assistmentId",
                # "startTime",
                "endTime",
                "timeTaken",
                # "correct",
                # "original",
                "hint",
                # "hintCount",
                "hintTotal",
                "scaffold",
                "bottomHint",
                # "attemptCount",
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

        data = data.rename(
            {
                "studentId": "user",
                "problemId": "question",
                "correct": "label",
                "skill": "skill",
                "assignmentId": "assignment",
                "hintCount": "hint_count",
                "attemptCount": "attempt_count",
            }
        )

        data = data.unique()
        data = data.collect()

        data = data.sort(["user", "startTime"])
        data = data.with_columns([pl.col("user").cast(pl.Int32)])
        data = data.filter(pl.col("skill").is_not_null())
        data = data.filter(pl.col("label").is_in([0, 1]))

        data = restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )

        # Keep data before ID mapping for question_data (skills are still strings)
        data_before_mapping = data.clone()

        data = map_to_continuous_ids(
            data, columns=["user", "question", "skill", "assignment"]
        )

        # Build question_data with skill splitting
        # ASSISTments 17 uses descriptive skill names with hyphens (e.g., "subtracting-decimals")
        question_data = build_question_data_from_cleared(
            data_before_mapping, skill_column="skill", question_column="question", separator="-"
        )

        self.cleared_data = data.clone()
        self.sequence_data = data.clone()
        self.question_data = question_data

        self.add_metadatas(
            {
                "num_users": self.cleared_data["user"].n_unique(),
                "num_questions": self.question_data["question"].n_unique(),
                "num_skills": self.question_data["skill"].n_unique(),
                "num_assignments": self.question_data["assignment"].n_unique(),
                "max_seq_len": self.args.max_seq_len,
                "min_seq_len": self.args.min_seq_len,
                "columns": data.columns,
            }
        )


__all__ = ["Assistments2017Data"]
