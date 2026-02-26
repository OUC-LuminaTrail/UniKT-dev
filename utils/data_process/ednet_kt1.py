import os

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import (
    DataSource,
    restrains_sequence_length,
)

logger = get_logger(__name__)


def _canonicalize_tags(tags: str, sep: str = ";") -> str | None:
    """Canonicalize tag string: deduplicate and sort.

    Args:
        tags: Tag string
        sep: Separator

    Returns:
        Canonicalized tag string, or None if input is empty
    """
    if tags is None:
        return None
    if not isinstance(tags, str):
        tags = str(tags)
    parts = [p.strip() for p in tags.split(sep) if p.strip()]
    if not parts:
        return None
    parts = sorted(dict.fromkeys(parts))
    return sep.join(parts)


@register_data_source("ednet_kt1")
class EdNetKT1Data(DataSource):
    def __init__(self, args):
        super().__init__(
            dataset="ednet_kt1",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/EdNetKT1.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_folder = os.path.join(self.data_folder, "raw")

    @override
    def load_src_data(self):
        """Load raw data using Polars lazy evaluation with streaming."""
        self._validate_data_paths()

        question_answer_map = self._load_question_answers()
        response_pattern = os.path.join(
            self.raw_data_folder, "EdNet-KT1", "KT1", "*.csv"
        )

        logger.info(f"Scanning CSV files from: {response_pattern}")

        lazy_df = (
            pl.scan_csv(
                response_pattern,
                schema_overrides={
                    "question_id": pl.Utf8,
                    "user_answer": pl.Utf8,
                    "timestamp": pl.Int64,
                },
                include_file_paths="__file_path__",
                glob=True,
            )
            .with_columns(
                pl.col("__file_path__")
                .str.extract(r"([^/]+)\.csv$", 1)
                .str.strip_prefix("u")
                .cast(pl.Int32)
                .alias("user_id")
            )
            .select(["user_id", "question_id", "user_answer", "timestamp"])
        )

        lazy_df = self._process_lazy_pipeline(lazy_df, question_answer_map)
        self.sequence_data_raw = lazy_df.collect(engine="streaming")
        logger.info(f"Loaded {len(self.sequence_data_raw)} raw interactions.")

    def _validate_data_paths(self):
        """Validate that required data paths exist."""
        if not os.path.exists(self.raw_data_folder):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_folder}")
        logger.info(f"Loading raw data from: {self.raw_data_folder}")

        question_path = os.path.join(
            self.raw_data_folder, "EdNet-Contents", "questions.csv"
        )
        if not os.path.exists(question_path):
            raise FileNotFoundError(f"Cannot find: {question_path}")

        response_path = os.path.join(self.raw_data_folder, "EdNet-KT1", "KT1")
        if not os.path.exists(response_path):
            raise FileNotFoundError(f"Cannot find: {response_path}")

    def _load_question_answers(self) -> dict[str, str]:
        """Load question-answer mapping.

        EdNet uses letter answers (a, b, c, d), returns string mapping.

        Returns:
            Dictionary mapping question_id to correct_answer
        """
        question_path = os.path.join(
            self.raw_data_folder, "EdNet-Contents", "questions.csv"
        )
        lazy_questions = pl.scan_csv(question_path, try_parse_dates=False)

        question_answer_map = (
            lazy_questions.select(["question_id", "correct_answer"])
            .filter(
                pl.col("question_id").is_not_null()
                & pl.col("correct_answer").is_not_null()
            )
            .with_columns(
                [
                    pl.col("question_id").cast(pl.Utf8),
                    pl.col("correct_answer").cast(pl.Utf8),
                ]
            )
            .sort("question_id")
            .unique(subset=["question_id"], keep="last")
            .collect()
            .to_dict(as_series=False)
        )

        question_answer_map = dict(
            zip(
                question_answer_map["question_id"],
                question_answer_map["correct_answer"],
            )
        )

        if not question_answer_map:
            raise ValueError("No valid question-answer mapping found for EdNet-KT1.")

        self.question_data_raw = lazy_questions.collect()
        return question_answer_map

    def _process_lazy_pipeline(
        self, lazy_df: pl.LazyFrame, question_answer_map: dict[str, str]
    ) -> pl.LazyFrame:
        """Process lazy DataFrame pipeline with vectorized operations.

        Args:
            lazy_df: Input LazyFrame
            question_answer_map: Question answer mapping

        Returns:
            Processed LazyFrame
        """
        lookup_lazy = pl.DataFrame(
            {
                "question_id": list(question_answer_map.keys()),
                "correct_answer": list(question_answer_map.values()),
            }
        ).lazy()

        return (
            lazy_df.join(lookup_lazy, on="question_id", how="inner")
            .with_columns(
                [
                    (pl.col("user_answer") == pl.col("correct_answer"))
                    .cast(pl.Int8)
                    .alias("label"),
                    pl.col("question_id").cast(pl.Utf8),
                ]
            )
            .select(["user_id", "question_id", "label", "timestamp"])
        )

    @override
    def clear_data(self):
        """Process and clean data."""
        logger.info("Processing Data...")

        if not hasattr(self, "sequence_data_raw") or not hasattr(
            self, "question_data_raw"
        ):
            self.load_src_data()

        question_data = self._process_question_data()
        sequence_data = self._process_sequence_data()
        sequence_data, question_data = self._remap_ids(sequence_data, question_data)

        self.sequence_data = sequence_data
        self.question_data = question_data
        self._save_metadata()

        logger.info(f"Processed {len(sequence_data)} interactions.")

    def _process_question_data(self) -> pl.DataFrame:
        """Process question data with vectorized operations.

        EdNet uses string skill names (e.g., "algebra;geometry"), requires
        splitting multi-skill questions.

        Returns:
            Processed question DataFrame with integer skill IDs
        """
        question_data = (
            self.question_data_raw.rename(
                {"tags": "skill", "question_id": "question", "bundle_id": "assignment"}
            )
            .filter(
                pl.col("correct_answer").is_not_null() & pl.col("skill").is_not_null()
            )
            .with_columns(
                [
                    pl.col("skill")
                    .map_elements(_canonicalize_tags, return_dtype=pl.Utf8)
                    .alias("skill_canonicalized"),
                    pl.col("question").cast(pl.Utf8),
                ]
            )
        )

        # Explode multi-skill questions (no effect if no semicolons)
        question_data = (
            question_data.with_columns(
                [pl.col("skill_canonicalized").str.split(";").alias("skill_list")]
            )
            .explode("skill_list")
            .with_columns([pl.col("skill_list").str.strip_chars().alias("skill")])
            .drop("skill_canonicalized", "skill_list")
        )

        # Remove duplicates
        question_data = question_data.unique(subset=["question", "skill"], keep="first")

        # Map skills to continuous integer IDs
        appeared_skills = (
            question_data.select(pl.col("skill").drop_nulls().unique())
            .to_series()
            .sort()
            .to_list()
        )
        skill_id_map = {skill: idx for idx, skill in enumerate(appeared_skills)}
        question_data = question_data.with_columns(
            [pl.col("skill").replace(skill_id_map).cast(pl.Int32)]
        )

        # For EdNet, skill is also the template
        question_data = question_data.with_columns([pl.col("skill").alias("template")])

        return question_data

    def _process_sequence_data(self) -> pl.DataFrame:
        """Process sequence data.

        Returns:
            Processed sequence DataFrame
        """
        required_cols = ["user_id", "question_id", "label", "timestamp"]
        current_cols = self.sequence_data_raw.columns
        missing_cols = [col for col in required_cols if col not in current_cols]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in sequence data: {missing_cols}"
            )

        sequence_data = (
            self.sequence_data_raw.select(required_cols)
            .with_columns([pl.col("question_id").cast(pl.Utf8)])
            .rename({"question_id": "question", "user_id": "user"})
            .filter(
                pl.col("user").is_not_null()
                & pl.col("question").is_not_null()
                & pl.col("label").is_not_null()
            )
            .sort(["user", "timestamp"])
        )

        return restrains_sequence_length(
            sequence_data, self.args.min_seq_len, self.args.max_seq_len
        )

    def _remap_ids(
        self, sequence_data: pl.DataFrame, question_data: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Remap IDs to continuous integers.

        Args:
            sequence_data: Sequence data
            question_data: Question data

        Returns:
            Tuple of (sequence_data, question_data) with remapped IDs
        """
        # Remap question IDs
        appeared_questions = (
            sequence_data.select(pl.col("question").drop_nulls().unique())
            .to_series()
            .sort()
            .to_list()
        )
        question_id_map = {q: idx for idx, q in enumerate(appeared_questions)}
        sequence_data = sequence_data.with_columns(
            [pl.col("question").replace(question_id_map).cast(pl.Int32)]
        )
        question_data = question_data.filter(
            pl.col("question").is_in(appeared_questions)
        ).with_columns([pl.col("question").replace(question_id_map).cast(pl.Int32)])

        # Remap user IDs
        appeared_users = (
            sequence_data.select(pl.col("user").drop_nulls().unique())
            .to_series()
            .sort()
            .to_list()
        )
        user_id_map = {user_id: idx for idx, user_id in enumerate(appeared_users)}
        sequence_data = sequence_data.with_columns(
            [pl.col("user").replace(user_id_map).cast(pl.Int32)]
        )

        return sequence_data, question_data

    def _save_metadata(self):
        """Save dataset metadata."""
        self.add_metadatas(
            {
                "num_users": self.sequence_data["user"].n_unique(),
                "num_questions": self.question_data["question"].n_unique(),
                "num_skills": self.question_data["skill"].n_unique(),
                "num_assignments": self.question_data["assignment"].n_unique(),
                "num_templates": self.question_data["template"].n_unique(),
                "max_seq_len": self.args.max_seq_len,
                "min_seq_len": self.args.min_seq_len,
                "sequence_columns": list(self.sequence_data.columns),
                "question_columns": list(self.question_data.columns),
            }
        )
