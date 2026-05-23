import os

import polars as pl
from typing_extensions import override

from utils.core import get_logger, register_data_source

from .data_source import DataSource, exclude_short_sequences

logger = get_logger(__name__)


class KDDCup2010Base(DataSource):
    """Base class for KDD Cup 2010 datasets (Algebra 2005/2006, Bridge 2006).

    These datasets share the same tab-separated format with columns:
    Row, Anon Student Id, Problem Hierarchy, Problem Name, Problem View,
    Step Name, ..., First Transaction Time, ..., Correct First Attempt,
    ..., KC(Default) or KC(SubSkills), Opportunity(...)
    """

    skill_column: str  # Subclass must set: e.g. "KC(Default)" or "KC(SubSkills)"

    def __init__(self, args, dataset: str, raw_filename: str):
        super().__init__(
            dataset=dataset,
            data_base_path=args.data_base_path,
            data_url=f"http://cdn.lionhao.top/KTDataset/{dataset}.zip",
            seed=args.seed,
        )
        self.args = args
        self.raw_data_path = os.path.join(self.data_folder, "raw", raw_filename)

    @override
    def load_src_data(self):
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        logger.info(f"Loading raw data from: {self.raw_data_path}")
        self.raw_data = pl.read_csv(
            self.raw_data_path,
            separator="\t",
            ignore_errors=True,
            try_parse_dates=False,
            null_values=[""],
        ).lazy()

    @override
    def clean_raw_data(self):
        if self.raw_data is None:
            self.load_src_data()

        skill_col = self.skill_column

        data = self.raw_data.select(
            [
                pl.col("Anon Student Id").alias("user"),
                (pl.col("Problem Name") + "___" + pl.col("Step Name")).alias(
                    "question"
                ),
                pl.col("First Transaction Time").alias("timestamp"),
                pl.col("Correct First Attempt").cast(pl.Int32).alias("label"),
                pl.col(skill_col).alias("skill"),
            ]
        )

        data = data.filter(
            pl.col("user").is_not_null()
            & pl.col("question").is_not_null()
            & pl.col("label").is_not_null()
            & pl.col("skill").is_not_null()
        )

        data = data.collect()

        # Parse timestamps and convert to relative seconds
        data = data.with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime("ms"), "%Y-%m-%d %H:%M:%S%.f", strict=False)
            .cast(pl.Int64)
            .alias("timestamp")
        )

        min_ts = data.select(pl.col("timestamp").min()).item()
        data = data.with_columns(
            ((pl.col("timestamp") - min_ts) / 1_000_000)
            .cast(pl.Int64)
            .alias("timestamp")
        )

        data = data.sort(["user", "timestamp"])

        data = exclude_short_sequences(data, self.args.min_seq_len)

        self.cleaned_raw_data = data

    @override
    def transform_data(self):
        logger.info(f"Processing {self.dataset} data...")

        if self.cleaned_raw_data is None:
            raise ValueError("clean_raw_data must be called before transform_data")

        # Build question ID mapping
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

        mapped_data = (
            self.cleaned_raw_data.join(question_map_df, on="question", how="left")
            .with_columns(pl.col("question_id").cast(pl.Int32))
            .drop("question")
            .rename({"question_id": "question"})
        )

        # Split multi-skill by "~~" and build question_skill relation
        question_skill = (
            mapped_data.select(["question", "skill"])
            .filter(pl.col("skill").is_not_null())
            .unique(subset=["question", "skill"], keep="first")
            .with_columns(pl.col("skill").str.split("~~").alias("skill_parts"))
            .explode("skill_parts")
            .with_columns(pl.col("skill_parts").cast(pl.String).alias("skill"))
            .select(["question", "skill"])
            .unique(subset=["question", "skill"], keep="first")
        )

        self._build_id_mapping(question_skill, ["skill"])
        question_skill = self._apply_id_mapping(question_skill, columns=["skill"])

        # Build final sequence_data
        sequence_data = mapped_data.select(["user", "question", "label", "timestamp"])

        self._build_id_mapping(sequence_data, ["user"])
        sequence_data = self._apply_id_mapping(sequence_data, columns=["user"])

        self.relation_data = {
            "question_skill": question_skill,
        }
        self.sequence_data = sequence_data


@register_data_source("algebra2005")
class Algebra2005Data(KDDCup2010Base):
    """Algebra 2005-2006 dataset handler.

    Uses KC(Default) as the skill column.
    """

    skill_column = "KC(Default)"

    def __init__(self, args):
        super().__init__(
            args=args,
            dataset="algebra2005",
            raw_filename="algebra_2005_2006_train.txt",
        )


@register_data_source("algebra2006")
class Algebra2006Data(KDDCup2010Base):
    """Algebra 2006-2007 dataset handler.

    Uses KC(Default) as the skill column.
    """

    skill_column = "KC(Default)"

    def __init__(self, args):
        super().__init__(
            args=args,
            dataset="algebra2006",
            raw_filename="algebra_2006_2007_train.txt",
        )


@register_data_source("bridge2006")
class Bridge2006Data(KDDCup2010Base):
    """Bridge to Algebra 2006-2007 dataset handler.

    Uses KC(SubSkills) as the skill column.
    """

    skill_column = "KC(SubSkills)"

    def __init__(self, args):
        super().__init__(
            args=args,
            dataset="bridge2006",
            raw_filename="bridge_to_algebra_2006_2007_train.txt",
        )
