"""Shared fixtures for data_process tests: minimal frames and a bypass DataSource.

Column names and dtypes mirror what the ETL pipeline produces
(``utils/data_process/assist09.py``): sequence frames carry
``user/question/label/timestamp`` (Int32/Int32/Int8/Int64) and relation tables
carry exactly two columns. ``make_data_source`` builds a DataSource WITHOUT
running any ETL: the ``__new__`` bypass (see test_skill_model_data.py) plus
explicitly assigned attributes, so tests exercise only the logic under test.
"""

import os
from types import SimpleNamespace

import polars as pl
import pytest

from utils.data_process.data_source import DataSource


class _MinimalDataSource(DataSource):
    """Concrete DataSource stub: the three abstract ETL steps are inert."""

    def load_src_data(self): ...

    def transform_data(self): ...

    def clean_raw_data(self): ...


@pytest.fixture
def make_sequence_frame():
    """Factory: minimal sequence frame with the exact schema ETL emits."""

    def _make(users, questions, labels=None, timestamps=None):
        n = len(users)
        return pl.DataFrame(
            {
                "user": pl.Series(users, dtype=pl.Int32),
                "question": pl.Series(questions, dtype=pl.Int32),
                "label": pl.Series(
                    labels if labels is not None else [0] * n, dtype=pl.Int8
                ),
                "timestamp": pl.Series(
                    timestamps if timestamps is not None else list(range(n)),
                    dtype=pl.Int64,
                ),
            }
        )

    return _make


@pytest.fixture
def make_question_skill_frame():
    """Factory: 2-column question->skill relation table."""

    def _make(pairs):
        return pl.DataFrame(
            {
                "question": pl.Series([q for q, _ in pairs], dtype=pl.Int32),
                "skill": pl.Series([s for _, s in pairs], dtype=pl.Int32),
            }
        )

    return _make


@pytest.fixture
def make_data_source(tmp_path):
    """Factory: DataSource via __new__ bypass with every attribute set
    explicitly -- no downloads, no metadata plumbing, no directory creation."""

    def _make(
        sequence_data=None,
        relation_data=None,
        max_seq_len=100,
        min_seq_len=1,
        dataset="stub",
        metadata=None,
        data_folder=None,
        seed=42,
    ):
        ds = _MinimalDataSource.__new__(_MinimalDataSource)
        ds.dataset = dataset
        ds.data_base_path = str(tmp_path)
        ds.data_folder = str(tmp_path) if data_folder is None else str(data_folder)
        ds.metadata_path = os.path.join(ds.data_folder, "metadata.json")
        ds.args = SimpleNamespace(
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            windowlate_users_per_batch=1,
        )
        ds.sequence_data = sequence_data
        ds.relation_data = {} if relation_data is None else relation_data
        ds.metadata = {} if metadata is None else metadata
        ds.seed = seed
        ds.raw_data = None
        ds.cleaned_raw_data = None
        ds.data_url = None
        ds._id_mappings = {}
        ds._data_cache = {}
        ds._data_config = {
            "sequence": {"lazy": False},
            "split_question_sequence": {"lazy": False},
            "split_skill_sequence": {"lazy": False},
            "windowlate": {"lazy": True},
        }
        return ds

    return _make
