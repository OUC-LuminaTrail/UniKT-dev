"""Shared fixtures for model_data tests: stub DataSource and parquet writer."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.model_data.skill_model_data import SkillModelData

_CANONICAL_WINDOWLATE_COLUMNS = (
    ("sample_id", pa.int32()),
    ("position", pa.int32()),
    ("skill", pa.int32()),
    ("question", pa.int32()),
    ("response", pa.int8()),
    ("mask", pa.int8()),
    ("user_id", pa.int32()),
    ("group_id", pa.int64()),
    ("true_label", pa.int8()),
)


class StubDataSource:
    """Minimal DataSource stand-in: metadata dict + in-memory data frames.

    Bypasses the md5/integrity plumbing so tests exercise only the id-space
    propagation logic under test.
    """

    def __init__(
        self,
        split_skill_sequence_data: Any = None,
        metadata: dict | None = None,
        windowlate_data: Any = None,
    ):
        self.dataset = "stub"
        self.data_folder = "/nonexistent"
        self._split_skill_sequence_data = split_skill_sequence_data
        self._metadata = metadata or {}
        self._windowlate_data = windowlate_data

    def get_split_skill_sequence_data(self):
        return self._split_skill_sequence_data

    def get_windowlate_data(self):
        return self._windowlate_data

    def get_metadata(self, key: str | None = None):
        if key is not None and key not in self._metadata:
            raise KeyError(f"Metadata key '{key}' not found")
        return self._metadata if key is None else self._metadata[key]


@pytest.fixture
def write_windowlate_parquet(tmp_path):
    """Return a writer bound to tmp_path; writes the canonical column schema."""

    def _write(filename: str, samples: dict[str, list]) -> str:
        schema = pa.schema(list(_CANONICAL_WINDOWLATE_COLUMNS))
        table = pa.table(
            {
                name: pa.array(samples[name], type=schema.field(name).type)
                for name in schema.names
            },
            schema=schema,
        )
        path = str(tmp_path / filename)
        pq.write_table(table, path)
        return path

    return _write


@pytest.fixture
def make_skill_model_data():
    """Factory: bind a concrete SkillModelData to a StubDataSource."""

    class _ConcreteSkillModelData(SkillModelData):
        def prepare_data(self, rc):
            raise NotImplementedError("stub; call build/load methods directly")

    def _make(
        split_frame: Any = None,
        metadata: dict | None = None,
        windowlate_data: Any = None,
    ) -> SkillModelData:
        return _ConcreteSkillModelData(
            StubDataSource(
                split_skill_sequence_data=split_frame,
                metadata=metadata,
                windowlate_data=windowlate_data,
            )
        )

    return _make
