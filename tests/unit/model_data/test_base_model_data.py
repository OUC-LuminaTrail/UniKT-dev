"""Tests for ``utils.model_data.base_model_data``: cache, folds, difficulty, relations.

Reuses the area conftest ``StubDataSource`` / ``make_skill_model_data``
factories. The disk cache normally writes to the repo-root ``.cache/``, so a
fixture redirects ``base_model_data.Path`` to keep every write inside
``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from tests.unit.model_data.conftest import StubDataSource
from utils.model_data.base_model_data import BaseModelData


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Redirect the disk cache directory under tmp_path and return it."""
    from utils.model_data import base_model_data as bmd

    # The wrapper computes Path(__file__).resolve().parent.parent.parent as the
    # project root; anchor that chain inside tmp_path so nothing leaks out.
    monkeypatch.setattr(bmd, "Path", lambda _file: tmp_path / "proj" / "pkg" / "mod")
    return tmp_path / ".cache"


class _CacheProbe(BaseModelData):
    """Counting probe for the disk_cache decorator (parenthesised usage)."""

    def __init__(self, cache=True):
        super().__init__(StubDataSource(), cache=cache)
        self.calls = []

    def prepare_data(self, args): ...

    @BaseModelData.disk_cache()
    def make(self, payload):
        self.calls.append(payload)
        return {"payload": payload, "fresh": True}


class _BareCacheProbe(_CacheProbe):
    """Bare ``@BaseModelData.disk_cache`` (no parentheses) usage."""

    @BaseModelData.disk_cache
    def make(self, payload):
        self.calls.append(payload)
        return {"payload": payload, "fresh": True}


class _SeqDataSource(StubDataSource):
    """StubDataSource extended with the sequence/relation accessors."""

    def __init__(self, sequence_data=None, relations=None, metadata=None):
        super().__init__(metadata=metadata)
        self._sequence = sequence_data
        self._relations = relations or {}

    def get_sequence_data(self):
        return self._sequence

    def get_relation(self, name):
        return self._relations[name]


class _SeqModelData(BaseModelData):
    """Concrete base-class instance bound to a _SeqDataSource."""

    def __init__(self, data_src):
        super().__init__(data_src)

    def prepare_data(self, args): ...


def _fold_frame(pairs):
    """Build a split-skill frame of (sequence_id, fold) rows."""
    return pl.DataFrame(
        {
            "sequence_id": [sid for sid, _ in pairs],
            "fold": [fold for _, fold in pairs],
        }
    )


# ---------------------------------------------------------------------------
# disk_cache decorator
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_cache_hit_calls_inner_once(self, cache_root):
        probe = _CacheProbe()
        first = probe.make(b"payload")
        second = probe.make(b"payload")

        assert len(probe.calls) == 1  # second call served from disk
        assert second == first
        assert any(cache_root.glob("_CacheProbe/*.pkl"))

    def test_bare_decorator_without_parentheses(self, cache_root):
        probe = _BareCacheProbe()
        probe.make("x")
        probe.make("x")
        assert len(probe.calls) == 1

    def test_cache_disabled_bypasses_disk(self, cache_root):
        probe = _CacheProbe(cache=False)
        probe.make("x")
        probe.make("x")
        assert len(probe.calls) == 2  # no caching without _cache=True

    def test_key_normalization_same_content_same_entry(self, cache_root):
        probe = _CacheProbe()
        probe.make(bytes([1, 2, 3]))  # distinct objects, identical content
        probe.make(bytes([1, 2, 3]))
        probe.make({"a", "b"})  # set ordering must not change the key
        probe.make({"b", "a"})
        assert len(probe.calls) == 2

    def test_corrupt_cache_file_rebuilt(self, cache_root):
        probe = _CacheProbe()
        probe.make(b"payload")

        pkl = next(iter(cache_root.glob("_CacheProbe/*.pkl")))
        pkl.write_bytes(b"not a pickle")

        result = probe.make(b"payload")
        assert len(probe.calls) == 2  # read failed -> inner re-ran
        assert result["fresh"] is True


# ---------------------------------------------------------------------------
# _build_user_folds
# ---------------------------------------------------------------------------


class TestBuildUserFolds:
    def test_missing_fold_column_raises(self, make_skill_model_data):
        frame = pl.DataFrame({"sequence_id": [0, 1], "skill": [1, 2]})
        model_data = make_skill_model_data(split_frame=frame)
        with pytest.raises(ValueError, match="K-fold labels not found"):
            model_data._build_user_folds(2)

    def test_inconsistent_user_folds_raises(self, make_skill_model_data):
        frame = _fold_frame([(0, 0), (0, 1), (1, 2)])  # user 0 in two folds
        model_data = make_skill_model_data(split_frame=frame)
        with pytest.raises(ValueError, match="inconsistent fold labels"):
            model_data._build_user_folds(2)

    def test_user_count_mismatch_raises(self, make_skill_model_data):
        frame = _fold_frame([(0, 0), (1, 1)])
        model_data = make_skill_model_data(split_frame=frame)
        with pytest.raises(ValueError, match="User count mismatch"):
            model_data._build_user_folds(5)

    def test_user_index_out_of_range_raises(self, make_skill_model_data):
        frame = _fold_frame([(0, 0), (7, 1)])  # 7 >= num_users=2
        model_data = make_skill_model_data(split_frame=frame)
        with pytest.raises(ValueError, match="User index out of range"):
            model_data._build_user_folds(2)

    def test_fold_mapping_array(self, make_skill_model_data):
        frame = _fold_frame([(0, 2), (1, 2), (2, 0), (3, 0), (4, -1)])
        model_data = make_skill_model_data(split_frame=frame)
        folds = model_data._build_user_folds(5)
        assert folds.tolist() == [2, 2, 0, 0, -1]
        assert folds.dtype == np.int32


# ---------------------------------------------------------------------------
# split_kfold_data
# ---------------------------------------------------------------------------


class TestSplitKfoldData:
    @pytest.fixture
    def split_model_data(self, make_skill_model_data):
        frame = _fold_frame([(0, 0), (0, 0), (1, 1), (2, 0), (3, -1)])
        return make_skill_model_data(split_frame=frame)

    def test_val_test_train_assignment_and_union(self, split_model_data):
        arr = np.arange(8, dtype=float).reshape(4, 2)
        train, val, test = split_model_data.split_kfold_data(arr, fold_idx=1)

        assert val[0].tolist() == [[2.0, 3.0]]  # fold == fold_idx
        assert test[0].tolist() == [[6.0, 7.0]]  # fold == -1
        assert train[0].tolist() == [[0.0, 1.0], [4.0, 5.0]]  # remaining folds

        all_rows = np.concatenate([train[0], val[0], test[0]])
        assert sorted(map(tuple, all_rows)) == sorted(
            map(tuple, arr)
        )  # union covers every user exactly once

    def test_torch_matches_numpy_indexing(self, split_model_data):
        base = np.arange(8, dtype=np.float64).reshape(4, 2)
        tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)

        np_train, np_val, _ = split_model_data.split_kfold_data(base, fold_idx=1)
        t_train, t_val, _ = split_model_data.split_kfold_data(tensor, fold_idx=1)

        assert t_train[0].tolist() == np_train[0].tolist()
        assert t_val[0].tolist() == np_val[0].tolist()
        assert isinstance(t_train[0], torch.Tensor)

    def test_inconsistent_first_dims_raises(self, split_model_data):
        with pytest.raises(ValueError, match="Input array 1 shape"):
            split_model_data.split_kfold_data(
                np.zeros((4, 2)), np.zeros((3, 2)), fold_idx=1
            )

    def test_no_arrays_raises(self, split_model_data):
        with pytest.raises(
            ValueError, match="split_kfold_data requires at least one input"
        ):
            split_model_data.split_kfold_data(fold_idx=1)

    def test_zero_rows_raises_domain_error(self, make_skill_model_data):
        empty = pl.DataFrame(
            {
                "sequence_id": pl.Series([], dtype=pl.Int64),
                "fold": pl.Series([], dtype=pl.Int32),
            }
        )
        model_data = make_skill_model_data(split_frame=empty)
        with pytest.raises(ValueError, match="no users"):
            model_data.split_kfold_data(np.zeros((0, 2)), fold_idx=0)


# ---------------------------------------------------------------------------
# calculate_question_difficulty
# ---------------------------------------------------------------------------


class TestQuestionDifficulty:
    def test_confidence_weighted_difficulty(self):
        # q1: 10 answers, 5 correct -> confidence 1.0 -> 0.5*1.0 + 0.5*0.0 = 0.5
        # q2: 2 answers, 0 correct -> confidence 0.2 -> 1.0*0.2 + 0.5*0.8 = 0.6
        seq = pl.DataFrame(
            {
                "question": [1] * 10 + [2] * 2,
                "label": [1, 0] * 5 + [0, 0],
            }
        )
        model_data = _SeqModelData(_SeqDataSource(sequence_data=seq))

        difficulty = model_data.calculate_question_difficulty()

        assert difficulty == {1: pytest.approx(0.5), 2: pytest.approx(0.6)}

    def test_exclude_fold_filters_rows(self):
        seq = pl.DataFrame(
            {
                "question": [1, 1, 1, 1, 2, 2, 2],
                "label": [1, 0, 1, 0, 0, 1, 0],
                "fold": [0, 0, 1, 1, 0, 0, 0],
            }
        )
        model_data = _SeqModelData(_SeqDataSource(sequence_data=seq))

        difficulty = model_data.calculate_question_difficulty(exclude_fold=1)

        # q1 keeps 2 answers (rate .5, conf .2): .5*.2 + .5*.8 = 0.5
        # q2 keeps 3 answers (rate 1/3, conf .3): (2/3)*.3 + .5*.7 = 0.55
        assert difficulty == {1: pytest.approx(0.5), 2: pytest.approx(0.55)}


# ---------------------------------------------------------------------------
# build_relationship_matrix
# ---------------------------------------------------------------------------


class TestRelationshipMatrix:
    @staticmethod
    def _model_data(rel, metadata=None):
        return _SeqModelData(
            _SeqDataSource(relations={"question_skill": rel}, metadata=metadata)
        )

    @pytest.fixture
    def relation(self):
        # (q0,s1) twice, (q0,s2) once, plus an out-of-range question 5.
        return pl.DataFrame({"question": [0, 0, 0, 5], "skill": [1, 1, 2, 0]})

    def test_binary_vs_count(self, relation):
        metadata = {"num_questions": 2, "num_skills": 3}
        binary = self._model_data(relation, metadata).build_relationship_matrix(
            ("question", "has", "skill")
        )
        count = self._model_data(relation, metadata).build_relationship_matrix(
            ("question", "has", "skill"), value_type="count"
        )

        assert binary.tolist() == [[0, 1, 1], [0, 0, 0]]
        assert count.tolist() == [[0, 2, 1], [0, 0, 0]]  # duplicate pair counted

    def test_out_of_range_pairs_dropped(self, relation):
        metadata = {"num_questions": 2, "num_skills": 3}
        matrix = self._model_data(relation, metadata).build_relationship_matrix(
            ("question", "has", "skill")
        )
        assert matrix.shape == (2, 3)  # question 5 silently dropped, no IndexError

    def test_metadata_num_fallback_computed_from_data(self, relation):
        matrix = self._model_data(relation, metadata={}).build_relationship_matrix(
            ("question", "has", "skill")
        )
        # num_questions/num_skills missing -> n_unique over the relation:
        # questions {0, 5} -> 2, skills {0, 1, 2} -> 3.
        assert matrix.shape == (2, 3)

    def test_unsupported_value_type_raises(self, relation):
        metadata = {"num_questions": 2, "num_skills": 3}
        with pytest.raises(ValueError, match="Unsupported value_type"):
            self._model_data(relation, metadata).build_relationship_matrix(
                ("question", "has", "skill"), value_type="tfidf"
            )
