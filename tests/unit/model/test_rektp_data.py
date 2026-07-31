import math

import numpy as np
import polars as pl

from model.ReKTP.ReKTP_data import (
    ReKTPModelData,
    build_question_skill_table,
    derive_max_gap_bins,
)
from utils.model_data import QuestionModelData


class _RelationOnlyDataSource:
    def __init__(self):
        self.relation = pl.DataFrame(
            {
                "question": [0, 0, 0, 1, 2, 2],
                "skill": [1, 0, 1, 2, 1, 0],
            }
        )
        self.metadata = {"num_questions": 3, "num_skills": 3}

    def get_relation(self, name):
        assert name == "question_skill"
        return self.relation

    def get_metadata(self, name):
        return self.metadata[name]


class _SplitDataDataSource:
    """Fake DataSource exposing a fixed split question sequence frame."""

    def __init__(self, dataset, df, max_seq_len):
        self.dataset = dataset
        self._df = df
        self._max_seq_len = max_seq_len

    def get_split_question_sequence_data(self):
        return self._df

    def get_metadata(self, name):
        assert name == "max_seq_len"
        return self._max_seq_len


def test_model_data_uses_question_level_base():
    assert issubclass(ReKTPModelData, QuestionModelData)


def test_question_skill_table_is_unique_sorted_and_padded():
    skill_ids, skill_mask = build_question_skill_table(_RelationOnlyDataSource())

    np.testing.assert_array_equal(skill_ids, [[0, 1], [2, 3], [0, 1]])
    np.testing.assert_array_equal(
        skill_mask,
        [[True, True], [True, False], [True, True]],
    )


def test_build_time_sequences_accumulates_dwell_for_assistments09():
    # assistments09's timestamp column is order_id, so dwell times are
    # accumulated instead; the timestamp column must be ignored.
    df = pl.DataFrame(
        {
            "user": [0, 0, 1, 1],
            "seq_pos": [0, 1, 0, 1],
            "question": [0, 1, 2, 3],
            "label": [1, 0, 1, 0],
            "ms_first_response": [5000.0, 2000.0, 1000.0, None],
            "timestamp": [999, 999, 999, 999],
        }
    )
    data_src = _SplitDataDataSource("assistments09", df, max_seq_len=2)
    times = ReKTPModelData(data_src)._build_time_sequences()
    # t: 0, then +dwell/1000 + 1 per step; None dwell counts as 0 seconds.
    np.testing.assert_allclose(times, [[0.0, 6.0], [0.0, 2.0]])


def test_build_time_sequences_converts_real_timestamps_to_seconds():
    df = pl.DataFrame(
        {
            "user": [0, 0, 1],
            "seq_pos": [0, 1, 0],
            "question": [0, 1, 0],
            "label": [1, 0, 1],
            "timestamp": [1000, 2500, 10000],
        }
    )
    data_src = _SplitDataDataSource("junyi2015", df, max_seq_len=2)
    times = ReKTPModelData(data_src)._build_time_sequences()
    np.testing.assert_allclose(times, [[1.0, 2.5], [10.0, 0.0]])


def test_build_time_sequences_falls_back_to_positions_without_timestamps():
    df = pl.DataFrame(
        {
            "user": [0, 0],
            "seq_pos": [0, 1],
            "question": [0, 1],
            "label": [1, 0],
        }
    )
    data_src = _SplitDataDataSource("future_dataset", df, max_seq_len=2)
    times = ReKTPModelData(data_src)._build_time_sequences()
    np.testing.assert_allclose(times, [[0.0, 1.0]])


def test_derive_max_gap_bins_covers_span_plus_one():
    for span in (1, 2, 3, 7, 8, 15, 16, 100, 1024, 114580):
        bins = derive_max_gap_bins(np.array([[0.0, float(span)]]))
        # The no-predecessor gap is span + 1 at most; its bucket must fit.
        max_gap = span + 1
        assert math.floor(math.log2(max_gap)) < bins, (span, bins)


def test_derive_max_gap_bins_value_and_minimum():
    assert derive_max_gap_bins(np.array([[0.0, 16.0]])) == 6
    assert derive_max_gap_bins(np.zeros((3, 5))) == 2
