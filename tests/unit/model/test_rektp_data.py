import numpy as np
import polars as pl

from model.ReKTP.ReKTP_data import ReKTPModelData, build_question_skill_table
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


def test_model_data_uses_question_level_base():
    assert issubclass(ReKTPModelData, QuestionModelData)


def test_question_skill_table_is_unique_sorted_and_padded():
    skill_ids, skill_mask = build_question_skill_table(_RelationOnlyDataSource())

    np.testing.assert_array_equal(skill_ids, [[0, 1], [2, 3], [0, 1]])
    np.testing.assert_array_equal(
        skill_mask,
        [[True, True], [True, False], [True, True]],
    )
