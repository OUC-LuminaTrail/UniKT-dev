import os
import pandas as pd
from typing_extensions import override
from .data_source import (
    DataSource,
    restrains_sequence_length,
    build_question_data_from_cleared,
    map_to_continuous_ids,
)
from utils.core import get_logger, register_data_source

logger = get_logger(__name__)


@register_data_source("assistments12")
class Assistments2012Data(DataSource):
    """
    Assistments 2012 数据集处理类
    """

    def __init__(self, args):
        super().__init__(
            dataset="assistments12",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments12.zip",
            seed=args.seed,
        )
        self.args = args
        # 原始数据文件路径
        self.raw_data_path = os.path.join(
            self.data_folder, "raw", "2012-2013-data-with-predictions-4-final.csv"
        )

    @override
    def load_src_data(self):
        """
        加载原始数据
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        logger.info(f"Loading raw data from: {self.raw_data_path}")
        self.raw_data = pd.read_csv(
            self.raw_data_path, encoding="latin1", low_memory=False
        )

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
        # 放弃不需要的列
        data = self.raw_data.drop(
            columns=[
                "problem_log_id",
                "skill",
                # "problem_id",
                # "user_id",
                # "assignment_id",
                "assistment_id",
                # "start_time",
                "end_time",
                "problem_type",
                "original",
                # "correct",
                "bottom_hint",
                # "hint_count",
                "actions",
                # "attempt_count",
                "ms_first_response",
                "tutor_mode",
                "sequence_id",
                "student_class_id",
                "position",
                "type",
                "base_sequence_id",
                # "skill_id",
                "teacher_id",
                "school_id",
                "overlap_time",
                # "template_id",
                "answer_id",
                "answer_text",
                "first_action",
                "problemlogid",
                "Average_confidence(FRUSTRATED)",
                "Average_confidence(CONFUSED)",
                "Average_confidence(CONCENTRATING)",
                "Average_confidence(BORED)",
            ]
        )
        # 重命名列
        data = data.rename(
            columns={
                "user_id": "user",
                "problem_id": "question",
                "correct": "label",
                "assignment_id": "assignment",
                "skill_id": "skill",
                "template_id": "template",
            }
        )
        # 按时间排序
        data = data.sort_values(by=["user", "start_time"])
        # 转换数据类型
        data["user"] = data["user"].astype(int)
        # 清除没有技能的问题
        data = data[data["skill"].notna()]
        # 清理label列中的异常值，只保留0和1
        data = data[data["label"].isin([0, 1])]
        # 移除重复的行
        data = data.drop_duplicates()
        # 重置索引
        data = data.reset_index(drop=True)

        # 限制序列长度
        data = restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )
        # 将问题ID和技能ID转换为连续整数
        data = map_to_continuous_ids(
            data, columns=["user", "question", "skill", "assignment", "template"]
        )

        self.cleared_data = data.copy()
        self.sequence_data = data.copy()
        self.question_data = build_question_data_from_cleared(
            self.cleared_data, skill_column="skill", question_column="question"
        )

        # 保存元信息
        self.add_metadatas(
            {
                "num_users": self.cleared_data["user"].nunique(),
                "num_questions": self.question_data["question"].nunique(),
                "num_skills": self.question_data["skill"].nunique(),
                "num_assignments": self.question_data["assignment"].nunique(),
                "num_templates": self.question_data["template"].nunique(),
                "max_seq_len": self.args.max_seq_len,
                "min_seq_len": self.args.min_seq_len,
                "columns": data.columns.tolist(),
            }
        )


__all__ = ["Assistments2012Data"]
