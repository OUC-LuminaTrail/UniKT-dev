import os
import pandas as pd
from typing_extensions import override
from .data_source import *


class Assistments2009Data(DataSource):
    """
    Assistments 2009-2010 数据集处理类
    数据集来源: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
    """

    def __init__(self, args):
        super().__init__(
            dataset="assistments09",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/assistments09.zip",
            seed=args.seed,
        )
        self.args = args
        # 原始数据文件路径
        self.raw_data_path = os.path.join(
            self.data_folder, "raw", "skill_builder_data_corrected_collapsed.csv"
        )

    @override
    def load_src_data(self):
        """
        加载原始数据
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        print("Loading raw data from:", self.raw_data_path)
        self.raw_data = pd.read_csv(
            self.raw_data_path, encoding="latin1", low_memory=False
        )

    @override
    def clear_data(self):
        print("Processing Data...")
        if self.raw_data is None:
            try:
                self.load_src_data()
            except FileNotFoundError:
                raise ValueError(
                    "Original data loading failed. Please check the data file."
                )

        data = self.raw_data.drop(
            columns=[
                "Unnamed: 0",
                # "order_id",
                # "assignment_id",
                # "user_id",
                "assistment_id",
                # "problem_id",
                # "original",
                # "correct",
                "attempt_count",
                "ms_first_response",
                "tutor_mode",
                "answer_type",
                "sequence_id",
                "student_class_id",
                "position",
                "type",
                "base_sequence_id",
                # "skill_id",
                "skill_name",
                "teacher_id",
                "school_id",
                "hint_count",
                "hint_total",
                "overlap_time",
                # "template_id",
                "answer_id",
                "answer_text",
                "first_action",
                "bottom_hint",
                "opportunity",
                "opportunity_original",
            ]
        )
        # 重新命名列
        data = data.rename(
            columns={
                "correct": "label",
                "user_id": "user",
                "problem_id": "question",
                "assignment_id": "assignment",
                "skill_id": "skill",
                "template_id": "template",
            }
        )
        # 清除缺失值
        data = data.dropna(subset=["user", "skill", "label"])
        # 移除重复的行
        data = data.drop_duplicates()
        # 按照时间排序
        data = data.sort_values(by=["user", "order_id"])
        # 限制序列长度到指定范围
        data = restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )
        # 将数据重编码为连续整数
        data = map_to_continuous_ids(
            data, columns=["user", "question", "assignment", "template"]
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


__all__ = ["Assistments2009Data"]
