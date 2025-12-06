import os
import pandas as pd
from typing_extensions import override
from .data_utility import DataSource


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
        # 转换数据类型
        data["user"] = data["user"].astype(int)
        # 清除缺失值
        data = data.dropna(subset=["user", "skill", "label"])
        # 重置索引
        data = data.reset_index(drop=True)
        # 限制序列长度
        data = DataSource.restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )
        # 将数据重编码为连续整数
        data = DataSource.map_to_continuous_ids(
            data, columns=["user", "question", "skill", "assignment", "template"]
        )
        # 安装时间排序
        data = data.sort_values(by=["user", "order_id"])

        self.processed_data = data

        # 保存元信息
        self.add_metadata("num_users", data["user"].nunique())
        self.add_metadata("num_questions", data["question"].nunique())
        self.add_metadata("num_skills", data["skill"].nunique())
        self.add_metadata("num_assignments", data["assignment"].nunique())
        self.add_metadata("num_templates", data["template"].nunique())
        self.add_metadata("max_seq_len", self.args.max_seq_len)
        self.add_metadata("min_seq_len", self.args.min_seq_len)
        self.add_metadata("columns", data.columns.tolist())


__all__ = ["Assistments2009Data"]
