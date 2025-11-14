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
            data_url="",
            seed=args.seed,
        )
        self.args = args
        # 原始数据文件路径
        self.raw_data_path = os.path.join(
            self.data_folder, "skill_builder_data_corrected.csv"
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
                "school_id",
                "skill_name",
                "teacher_id",
                "opportunity",
                "opportunity_original",
                "overlap_time",
                "type",
                "tutor_mode",
                "bottom_hint",
                "position",
                "answer_text",
                "answer_id",
            ]
        )
        # 重新命名列
        data = data.rename(
            columns={
                "correct": "label",
                "problem_id": "question_id",
            }
        )
        # 转换数据类型
        data["user_id"] = data["user_id"].astype(int)
        # 清除缺失值
        data = data.dropna(subset=["user_id", "skill_id", "label"])
        # 重置索引
        data = data.reset_index(drop=True)
        # 限制序列长度
        data = DataSource.restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )
        # 将问题ID和技能ID重编码为连续整数
        data = DataSource.map_to_continuous_ids(
            data, columns=["user_id", "question_id", "skill_id"]
        )

        self.processed_data = data

        # 保存元信息
        self.add_metadata("num_users", data["user_id"].nunique())
        self.add_metadata("num_questions", data["question_id"].nunique())
        self.add_metadata("num_skills", data["skill_id"].nunique())
        self.add_metadata("max_seq_len", self.args.max_seq_len)
        self.add_metadata("min_seq_len", self.args.min_seq_len)
        self.add_metadata("columns", data.columns.tolist())

    @override
    def fetch_data(self):
        pass


__all__ = ["Assistments2009Data"]
