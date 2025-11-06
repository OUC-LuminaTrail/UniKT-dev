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
            dataset="assistment09", data_base_path=args.data_base_path, data_url=""
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
            raise FileNotFoundError(f"未找到数据文件: {self.raw_data_path}")
        self.raw_data = pd.read_csv(
            self.raw_data_path, encoding="latin1", low_memory=False
        )

    @override
    def load_processed_data(self):
        """
        加载预处理后的数据
        """
        if not os.path.exists(self.data_processed_folder):
            raise FileNotFoundError(
                f"未找到预处理数据文件: {self.data_processed_folder}"
            )
        self.processed_data = pd.read_parquet(self.data_processed_folder)

    @override
    def clear_data(self):
        """
        清理数据
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")

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

        # 过滤掉答题次数少于min_seq_len的学生
        min_seq_len = self.args.min_seq_len
        is_valid_user = data.groupby("user_id").size() >= min_seq_len
        valid_user_ids = is_valid_user[is_valid_user].index.tolist()
        data = data[data["user_id"].isin(valid_user_ids)]
        # 过滤掉序列长度超过max_seq_len的学生
        max_seq_len = self.args.max_seq_len
        is_valid_user = data.groupby("user_id").size() <= max_seq_len
        valid_user_ids = is_valid_user[is_valid_user].index.tolist()
        data = data[data["user_id"].isin(valid_user_ids)]

        # 将问题ID和技能ID重编码为连续整数
        data["user_id"] = data["user_id"].astype("category").cat.codes.astype(int)
        data["question_id"] = (
            data["question_id"].astype("category").cat.codes.astype(int)
        )
        data["skill_id"] = data["skill_id"].astype("category").cat.codes.astype(int)

        self.processed_data = data

    @override
    def save_data(self):
        """
        保存预处理后的数据
        """
        if self.processed_data is None:
            raise ValueError("请先清理数据")

        data_processed_path = os.path.join(
            self.data_base_path, f"{self.dataset}_processed"
        )
        self.processed_data.to_parquet(data_processed_path, index=False)

    @override
    def fetch_data(self):
        pass


__all__ = ["Assistments2009Data"]
