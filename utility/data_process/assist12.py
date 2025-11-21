import os
import pandas as pd
from typing_extensions import override
from .data_utility import DataSource


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
        r"""
        加载原始数据
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_path}")
        print("Loading raw data from:", self.raw_data_path)
        self.raw_data = pd.read_csv(
            self.raw_data_path, encoding="latin1", low_memory=False
        )

    def clear_data(self):
        print("Processing Data...")
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
                "position",
                "answer_id",
                "answer_text",
                "problemlogid",
                "overlap_time",
                "ms_first_response",
                "problem_type",
                "skill",
            ]
        )
        # 重命名列
        data.rename(
            columns={
                "problem_id": "question_id",
                "correct": "label",
            },
            inplace=True,
        )
        # 按时间排序
        data.sort_values(by=["user_id", "start_time"], inplace=True)
        # 转换数据类型
        data["user_id"] = data["user_id"].astype(int)
        # 清除没有技能的问题
        data = data[data["skill_id"].notna()]
        # 清理label列中的异常值，只保留0和1
        data = data[data["label"].isin([0, 1])]
        # 重置索引
        data = data.reset_index(drop=True)

        # 限制序列长度
        data = DataSource.restrains_sequence_length(
            data, self.args.min_seq_len, self.args.max_seq_len
        )
        # 将问题ID和技能ID转换为连续整数
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
