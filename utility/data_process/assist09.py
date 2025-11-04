import os
import pandas as pd
from typing_extensions import override
from .data_utility import DataSource


class Assistments2009Data(DataSource):
    """
    Assistments 2009-2010 数据集处理类
    数据集来源: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data
    """

    def __init__(self, data_base_path: str, data_url: str = ""):
        super().__init__(dataset="assistment09", data_url=data_url)
        # 数据文件夹路径
        self.data_folder = os.path.join(data_base_path, self.dataset)
        # 原始数据文件路径
        self.raw_data_path = os.path.join(
            self.data_folder, "skill_builder_data_corrected.csv"
        )

    @override
    def load_data(self):
        """
        加载原始数据
        """
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"未找到数据文件: {self.raw_data_path}")
        self.raw_data = pd.read_csv(
            self.raw_data_path, encoding="latin1", low_memory=False
        )

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
            }
        )
        # 转换数据类型
        data["user_id"] = data["user_id"].astype(int)
        # 清除缺失值
        data = data.dropna(subset=["user_id", "skill_id", "label"])
        # 重置索引
        data = data.reset_index(drop=True)

        self.processed_data = data

    @override
    def save_data(self, data_path: str):
        """
        保存预处理后的数据

        参数:
            data_path: 保存数据的路径
        """


__all__ = ["Assistments2009Data"]
