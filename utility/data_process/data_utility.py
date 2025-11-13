from abc import ABC, abstractmethod
import os
from sklearn.model_selection import KFold


class DataSource(ABC):
    """
    数据源基类

    参数:
        dataset: 数据集名称（自动转换为小写）
        data_base_path: 数据存储的基础路径
        data_url: 数据下载链接 (可选)

    属性:
        dataset: 数据集名称
        data_base_path: 数据存储的基础路径
        data_folder: 数据集文件夹路径
        data_processed_folder: 预处理后数据的存储路径
        raw_data: 原始数据 (Pandas DataFrame)
        processed_data: 预处理后的数据 (Pandas DataFrame)
        data_url: 数据下载链接 (可选)
        metadata: 数据元信息字典

    已实现的方法:
        add_metadata(key, value): 添加数据元信息
        save_metadata(): 保存数据元信息到 JSON 文件
        add_kfold_labels(n_splits, random_state, user_id_column): 添加K折交叉验证的分层划分标签
    """

    def __init__(
        self, dataset: str, data_base_path: str, data_url: str = None, seed: int = 42
    ):
        super().__init__()
        self.dataset = dataset.lower()
        # 数据存储的基础路径
        self.data_base_path = data_base_path
        # 数据集文件夹路径
        self.data_folder = os.path.join(self.data_base_path, self.dataset)
        # 元数据JSON文件路径
        self.metadata_path = os.path.join(self.data_folder, "metadata.json")
        self.raw_data = None
        self.processed_data = None
        self.data_url = data_url
        self.metadata = {}
        # 设置随机种子
        self.seed = seed
        self.set_random_seed()

    def set_random_seed(self):
        import random
        import numpy as np

        random.seed(42)
        np.random.seed(42)

        self.add_metadata("random_seed", self.seed)

    @abstractmethod
    def fetch_data(self):
        """
        下载数据
        """
        raise NotImplementedError("Subclasses should implement fetch_data method")

    @abstractmethod
    def load_src_data(self):
        """
        加载原始数据
        """
        raise NotImplementedError("Subclasses should implement load_data method")

    @abstractmethod
    def load_processed_data(self):
        """
        加载预处理后的数据
        """
        raise NotImplementedError(
            "Subclasses should implement load_processed_data method"
        )

    @abstractmethod
    def clear_data(self):
        """
        预处理数据

        注：如果原始数据未加载，应先调用 load_src_data()
        处理完成后应将结果存储在 self.processed_data 中
        """
        raise NotImplementedError("Subclasses should implement clear_data method")

    @abstractmethod
    def save_data(self):
        """
        保存预处理后的数据

        注：在该方法中应调用 save_metadata() 保存元信息
        """
        raise NotImplementedError("Subclasses should implement save_data method")

    def get_processed_data(self):
        """
        获取预处理后的数据

        返回:
            预处理后的数据
        """
        if self.processed_data is None:
            try:
                self.load_processed_data()
            except FileNotFoundError:
                raise ValueError(
                    "No processed data available. Please run clear_data() first."
                )
        return self.processed_data

    def add_metadata(self, key: str, value):
        """
        添加数据元信息

        参数:
            key: 元信息键
            value: 元信息值
        """
        self.metadata[key] = value

    def save_metadata(self):
        """
        保存数据元信息

        必须保存的元信息:
        - num_users: 学生总数
        - num_questions: 题目总数
        - num_skills: 技能总数
        - max_seq_len: 最大序列长度
        - min_seq_len: 最小序列长度
        """
        self.add_metadata("dataset", self.dataset)
        self.add_metadata("data_base_path", self.data_base_path)
        import json

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def load_metadata(self):
        """
        加载数据元信息
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        import json

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

    def get_metadata(self, key: str | None = None):
        """
        获取指定键的元信息

        参数:
            key: 元信息键

        返回:
            元信息值
        """
        if not self.metadata:
            self.load_metadata()
        if key is None:
            return self.metadata
        return self.metadata.get(key, None)

    def add_kfold_labels(self, n_splits: int = 5):
        """
        为数据集添加K折交叉验证的分层划分标签

        按用户维度进行分层 K折划分，确保每个用户的所有数据都在同一个fold中，
        避免数据泄露。

        参数:
            n_splits: K折的数量，默认为5
            random_state: 随机种子，确保可重复性，默认为42

        返回:
            添加了fold标签的数据集 DataFrame（列名为 'fold'，值为 0 到 n_splits-1）

        异常:
            ValueError: 如果processed_data未加载或user_id_column列不存在

        说明:
            - 添加的新列名为 'fold'（值为 0 到 n_splits-1）
            - 如果按用户分层，每个用户的所有数据都会分到同一个fold
            - 元数据中会记录 'kfold_n_splits' 和 'kfold_random_state'
            - 会覆盖已存在的 'fold' 列
        """
        import pandas as pd
        from tqdm import tqdm

        if self.processed_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )

        print(f"Adding K-Fold labels with n_splits={n_splits}")

        # 复制数据以避免修改原始数据
        data = self.processed_data.copy()
        data["fold"] = -1

        # 获取唯一的用户ID
        unique_users = data["user_id"].unique()
        user_to_fold = {}

        # 对用户ID进行KFold划分
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for fold_idx, (_, test_user_idx) in tqdm(
            enumerate(kfold.split(unique_users)), total=n_splits, desc="Assigning folds"
        ):
            test_users = unique_users[test_user_idx]
            for user in test_users:
                user_to_fold[user] = fold_idx

        # 为每个用户的所有行分配对应的fold值
        data["fold"] = data["user_id"].map(user_to_fold)

        # 更新processed_data
        self.processed_data = data

        # 更新元数据
        self.add_metadata("kfold_n_splits", n_splits)

        return data
