from abc import ABC, abstractmethod
import os


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
    """

    def __init__(self, dataset: str, data_base_path: str, data_url: str = None):
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
