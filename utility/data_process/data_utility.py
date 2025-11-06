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
    """

    def __init__(self, dataset: str, data_base_path: str, data_url: str = None):
        super().__init__()
        self.dataset = dataset.lower()
        self.data_base_path = data_base_path
        self.data_folder = os.path.join(self.data_base_path, self.dataset)
        self.data_processed_folder = os.path.join(
            self.data_base_path, f"{self.dataset}_processed"
        )
        self.raw_data = None
        self.processed_data = None
        self.data_url = data_url

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
        raise NotImplementedError("Subclasses should implement load_processed_data method")

    @abstractmethod
    def clear_data(self):
        """
        预处理数据
        """
        raise NotImplementedError("Subclasses should implement clear_data method")

    @abstractmethod
    def save_data(self, data_path: str):
        """
        保存预处理后的数据

        参数:
            data_path: 保存数据的路径
        """
        raise NotImplementedError("Subclasses should implement save_data method")

    def get_processed_data(self):
        """
        获取预处理后的数据

        返回:
            预处理后的数据
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Please run clear_data() first.")
        return self.processed_data