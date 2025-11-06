from abc import ABC, abstractmethod


class DataSource(ABC):
    """
    数据源基类
    """

    def __init__(self, dataset: str, data_url: str = None):
        super().__init__()
        self.dataset = dataset.lower()
        self.data_path = None
        self.raw_data = None
        self.processed_data = None
        self.data_url = data_url

    @abstractmethod
    def fetch_data(self):
        """
        下载数据
        """
        pass


    @abstractmethod
    def load_data(self):
        """
        加载原始数据
        """
        pass

    @abstractmethod
    def clear_data(self):
        """
        预处理数据
        """
        pass

    @abstractmethod
    def save_data(self, data_path: str):
        """
        保存预处理后的数据

        参数:
            data_path: 保存数据的路径
        """
        pass

    def get_data(self):
        """
        获取预处理后的数据

        返回:
            预处理后的数据
        """
        if self.processed_data is None:
            raise ValueError("无已预处理数据")
        return self.processed_data