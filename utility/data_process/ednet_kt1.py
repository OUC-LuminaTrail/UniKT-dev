from .data_utility import DataSource 


class EdNetKT1Data(DataSource):
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.processed_data = None

    def load_data(self):
        # 实现数据加载逻辑
        pass

    def clear_data(self):
        # 实现数据预处理逻辑
        pass