from .data_utility import DataSource


class Assistments2017Data(DataSource):
    """
    Assistments2017数据集处理类
    """

    def __init__(self, args):
        super().__init__(
            dataset="assistments17", data_base_path=args.data_base_path, data_url=""
        )
        self.args = args

    def load_src_data(self):
        # 实现数据加载逻辑
        pass

    def load_processed_data(self):
        # 实现预处理数据加载逻辑
        pass

    def clear_data(self):
        # 实现数据清理逻辑
        pass

    def fetch_data(self):
        # 实现数据下载逻辑
        pass

    def save_data(self, data_path):
        # 实现数据保存逻辑
        pass
