from abc import ABC, abstractmethod

from utils.data_process import DataSource


class BaseModelData(ABC):
    r"""
    模型数据基类

    参数:
        data_src: 数据源对象
    """

    def __init__(self, data_src: DataSource):
        self.data_src = data_src
        self.data_src.load_processed_data()

    @abstractmethod
    def prepare_data(self, args):
        """
        准备模型所需的数据
        """
        raise NotImplementedError("Subclasses should implement prepare_data method")

    def split_kfold_data(self, *arrays, fold_idx: int):
        r"""
        根据K折交叉验证的fold索引获取训练集、验证集和测试集

        参数:
            *arrays: 任意个数、首维为用户数的数组或张量。
            fold_idx: 当前的fold索引（关键字参数，必填）。

        返回:
            train_data: 与输入相同结构的元组，包含训练集切片
            val_data:   与输入相同结构的元组，包含验证集切片
            test_data:  与输入相同结构的元组，包含测试集切片

        说明:
            - 验证集为指定fold (fold_idx) 的数据，测试集为 fold == -1 的数据，训练集为剩余的fold数据
            - 需要数据源中有用户到行索引的映射信息
        """
        import numpy as np
        from tqdm import tqdm

        # 校验输入参数
        if len(arrays) == 0:
            raise ValueError(
                "get_kfold_split_data requires at least one input array/tensor"
            )

        # 加载数据以获取折信息
        data = self.data_src.get_sequence_data().to_pandas()
        # 检查是否已添加fold列
        if "fold" not in data.columns:
            raise ValueError(
                "K-fold labels not found in data. Please call data_src.add_kfold_labels() first."
            )

        # 获取有效的用户索引
        num_users = arrays[0].shape[0]
        # 校验所有输入的用户数一致
        for i, arr in enumerate(arrays):
            if arr.shape[0] != num_users:
                raise ValueError(
                    f"Input array {i} shape is {arr.shape}, but expected shape is (num_users, *)"
                )

        # 创建用户fold信息映射
        user_folds = np.ones(num_users, dtype=int) * -1
        for row in tqdm(
            data.itertuples(),
            total=data.shape[0],
            desc=f"Mapping users to fold {fold_idx}",
        ):
            user_idx = row.user
            fold_label = row.fold
            if user_idx < num_users:
                user_folds[user_idx] = fold_label

        # 根据fold标签分割用户数据
        # 验证集：fold == fold_idx
        # 测试集：fold == -1
        # 训练集：fold != fold_idx 且 fold != -1
        train_user_indices = np.where((user_folds != fold_idx) & (user_folds != -1))[0]
        val_user_indices = np.where(user_folds == fold_idx)[0]
        test_user_indices = np.where(user_folds == -1)[0]

        train_idx_list = train_user_indices[train_user_indices < num_users].tolist()
        val_idx_list = val_user_indices[val_user_indices < num_users].tolist()
        test_idx_list = test_user_indices[test_user_indices < num_users].tolist()

        train_slices = []
        val_slices = []
        test_slices = []
        for arr in arrays:
            # 识别 torch.Tensor
            is_torch_tensor = False
            try:
                import torch  # noqa: F401

                is_torch_tensor = hasattr(arr, "dim") and hasattr(arr, "index_select")
            except Exception:
                is_torch_tensor = False

            if is_torch_tensor:
                import torch

                train_idx = torch.tensor(
                    train_idx_list, dtype=torch.long, device=arr.device
                )
                val_idx = torch.tensor(
                    val_idx_list, dtype=torch.long, device=arr.device
                )
                test_idx = torch.tensor(
                    test_idx_list, dtype=torch.long, device=arr.device
                )
                train_slices.append(arr.index_select(0, train_idx))
                val_slices.append(arr.index_select(0, val_idx))
                test_slices.append(arr.index_select(0, test_idx))
            else:
                train_slices.append(arr[train_idx_list])
                val_slices.append(arr[val_idx_list])
                test_slices.append(arr[test_idx_list])

        return tuple(train_slices), tuple(val_slices), tuple(test_slices)

    def split_data(self, *arrays, val_ratio: float = 0.2, test_ratio: float = 0.0):
        r"""
        随机划分训练集、验证集和测试集。

        参数:
            *arrays: 任意个数、首维为样本数的数组或张量。
                     例如：
                     - GIKT: (sequences, responses, masks)
                     - SQGKT: (sequences, responses, masks, user_id_sequence)
            val_ratio: 验证集比例(默认为0.2)
            test_ratio: 测试集比例(默认为0.0，不分割测试集)

        返回:
            如果 test_ratio == 0:
                (train_data, val_data):
                    - train_data: 与输入相同结构的元组，包含训练集切片
                    - val_data:   与输入相同结构的元组，包含验证集切片
            如果 test_ratio > 0:
                (train_data, val_data, test_data):
                    - train_data: 与输入相同结构的元组，包含训练集切片
                    - val_data:   与输入相同结构的元组，包含验证集切片
                    - test_data:  与输入相同结构的元组，包含测试集切片

        说明:
            - 将依据第一个输入的首维作为样本维度进行打乱与划分。
            - 要求所有输入的首维大小一致。
            - 同时兼容 numpy.ndarray 与 torch.Tensor（若可用）。
            - 分割顺序：先分割测试集，再从剩余数据中分割验证集，最后为训练集
        """
        import numpy as np

        if len(arrays) == 0:
            raise ValueError("split_data requires at least one input array/tensor")

        # 校验比例参数
        if val_ratio + test_ratio >= 1.0:
            raise ValueError(
                f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be less than 1.0"
            )

        num_users = arrays[0].shape[0]

        # 校验所有数组首维一致
        for i, arr in enumerate(arrays):
            if arr.shape[0] != num_users:
                raise ValueError(
                    f"Input array {i} shape is {arr.shape}, but expected shape is (num_users, *)"
                )

        indices = np.arange(num_users)
        np.random.shuffle(indices)
        indices = indices.tolist()

        # 分割测试集
        test_size = int(num_users * test_ratio)
        test_indices = indices[:test_size]
        remaining_indices = indices[test_size:]

        # 从剩余数据中分割验证集和训练集
        val_size = int(len(remaining_indices) * val_ratio)
        val_indices = remaining_indices[:val_size]
        train_indices = remaining_indices[val_size:]

        train_slices = []
        val_slices = []
        test_slices = []

        for arr in arrays:
            # 尝试识别 torch.Tensor
            is_torch_tensor = False
            try:
                import torch  # noqa: F401

                is_torch_tensor = hasattr(arr, "dim") and hasattr(arr, "index_select")
            except Exception:
                is_torch_tensor = False

            if is_torch_tensor:
                import torch

                train_idx = torch.tensor(
                    train_indices, dtype=torch.long, device=arr.device
                )
                val_idx = torch.tensor(val_indices, dtype=torch.long, device=arr.device)
                test_idx = torch.tensor(
                    test_indices, dtype=torch.long, device=arr.device
                )
                train_slices.append(arr.index_select(0, train_idx))
                val_slices.append(arr.index_select(0, val_idx))
                test_slices.append(arr.index_select(0, test_idx))
            else:
                # 视作 numpy 数组或支持 list 索引的结构
                train_slices.append(arr[train_indices])
                val_slices.append(arr[val_indices])
                test_slices.append(arr[test_indices])

        train_data = tuple(train_slices)
        val_data = tuple(val_slices)
        test_data = tuple(test_slices)

        if test_ratio > 0:
            return train_data, val_data, test_data
        else:
            return train_data, val_data

    def calculate_question_difficulty(self, exclude_fold: int = None):
        """
        计算每个问题的难度指标

        基于以下特征计算难度：
        1. 正确率（correct_rate）：正确回答次数 / 总回答次数
        2. 平均作答时间（avg_time）：如果数据集有时间字段
        3. 提示率（hint_rate）：如果数据集有提示字段

        参数:
            exclude_fold: 要排除的fold索引（用于在交叉验证时排除验证集数据）

        返回:
            dict: 问题ID -> 难度分数的字典，难度分数为0-1之间，越大表示越难

        公式：
            difficulty = (1 - correct_rate) * 0.6 + normalized_time * 0.3 + hint_rate * 0.1
        """
        import tqdm

        data = self.data_src.get_sequence_data().to_pandas()

        # 如果指定了排除的fold，则过滤掉该fold的数据
        if exclude_fold is not None and "fold" in data.columns:
            data = data[data["fold"] != exclude_fold]
            self.logger.info(
                f"Excluding fold {exclude_fold} from difficulty calculation."
            )

        # 计算每个问题的正确率
        question_stats = (
            data.groupby("question")
            .agg({"label": ["mean", "count"]})  # 正确率和回答次数
            .reset_index()
        )
        question_stats.columns = ["question", "correct_rate", "count"]

        # 错误率作为难度的主要指标
        question_stats["error_rate"] = 1 - question_stats["correct_rate"]

        # 标准化错误率
        difficulty_scores = {}
        for _, row in tqdm.tqdm(
            question_stats.iterrows(),
            total=len(question_stats),
            desc="Calculating question difficulty",
        ):
            qid = int(row["question"])
            # 加权：错误率占主要权重，但考虑样本数（少样本的难度评估不太可靠）
            confidence = min(row["count"] / 10.0, 1.0)  # 10次以上回答视为可靠
            difficulty = row["error_rate"] * confidence + 0.5 * (1 - confidence)
            difficulty_scores[qid] = float(difficulty)

        return difficulty_scores

    def build_relationship_matrix(
        self, edge_type: tuple[str, str, str], value_type: str = "binary"
    ):
        """
        构建实体之间的关系矩阵

        参数:
            edge_type: 边类型三元组 (源节点类型, 边关系名, 目标节点类型)
                      节点类型对应数据中的列名（如 'user', 'question', 'skill', 'template', 'assignment'等）
                      例如: ('user', 'answers', 'question')
                           ('question', 'has', 'skill')
                           ('question', 'belongs_to', 'template')
                           ('skill', 'related_to', 'assignment')
            value_type: 矩阵值类型，可选:
                       'binary': 二值矩阵,表示是否存在关系 (默认)
                       'count': 计数矩阵,表示关系出现的次数

        返回:
            data_matrix: numpy数组,形状为 (源节点数量, 目标节点数量)

        示例:
            # 构建用户-问题二值关系矩阵
            matrix = model_data.build_data_matrix(('user', 'answers', 'question'))

            # 构建问题-技能关系矩阵
            matrix = model_data.build_data_matrix(('question', 'has', 'skill'))

            # 构建问题-模板关系矩阵
            matrix = model_data.build_data_matrix(('question', 'belongs_to', 'template'))

            # 构建技能-作业关系矩阵
            matrix = model_data.build_data_matrix(('skill', 'related_to', 'assignment'))

            # 构建用户-问题计数矩阵
            matrix = model_data.build_data_matrix(('user', 'answers', 'question'), value_type='count')
        """
        import numpy as np
        from tqdm import tqdm

        data = self.data_src.get_question_data().to_pandas()

        src_type, _, dst_type = edge_type

        # 直接使用节点类型作为列名
        src_col = src_type
        dst_col = dst_type

        # 验证列是否存在
        if src_col not in data.columns or dst_col not in data.columns:
            raise ValueError(
                f"Required columns '{src_col}' or '{dst_col}' not found in data. "
                f"Available columns: {data.columns.tolist()}"
            )

        # 获取节点数量
        # 首先尝试从元数据获取
        src_meta_key = f"num_{src_type}s"
        dst_meta_key = f"num_{dst_type}s"

        try:
            num_src = self.data_src.get_metadata(src_meta_key)
        except (KeyError, AttributeError):
            # 如果元数据中没有，从数据中计算
            num_src = data[src_col].nunique()
            self.logger.warning(
                f"{src_meta_key} not found in metadata, calculated from data: {num_src}"
            )

        try:
            num_dst = self.data_src.get_metadata(dst_meta_key)
        except (KeyError, AttributeError):
            # 如果元数据中没有，从数据中计算
            num_dst = data[dst_col].nunique()
            self.logger.warning(
                f"{dst_meta_key} not found in metadata, calculated from data: {num_dst}"
            )

        # 初始化矩阵
        data_matrix = np.zeros((num_src, num_dst), dtype=int)

        # 填充矩阵
        for row in tqdm(
            data.itertuples(),
            total=data.shape[0],
            desc=f"Building {src_type}-{dst_type} matrix",
        ):
            src_idx = getattr(row, src_col)
            dst_idx = getattr(row, dst_col)

            # 跳过无效索引（NaN或超出范围）
            if (
                src_idx is None
                or dst_idx is None
                or np.isnan(src_idx)
                or np.isnan(dst_idx)
                or src_idx < 0
                or dst_idx < 0
                or src_idx >= num_src
                or dst_idx >= num_dst
            ):
                continue

            if value_type == "binary":
                data_matrix[int(src_idx), int(dst_idx)] = 1
            elif value_type == "count":
                data_matrix[int(src_idx), int(dst_idx)] += 1
            else:
                raise ValueError(
                    f"Unsupported value_type: {value_type}. Supported types: 'binary', 'count'"
                )

        return data_matrix
