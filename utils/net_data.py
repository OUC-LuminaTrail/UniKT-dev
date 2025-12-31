from abc import ABC, abstractmethod
from utils.data_process import DataSource
from utils.core import get_logger


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
        根据K折交叉验证的fold索引获取训练集和验证集

        参数:
            *arrays: 任意个数、首维为样本数的数组或张量（与 split_data 一致）。
                     例如：
                     - GIKT: (sequences, responses, masks)
                     - SQGKT: (sequences, responses, masks, user_id_sequence)
            fold_idx: 当前的fold索引（关键字参数，必填）。

        返回:
            train_data: 与输入相同结构的元组，包含训练集切片
            val_data:   与输入相同结构的元组，包含验证集切片

        说明:
            - 需要数据源中已添加K折标签（通过 add_kfold_labels）
            - 验证集为指定fold的数据，训练集为其他fold的数据
            - 需要数据源中有用户到行索引的映射信息
        """
        from tqdm import tqdm
        import numpy as np

        if len(arrays) == 0:
            raise ValueError(
                "get_kfold_split_data requires at least one input array/tensor"
            )

        # 加载数据以获取折信息
        data = self.data_src.get_sequence_data()

        # 检查是否已添加fold列
        if "fold" not in data.columns:
            raise ValueError(
                "K-fold labels not found in data. Please call data_src.add_kfold_labels() first."
            )

        # 获取有效的用户索引（基于序列中实际存在的用户）
        num_users = arrays[0].shape[0]

        # 校验所有输入的首维一致
        for i, arr in enumerate(arrays):
            if arr.shape[0] != num_users:
                raise ValueError(
                    f"第 {i} 个输入首维为 {arr.shape[0]}，与预期的 {num_users} 不一致"
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
        val_user_indices = np.where(user_folds == fold_idx)[0]
        train_user_indices = np.where(user_folds != fold_idx)[0]

        # 过滤掉fold标签为-1的用户（不在fold中的用户）
        val_user_indices = val_user_indices[val_user_indices < num_users]
        train_user_indices = train_user_indices[train_user_indices < num_users]

        # 索引列表
        val_idx_list = val_user_indices.tolist()
        train_idx_list = train_user_indices.tolist()

        train_slices = []
        val_slices = []
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
                train_slices.append(arr.index_select(0, train_idx))
                val_slices.append(arr.index_select(0, val_idx))
            else:
                train_slices.append(arr[train_idx_list])
                val_slices.append(arr[val_idx_list])

        return tuple(train_slices), tuple(val_slices)

    def split_data(self, *arrays, val_ratio: float = 0.2):
        r"""
        随机划分训练集和验证集（支持可变数量的输入数组/张量）。

        参数:
            *arrays: 任意个数、首维为样本数的数组或张量。
                     例如：
                     - GIKT: (sequences, responses, masks)
                     - SQGKT: (sequences, responses, masks, user_id_sequence)
            val_ratio: 验证集比例(默认为0.2)

        返回:
            (train_data, val_data):
                - train_data: 与输入相同结构的元组，包含训练集切片
                - val_data:   与输入相同结构的元组，包含验证集切片

        说明:
            - 将依据第一个输入的首维作为样本维度进行打乱与划分。
            - 要求所有输入的首维大小一致。
            - 同时兼容 numpy.ndarray 与 torch.Tensor（若可用）。
        """
        import numpy as np

        if len(arrays) == 0:
            raise ValueError("split_data requires at least one input array/tensor")

        num_users = arrays[0].shape[0]

        # 校验所有数组首维一致
        for i, arr in enumerate(arrays):
            if arr.shape[0] != num_users:
                raise ValueError(
                    f"第 {i} 个输入首维为 {arr.shape[0]}，与预期的 {num_users} 不一致"
                )

        indices = np.arange(num_users)
        np.random.shuffle(indices)
        indices = indices.tolist()

        val_size = int(num_users * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # 兼容 numpy 与 torch 的索引切片
        train_slices = []
        val_slices = []
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
                train_slices.append(arr.index_select(0, train_idx))
                val_slices.append(arr.index_select(0, val_idx))
            else:
                # 视作 numpy 数组或支持 list 索引的结构
                train_slices.append(arr[train_indices])
                val_slices.append(arr[val_indices])

        train_data = tuple(train_slices)
        val_data = tuple(val_slices)

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

        data = self.data_src.get_sequence_data()

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

    def calculate_question_discrimination(self, exclude_fold: int = None):
        """
        计算每个问题的区分度指标 (Discrimination)

        区分度反映了题目区分不同能力水平学生的能力。
        这里采用点二系列相关系数 (Point-Biserial Correlation) 的简化版本：
        计算每个题目得分与学生总平均分之间的相关性。

        参数:
            exclude_fold: 要排除的fold索引（用于在交叉验证时排除验证集数据）

        返回:
            dict: 问题ID -> 区分度分数的字典，通常在 -1 到 1 之间，越大表示区分度越高
        """
        import tqdm
        import numpy as np

        data = self.data_src.get_sequence_data()

        # 如果指定了排除的fold，则过滤掉该fold的数据
        if exclude_fold is not None and "fold" in data.columns:
            data = data[data["fold"] != exclude_fold]
            self.logger.info(
                f"Excluding fold {exclude_fold} from discrimination calculation."
            )

        # 1. 计算每个学生的平均正确率作为能力代理
        user_stats = data.groupby("user")["label"].mean().reset_index()
        user_stats.columns = ["user", "user_mean"]

        # 2. 将学生平均分合并回原始数据
        data_with_user_mean = data.merge(user_stats, on="user")

        # 3. 计算每个问题的区分度（题目得分与学生平均分的相关系数）
        discrimination_scores = {}

        # 按问题分组计算相关性
        grouped = data_with_user_mean.groupby("question")

        for qid, group in tqdm.tqdm(
            grouped,
            desc="Calculating question discrimination",
        ):
            if len(group) < 2:
                discrimination_scores[int(qid)] = 0.0
                continue

            # 计算 label 和 user_mean 的相关系数
            # np.corrcoef 返回相关系数矩阵
            corr = np.corrcoef(group["label"], group["user_mean"])[0, 1]

            # 处理 NaN 情况（例如所有学生对该题得分相同）
            if np.isnan(corr):
                corr = 0.0

            # 考虑样本量置信度
            confidence = min(len(group) / 10.0, 1.0)
            discrimination = corr * confidence
            discrimination_scores[int(qid)] = float(discrimination)

        return discrimination_scores

    def calculate_question_error_rate(self):
        """
        计算每个(问题, 技能)对的错误率

        Returns:
            error_patterns: Dict[tuple[int, int], Dict[str, float]]
                键为(question_id, skill_id)，值为包含错误率和计数的字典
                - error_rate: 错误率（0-1）
                - count: 该(问题,技能)对的出现次数
        """
        data = self.data_src.get_sequence_data()

        # 统计每个(问题, 技能)对的错误率
        error_patterns = {}

        grouped = data.groupby(["question", "skill"])["label"].agg(["mean", "count"])

        for (question_id, skill_id), row in grouped.iterrows():
            error_rate = 1 - row["mean"]  # 错误率 = 1 - 正确率
            count = row["count"]
            error_patterns[(int(question_id), int(skill_id))] = {
                "error_rate": float(error_rate),
                "count": int(count),
            }

        return error_patterns

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

        data = self.data_src.get_question_data()

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
                    f"Unsupported value_type: {value_type}. "
                    f"Supported types: 'binary', 'count'"
                )

        return data_matrix


class GraphModelData(BaseModelData):
    r"""
    图数据基类
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    @abstractmethod
    def prepare_data(self, args):
        """
        准备图模型所需的数据
        """
        raise NotImplementedError("Subclasses should implement prepare_data method")

    def build_multiple_hypergraphs(
        self,
        edge_types: list[tuple[str, str, str]],
        vertex_type: str = None,
    ):
        """
        批量构建多个超图

        参数:
            edge_types: 边类型列表，每个元素为三元组 (顶点类型, 边关系名, 超边类型)
                       例如: [
                           ('question', 'has', 'skill'),
                           ('question', 'belongs_to', 'template'),
                           ('question', 'in', 'assignment')
                       ]
            vertex_type: 顶点类型（可选），默认使用每个edge_type的第一个元素

        返回:
            dict: 字典，键为超边类型名称，值为对应的超图对象
                 例如: {
                     'skill': skill_hypergraph,
                     'template': template_hypergraph,
                     'assignment': assignment_hypergraph
                 }

        示例:
            # 批量构建多个超图
            hypergraphs = model_data.build_multiple_hypergraphs([
                ('question', 'has', 'skill'),
                ('question', 'belongs_to', 'template'),
            ])

            skill_hg = hypergraphs['skill']
            template_hg = hypergraphs['template']
        """
        hypergraphs = {}

        for edge_type in edge_types:
            _, _, hyperedge_type = edge_type
            hypergraph = self.build_hyper_graph(edge_type, vertex_type=vertex_type)
            hypergraphs[hyperedge_type] = hypergraph

        return hypergraphs

    def build_sequence_data(self, max_seq_len: int, min_seq_len: int):
        from tqdm import tqdm
        import numpy as np

        data = self.data_src.get_sequence_data()
        num_users = self.data_src.get_metadata("num_users")

        # 构建用户答题序列
        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        # 构建用户ID序列
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        # 用户作答正确与否序列
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        # 序列掩码，用于区分是否存在作答数据
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)
        # 用户序列长度计数器，用于索引
        num_sequence = [0] * num_users

        for row in tqdm(
            data.itertuples(), total=data.shape[0], desc="Building user sequences"
        ):
            # 获取用户ID、问题ID和作答正确与否
            user_idx = row.user
            question_idx = row.question
            label = row.label
            # 如果当前用户的序列长度未达到最大长度，则添加数据
            if num_sequence[user_idx] < max_seq_len:
                user_sequence[user_idx, num_sequence[user_idx]] = question_idx
                user_id_sequence[user_idx, num_sequence[user_idx]] = user_idx
                user_response[user_idx, num_sequence[user_idx]] = label
                user_mask[user_idx, num_sequence[user_idx]] = 1
                # 自增对应的用户序列长度
                num_sequence[user_idx] += 1

        return user_sequence, user_response, user_mask, user_id_sequence

    def build_hetero_graph(
        self,
        edge_types: list[tuple[str, str, str]],
        edge_attrs: dict[tuple[str, str, str], list[str]] = None,
        directed: bool = False,
        node_features: dict[str, any] = None,
    ):
        """
        构建异构图，支持灵活配置节点类型和边类型

        参数:
            edge_types: 边类型列表，每个元素为三元组 (源节点类型, 边关系名, 目标节点类型)
                       例如: [('user', 'answers', 'question'), ('question', 'has', 'skill')]
            edge_attrs: 边属性字典，键为边类型三元组，值为属性列名列表
                       例如: {('user', 'answers', 'question'): ['label', 'order_id']}
                       默认为 None（不添加边属性）
            directed: 是否构建有向图，默认为 False（无向图）
            node_features: 节点特征字典，键为节点类型，值为特征张量或None
                          例如: {'question': question_difficulty_tensor}
                          默认使用节点ID作为特征

        返回:
            HeteroData: PyTorch Geometric 异构图对象

        示例:
            # 示例1: 构建问题-技能无向图
            graph = model_data.build_hetero_graph(
                edge_types=[('question', 'has', 'skill')],
                directed=False
            )

            # 示例2: 构建学生-问题和问题-技能的组合图
            graph = model_data.build_hetero_graph(
                edge_types=[
                    ('user', 'answers', 'question'),
                    ('question', 'has', 'skill')
                ],
                directed=False
            )

            # 示例3: 构建带边属性的图
            graph = model_data.build_hetero_graph(
                edge_types=[('user', 'answers', 'question')],
                edge_attrs={('user', 'answers', 'question'): ['label', 'order_id']},
                directed=True
            )
        """
        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import ToUndirected
        import numpy as np
        import torch
        from tqdm import tqdm

        if edge_attrs is None:
            edge_attrs = {}

        graph = HeteroData()

        # 收集所有需要的节点类型
        node_types = set()
        for src_type, _, dst_type in edge_types:
            node_types.add(src_type)
            node_types.add(dst_type)

        # 获取每种节点类型的数量
        node_counts = {}
        for node_type in node_types:
            # 尝试从元数据获取
            meta_key = f"num_{node_type}s"
            try:
                node_counts[node_type] = self.data_src.get_metadata(meta_key)
            except (KeyError, AttributeError):
                # 如果元数据中没有，从数据中计算
                data = self.data_src.get_question_data()
                if node_type in data.columns:
                    node_counts[node_type] = data[node_type].nunique()
                else:
                    raise ValueError(
                        f"Cannot determine node count for type '{node_type}'"
                    )

        # 设置节点数量和特征
        for node_type in node_types:
            graph[node_type].num_nodes = node_counts[node_type]

            # 设置节点特征
            if node_features and node_type in node_features:
                graph[node_type].x = node_features[node_type]
            else:
                # 默认使用节点ID作为特征
                graph[node_type].x = (
                    torch.arange(node_counts[node_type]).view(-1, 1).float()
                )

        # 为每种边类型构建边
        for edge_type in edge_types:
            src_type, relation, dst_type = edge_type

            # 使用 build_relationship_matrix 构建关联矩阵
            self.logger.info(
                f"Building relationship matrix for {src_type}-{relation}-{dst_type}"
            )
            rel_matrix = self.build_relationship_matrix(edge_type, value_type="binary")

            # 从关联矩阵中提取边索引
            # nonzero 返回非零元素的行列索引
            src_indices, dst_indices = np.nonzero(rel_matrix)

            if len(src_indices) == 0:
                self.logger.warning(f"No edges found for {edge_type}")
                continue

            # 转换为 PyTorch 张量
            edge_index = torch.tensor(
                np.vstack([src_indices, dst_indices]), dtype=torch.long
            ).contiguous()

            # 添加边索引到图
            graph[src_type, relation, dst_type].edge_index = edge_index

            # 处理边属性
            attr_cols = edge_attrs.get(edge_type, [])
            if attr_cols:
                # 需要从原始数据中提取边属性
                data = self.data_src.get_sequence_data()
                src_col = src_type
                dst_col = dst_type

                # 检查列是否存在
                if src_col not in data.columns or dst_col not in data.columns:
                    self.logger.warning(
                        f"Columns {src_col} or {dst_col} not found. Skipping edge attributes."
                    )
                    continue

                # 构建边到属性的映射
                edge_attr_dict = {}
                cols_to_select = [src_col, dst_col] + attr_cols

                for row in tqdm(
                    data[cols_to_select].itertuples(index=False),
                    total=len(data),
                    desc=f"Extracting edge attributes for {src_type}-{relation}-{dst_type}",
                ):
                    src_id = int(getattr(row, src_col))
                    dst_id = int(getattr(row, dst_col))
                    edge_key = (src_id, dst_id)

                    # 如果边已存在，更新属性（取最后一次出现）
                    edge_attr_dict[edge_key] = {
                        attr: getattr(row, attr) for attr in attr_cols
                    }

                # 按照 edge_index 的顺序提取属性值
                for attr in attr_cols:
                    attr_values = []
                    for i in range(edge_index.shape[1]):
                        src_id = int(edge_index[0, i].item())
                        dst_id = int(edge_index[1, i].item())
                        edge_key = (src_id, dst_id)

                        if edge_key in edge_attr_dict:
                            attr_values.append(edge_attr_dict[edge_key][attr])
                        else:
                            # 如果找不到属性，使用默认值 0
                            attr_values.append(0.0)

                    attr_tensor = torch.tensor(attr_values, dtype=torch.float32)
                    # 边属性存储为 edge_attr_<attr_name>
                    setattr(
                        graph[src_type, relation, dst_type],
                        f"edge_attr_{attr}",
                        attr_tensor,
                    )

        # 如果需要无向图，应用转换
        if not directed:
            graph = ToUndirected()(graph)

        return graph

    def build_hyper_graph(
        self,
        edge_type: tuple[str, str, str],
        vertex_type: str = None,
    ):
        """
        构建超图，支持灵活配置超边类型

        超图定义：
            - 顶点(vertices): 通常是问题(question)节点
            - 超边(hyperedges): 每个超边连接一组相关的顶点
              例如：具有相同知识点/技能的题目、属于相同模板的题目等

        参数:
            edge_type: 边类型三元组 (顶点类型, 边关系名, 超边类型)
                      例如: ('question', 'has', 'skill') - 知识点超边
                           ('question', 'belongs_to', 'template') - 模板超边
                           ('question', 'in', 'assignment') - 作业超边
            vertex_type: 顶点类型（可选），默认使用 edge_type 的第一个元素
                        通常是 'question'

        返回:
            dhg.Hypergraph: DHG框架的超图对象

        工作原理:
            1. 从数据中提取顶点-超边的关联关系
            2. 将相同超边类型(如相同skill_id)的所有顶点分组
            3. 每组顶点形成一个超边
            4. 使用DHG框架创建超图对象

        示例:
            # 构建知识点超图：每个知识点连接包含它的所有题目
            skill_hg = model_data.build_hypergraph(
                ('question', 'has', 'skill')
            )

            # 构建模板超图：每个模板连接属于它的所有题目
            template_hg = model_data.build_hypergraph(
                ('question', 'belongs_to', 'template')
            )

            # 构建作业超图：每个作业连接其中的所有题目
            assignment_hg = model_data.build_hypergraph(
                ('question', 'in', 'assignment')
            )
        """
        from dhg import Hypergraph
        from tqdm import tqdm
        import numpy as np

        vertex_node_type, relation, hyperedge_node_type = edge_type

        # 如果未指定顶点类型，使用边类型的第一个元素
        if vertex_type is None:
            vertex_type = vertex_node_type

        # 获取关联矩阵
        H = self.build_relationship_matrix(edge_type, value_type="binary")

        # 获取顶点数量
        num_vertices = H.shape[0]

        # 将关联矩阵转换为超边列表
        rows, cols = np.nonzero(H)

        # 按列（超边类型）分组，每个超边类型对应一个超边
        # 使用字典收集每个超边包含的顶点
        edge_dict = {}
        for vertex_idx, hyperedge_idx in tqdm(
            zip(rows, cols),
            total=len(rows),
            desc=f"Building {hyperedge_node_type} hyperedges",
        ):
            if hyperedge_idx not in edge_dict:
                edge_dict[hyperedge_idx] = []
            edge_dict[hyperedge_idx].append(int(vertex_idx))

        # 转换为超边列表（过滤空超边）
        e_list = [vertices for vertices in edge_dict.values() if len(vertices) > 0]

        # 处理没有超边的情况
        if len(e_list) == 0:
            self.logger.warning(
                f"No hyperedges found for {edge_type}. Creating self-loop hypergraph."
            )
            # 创建自环超图：每个顶点自成一个超边
            e_list = [[i] for i in range(num_vertices)]

        # 使用 DHG 框架创建超图
        hypergraph = Hypergraph(num_v=num_vertices, e_list=e_list)

        self.logger.info(f"{hyperedge_node_type.capitalize()} Hypergraph constructed:")
        self.logger.info(f"  - Number of vertices ({vertex_type}s): {hypergraph.num_v}")
        self.logger.info(
            f"  - Number of hyperedges ({hyperedge_node_type}s): {hypergraph.num_e}"
        )

        return hypergraph
