from abc import abstractmethod

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import BaseModelData


class QuestionModelData(BaseModelData):
    """
    问题级模型数据基类
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    @abstractmethod
    def prepare_data(self, args):
        """
        准备问题级模型所需的数据
        """
        raise NotImplementedError("Subclasses should implement prepare_data method")

    def build_sequence_data(self):
        """
        构建用户答题序列

        参数:
            max_seq_len: 最大序列长度

        说明：
            - 从磁盘加载切分后的序列数据
            - 构建序列数组
        """
        import numpy as np

        self.logger.info("Building response sequences from split data...")

        # 加载切分后的序列数据
        data = self.data_src.get_split_sequence_data()
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        num_users = data["user"].nunique()

        # 构建序列数组
        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)

        user_indices = data["user"].values
        seq_positions = data["seq_pos"].values

        user_sequence[user_indices, seq_positions] = data["question"].values
        user_id_sequence[user_indices, seq_positions] = user_indices
        user_response[user_indices, seq_positions] = data["label"].values
        user_mask[user_indices, seq_positions] = 1

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
                       例如: {('user', 'answers', 'question'): ['label', 'timestamp']}
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
                edge_attrs={('user', 'answers', 'question'): ['label', 'timestamp']},
                directed=True
            )
        """
        import numpy as np
        import torch
        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import ToUndirected
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
        import numpy as np
        from dhg import Hypergraph
        from tqdm import tqdm

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
