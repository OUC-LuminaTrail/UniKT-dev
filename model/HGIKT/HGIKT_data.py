import torch
from utility.data_process.data_utility import DataSource, ModelData
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from typing_extensions import override


class HGIKTDataset(Dataset):
    def __init__(self, sequences, responses, masks):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.masks[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.sequences)


class HGIKTModelData(ModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        r"""
        准备HGIKT模型所需的数据
        """
        fold_idx = args.fold if args.fold > 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        max_seq_len = self.data_src.get_metadata("max_seq_len")
        min_seq_len = self.data_src.get_metadata("min_seq_len")

        # 构建用户答题序列
        user_sequence, user_response, user_mask, _ = self.build_sequence_data(
            max_seq_len, min_seq_len
        )

        # 初始化变量
        pos_hypergraph = None
        neg_hypergraph = None

        # 构建主超图（难度加权超图）
        # 使用难度加权超图作为主超图
        skill_hypergraph = self.build_difficulty_weighted_hypergraph(
            ("question", "has", "skill"),
            num_difficulty_clusters=getattr(args, "num_difficulty_clusters", 3),
        )

        # 构建正负超图
        # 构建对比学习超图（正负超边）
        pos_hypergraph, neg_hypergraph = self.build_contrastive_hypergraph(
            ("question", "has", "skill"),
            easy_threshold=getattr(args, "contrastive_easy_threshold", 0.3),
            hard_threshold=getattr(args, "contrastive_hard_threshold", 0.6),
            min_samples=getattr(args, "contrastive_min_samples", 5),
        )

        print(f"  - Primary hypergraph: Difficulty-weighted hypergraph ({skill_hypergraph.num_e} hyperedges)")
        print(
            f"  - Contrastive hypergraph: {pos_hypergraph.num_e} positive hyperedges, {neg_hypergraph.num_e} negative hyperedges"
        )

        # 构建异构图
        hetero_graph = self.build_hetero_graph(
            [
                (
                    "question",
                    "has",
                    "skill",
                ),
                (
                    "skill",
                    "related_to",
                    "assignment",
                ),
                (
                    "assignment",
                    "contains",
                    "question",
                ),
                (
                    "question",
                    "belongs_to",
                    "template",
                ),
            ]
        )

        # 划分训练集和验证集
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            print(f"Using K-fold cross-validation: fold {fold_idx}/{kfold_n_splits}")
            train_data, val_data = self.get_kfold_split_data(
                user_sequence, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            train_data, val_data = self.split_data(
                user_sequence, user_response, user_mask
            )

        # 构建模型数据集
        train_dataset = HGIKTDataset(train_data[0], train_data[1], train_data[2])
        val_dataset = HGIKTDataset(val_data[0], val_data[1], val_data[2])

        # 构建数据加载器
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        # 返回数据时包含边权重信息和对比超图
        return_data = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "skill_hypergraph": skill_hypergraph,
            "hetero_graph": hetero_graph,
            "pos_hypergraph": pos_hypergraph,
            "neg_hypergraph": neg_hypergraph,
        }

        return return_data

    def build_contrastive_hypergraph(
        self,
        edge_type: tuple[str, str, str],
        vertex_type: str = None,
        easy_threshold: float = 0.3,
        hard_threshold: float = 0.6,
        min_samples: int = 5,
    ):
        """
        构建对比学习超图：区分"正确超边"和"易错超边"

        核心思想：
        - 正超边（positive hyperedges）：连接学生易于正确回答的同技能题目
        - 负超边（negative hyperedges）：连接学生容易出错的同技能题目
        - 通过对比损失最大化正负超边特征的区分度

        参数:
            edge_type: 边类型 (source_type, relation, target_type)
                例如：("question", "has", "skill")
            vertex_type: 超图顶点类型（默认为source_type）
            easy_threshold: 简单题目的错误率阈值（低于此值视为简单）
            hard_threshold: 困难题目的错误率阈值（高于此值视为困难）
            min_samples: 每个(问题,技能)对的最小样本数阈值（用于过滤噪声）

        返回:
            positive_hg: dhg.Hypergraph 正超图（简单题目）
            negative_hg: dhg.Hypergraph 负超图（困难题目）
        """
        import dhg
        import numpy as np
        from tqdm import tqdm

        source_type, _, target_type = edge_type

        if vertex_type is None:
            vertex_type = source_type

        if vertex_type != source_type:
            raise ValueError(
                f"vertex_type ({vertex_type}) must match the source type ({source_type}) of edge_type"
            )

        # 获取节点数量
        num_vertices = self.data_src.get_metadata(f"num_{source_type}s")
        num_targets = self.data_src.get_metadata(f"num_{target_type}s")

        print(
            f"Building contrastive hypergraph: {num_vertices} {vertex_type}s, {num_targets} {target_type}s"
        )
        print(f"Thresholds: Easy < {easy_threshold}, Hard > {hard_threshold}")

        # 计算(问题, 技能)对的错误率
        error_patterns = self.calculate_question_error_rate()

        # 为每个技能构建正负超边
        pos_hyperedge_list = []
        neg_hyperedge_list = []
        pos_edge_weights = []
        neg_edge_weights = []

        data = self.data_src.get_processed_data()
        target_groups = data.groupby(target_type)[source_type].apply(
            lambda x: list(set(x))
        )

        for target_id, source_list in tqdm(
            target_groups.items(), desc=f"构建对比超边 ({target_type})"
        ):
            # 过滤出该技能下的题目及其错误率
            easy_questions = []
            hard_questions = []

            for question_id in source_list:
                key = (int(question_id), int(target_id))
                if key not in error_patterns:
                    continue

                pattern = error_patterns[key]
                if pattern["count"] < min_samples:
                    continue  # 过滤样本数不足的题目

                error_rate = pattern["error_rate"]
                if error_rate < easy_threshold:
                    easy_questions.append(int(question_id))
                elif error_rate > hard_threshold:
                    hard_questions.append(int(question_id))

            # 只保留至少有2个题目的超边（单个题目无法形成超边）
            if len(easy_questions) >= 2:
                pos_hyperedge_list.append(easy_questions)
                # 计算正超边权重：基于题目之间的错误率一致性
                error_rates = [
                    error_patterns[(q, int(target_id))]["error_rate"]
                    for q in easy_questions
                ]
                # 权重 = 1 - 错误率标准差（越一致权重越高）
                weight = 1.0 - np.std(error_rates)
                pos_edge_weights.append(max(0.1, weight))  # 确保权重至少为0.1

            if len(hard_questions) >= 2:
                neg_hyperedge_list.append(hard_questions)
                # 计算负超边权重
                error_rates = [
                    error_patterns[(q, int(target_id))]["error_rate"]
                    for q in hard_questions
                ]
                weight = 1.0 - np.std(error_rates)
                neg_edge_weights.append(max(0.1, weight))

        # 创建正负超图
        if len(pos_hyperedge_list) == 0:
            print(f"Warning: Not enough easy-question hyperedges found (threshold={easy_threshold})")
            # 创建空超图
            positive_hg = dhg.Hypergraph(num_vertices, [])
            pos_edge_weights = []
        else:
            positive_hg = dhg.Hypergraph(
                num_vertices, pos_hyperedge_list, e_weight=pos_edge_weights
            )

        if len(neg_hyperedge_list) == 0:
            print(f"Warning: Not enough hard-question hyperedges found (threshold={hard_threshold})")
            # 创建空超图
            negative_hg = dhg.Hypergraph(num_vertices, [])
            neg_edge_weights = []
        else:
            negative_hg = dhg.Hypergraph(
                num_vertices, neg_hyperedge_list, e_weight=neg_edge_weights
            )

        print(
            f"Contrastive hypergraph construction finished: positive_hyperedges={len(pos_hyperedge_list)}, negative_hyperedges={len(neg_hyperedge_list)}"
        )

        return positive_hg, negative_hg

    def build_difficulty_weighted_hypergraph(
        self,
        edge_type: tuple[str, str, str],
        vertex_type: str = None,
        num_difficulty_clusters: int = 3,
    ):
        """
        构建基于难度加权的超图

        核心思想：
        1. 将同一技能下的题目按难度聚类（简单/中等/困难）
        2. 每个难度簇形成一个子超边
        3. 超边权重反映簇内题目的难度一致性

        参数:
            edge_type: 边类型三元组 (顶点类型, 边关系名, 超边类型)
                      例如: ('question', 'has', 'skill')
            vertex_type: 顶点类型（可选），默认使用 edge_type 的第一个元素
            num_difficulty_clusters: 难度聚类数量，默认为3（简单/中等/困难）

        返回:
            tuple: (hypergraph, edge_weights)
                - hypergraph: DHG框架的超图对象
                - edge_weights: 超边权重列表，与超图的超边顺序对应

        示例:
            # 构建难度加权的技能超图
            hg, weights = model_data.build_difficulty_weighted_hypergraph(
                ('question', 'has', 'skill'),
                num_difficulty_clusters=3
            )
        """
        from dhg import Hypergraph
        from sklearn.cluster import KMeans
        from tqdm import tqdm
        import numpy as np

        vertex_node_type, _, hyperedge_node_type = edge_type

        if vertex_type is None:
            vertex_type = vertex_node_type

        # 获取数据和难度分数
        data = self.data_src.get_processed_data()
        difficulty_scores = self.calculate_question_difficulty()

        # 获取关联矩阵
        H = self.build_data_matrix(edge_type, value_type="binary")
        num_vertices = H.shape[0]

        # 将关联矩阵转换为超边字典
        rows, cols = np.nonzero(H)
        edge_dict = {}
        for vertex_idx, hyperedge_idx in zip(rows, cols):
            if hyperedge_idx not in edge_dict:
                edge_dict[hyperedge_idx] = []
            edge_dict[hyperedge_idx].append(int(vertex_idx))

        # 为每个超边（如技能）内的题目按难度聚类
        e_list = []
        edge_weights = []

        print(f"Building difficulty-weighted {hyperedge_node_type} hypergraph...")
        for hyperedge_idx, vertices in tqdm(
            edge_dict.items(), desc=f"Clustering {hyperedge_node_type} by difficulty"
        ):
            if len(vertices) == 0:
                continue

            # 获取这些题目的难度分数
            difficulties = np.array(
                [difficulty_scores.get(v, 0.5) for v in vertices]
            ).reshape(-1, 1)

            # 如果题目数量少于聚类数，每个题目单独成簇
            if len(vertices) < num_difficulty_clusters:
                for v in vertices:
                    e_list.append([v])
                    edge_weights.append(1.0)  # 单题目超边权重为1
                continue

            # K-means聚类
            try:
                kmeans = KMeans(
                    n_clusters=min(num_difficulty_clusters, len(vertices)),
                    random_state=42,
                    n_init=10,
                )
                cluster_labels = kmeans.fit_predict(difficulties)

                # 为每个簇创建子超边
                for cluster_id in range(kmeans.n_clusters):
                    cluster_vertices = [
                        vertices[i]
                        for i in range(len(vertices))
                        if cluster_labels[i] == cluster_id
                    ]

                    if len(cluster_vertices) == 0:
                        continue

                    # 计算簇内难度一致性（方差越小，一致性越高，权重越大）
                    cluster_difficulties = difficulties[cluster_labels == cluster_id]
                    difficulty_variance = np.var(cluster_difficulties)
                    # 权重：一致性高的簇权重更大
                    # 使用指数衰减：variance越大，权重越小
                    weight = np.exp(-difficulty_variance * 5.0)
                    weight = max(0.1, min(1.0, weight))  # 限制在[0.1, 1.0]

                    e_list.append(cluster_vertices)
                    edge_weights.append(float(weight))

            except Exception as e:
                print(f"Warning: Clustering failed for hyperedge {hyperedge_idx}: {e}")
                # 如果聚类失败，将所有题目放在一个超边中
                e_list.append(vertices)
                edge_weights.append(1.0)

        # 处理空超边情况
        if len(e_list) == 0:
            print(f"Warning: No hyperedges found. Creating self-loop hypergraph.")
            e_list = [[i] for i in range(num_vertices)]
            edge_weights = [1.0] * num_vertices

        # 确保超边列表和权重列表长度一致
        assert len(e_list) == len(
            edge_weights
        ), f"Mismatch: {len(e_list)} edges but {len(edge_weights)} weights"

        # 创建超图
        hypergraph = Hypergraph(
            num_v=num_vertices, e_list=e_list, e_weight=edge_weights
        )

        print(
            f"Difficulty-weighted {hyperedge_node_type.capitalize()} Hypergraph constructed:"
        )
        print(f"  - Number of vertices ({vertex_type}s): {hypergraph.num_v}")
        print(f"  - Number of hyperedges (difficulty clusters): {hypergraph.num_e}")
        
        return hypergraph
