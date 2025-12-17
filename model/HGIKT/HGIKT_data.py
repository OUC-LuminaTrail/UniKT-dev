import torch
from utils.data_process import DataSource
from utils.net_data import GraphModelData
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


class HGIKTModelData(GraphModelData):
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

        # 构建问题-技能关联矩阵，并转换为torch张量
        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

        # 构建难度加权超图
        skill_hypergraph = self.build_difficulty_weighted_hypergraph(
            ("question", "has", "skill"),
            num_difficulty_clusters=getattr(args, "num_difficulty_clusters", 3),
        )

        print(
            f"  - Primary hypergraph: Difficulty-weighted hypergraph ({skill_hypergraph.num_e} hyperedges)"
        )

        # 构建异构图
        hetero_graph = self.build_hetero_graph(
            [
                # 问题技能图
                (
                    "question",
                    "has",
                    "skill",
                ),
                # 技能作业图
                (
                    "skill",
                    "related_to",
                    "assignment",
                ),
                # 题目模板图
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
            train_data, val_data = self.split_kfold_data(
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

        # 返回数据
        return_data = {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "skill_hypergraph": skill_hypergraph,
            "hetero_graph": hetero_graph,
            "question_skill_matrix": question_skill_matrix,
        }

        return return_data

    def build_difficulty_weighted_hypergraph(
        self,
        edge_type: tuple[str, str, str],
        vertex_type: str = None,
        num_difficulty_clusters: int = 3,
    ):
        """
        构建基于难度加权的超图

        核心思想：
        1. 将同一技能下的题目按难度聚类
        2. 每个难度簇形成一个子超边
        3. 超边权重反映簇内题目的平均难度

        参数:
            edge_type: 边类型三元组 (顶点类型, 边关系名, 超边类型)
                      例如: ('question', 'has', 'skill')
            vertex_type: 顶点类型（可选），默认使用 edge_type 的第一个元素
            num_difficulty_clusters: 难度聚类数量，默认为3

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
        difficulty_scores = self.calculate_question_difficulty()

        # 获取关联矩阵
        H = self.build_relationship_matrix(edge_type, value_type="binary")
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

                # 存储每个簇的信息以便排序
                current_skill_clusters = []

                for cluster_id in range(kmeans.n_clusters):
                    cluster_vertices = [
                        vertices[i]
                        for i in range(len(vertices))
                        if cluster_labels[i] == cluster_id
                    ]

                    if len(cluster_vertices) == 0:
                        continue

                    # 计算簇内题目的平均难度作为边权
                    cluster_difficulties = difficulties[cluster_labels == cluster_id]
                    avg_difficulty = float(np.mean(cluster_difficulties))

                    current_skill_clusters.append(
                        {
                            "vertices": cluster_vertices,
                            "weight": avg_difficulty,
                            "avg_difficulty": avg_difficulty,
                        }
                    )

                # 按照平均难度对簇进行排序
                current_skill_clusters.sort(key=lambda x: x["avg_difficulty"])

                # 构建超边和权重
                for cluster in current_skill_clusters:
                    e_list.append(cluster["vertices"])
                    edge_weights.append(cluster["weight"])

            except Exception:
                e_list.append(vertices)
                edge_weights.append(1.0)

        # 处理空超边情况
        if len(e_list) == 0:
            print("Warning: No hyperedges found. Creating self-loop hypergraph.")
            e_list = [[i] for i in range(num_vertices)]
            edge_weights = [1.0] * num_vertices

        # 确保超边列表和权重列表长度一致
        assert len(e_list) == len(edge_weights), (
            f"Mismatch: {len(e_list)} edges but {len(edge_weights)} weights"
        )

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
