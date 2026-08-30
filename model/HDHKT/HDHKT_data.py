from typing import Any

import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from model.HDHKT.skill_index import build_skill_index_table
from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class HDHKTDataset(Dataset):
    def __init__(self, sequences, responses, masks, user_ids=None):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.responses = torch.as_tensor(responses, dtype=torch.long)
        self.masks = torch.as_tensor(masks, dtype=torch.bool)
        self.user_ids = (
            torch.as_tensor(user_ids, dtype=torch.long)
            if user_ids is not None
            else None
        )

    def __getitem__(self, index):
        if self.user_ids is not None:
            return (
                self.user_ids[index],
                self.sequences[index],
                self.responses[index],
                self.masks[index],
            )
        return (
            self.sequences[index],
            self.responses[index],
            self.masks[index],
        )

    def __len__(self):
        return len(self.sequences)


class HDHKTModelData(QuestionModelData):
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any, with_user_ids: bool = False):
        r"""
        准备HDHKT模型所需的数据

        When ``with_user_ids=True``, each Dataset additionally returns the
        user id per sample (for case analysis); the training path keeps
        the 3-tuple layout unchanged.
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        user_sequence, user_response, user_mask, user_id_sequence = (
            self.load_sequence_data()
        )

        question_skill_matrix = torch.from_numpy(
            self.build_relationship_matrix(("question", "has", "skill"))
        ).float()

        # Precomputed per-question related-skill id table
        num_skills = self.data_src.get_metadata("num_skills")
        skill_ids_per_question = build_skill_index_table(
            question_skill_matrix, padding_index=num_skills
        )

        if rc.model.use_hypergraph:
            skill_hypergraph = self.build_difficulty_weighted_hypergraph(
                ("question", "has", "skill"),
                num_difficulty_clusters=rc.model.num_difficulty_clusters,
                use_difficulty_clustering=rc.model.use_difficulty_clustering,
                use_edge_weights=rc.model.use_edge_weights,
            )

            logger.debug(
                f"  - Primary hypergraph: Difficulty-weighted hypergraph ({skill_hypergraph.num_e} hyperedges)"
            )
        else:
            skill_hypergraph = None
            logger.debug("  - Hypergraph disabled (ablation)")

        if rc.model.use_hetero_graph:
            edge_types = [
                (
                    "question",
                    "has",
                    "skill",
                ),
            ]
            if rc.model.use_sa_relation:
                edge_types.append(
                    (
                        "skill",
                        "related_to",
                        "assignment",
                    )
                )
            if rc.model.use_qt_relation:
                edge_types.append(
                    (
                        "question",
                        "belongs_to",
                        "template",
                    )
                )
            hetero_graph = self.build_hetero_graph(edge_types)
        else:
            hetero_graph = None
            logger.debug("  - Heterogeneous graph disabled (ablation)")

        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            logger.info(
                f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
            )
            if with_user_ids:
                train_data, val_data, test_data = self.split_kfold_data(
                    user_sequence,
                    user_response,
                    user_mask,
                    user_id_sequence,
                    fold_idx=fold_idx,
                )
            else:
                train_data, val_data, test_data = self.split_kfold_data(
                    user_sequence, user_response, user_mask, fold_idx=fold_idx
                )
        else:
            raise ValueError(
                "K-fold cross-validation is required for HDHKT. Please specify a valid fold index."
            )

        train_dataset = HDHKTDataset(
            train_data[0],
            train_data[1],
            train_data[2],
            user_ids=train_data[3] if with_user_ids else None,
        )
        val_dataset = HDHKTDataset(
            val_data[0],
            val_data[1],
            val_data[2],
            user_ids=val_data[3] if with_user_ids else None,
        )
        test_dataset = HDHKTDataset(
            test_data[0],
            test_data[1],
            test_data[2],
            user_ids=test_data[3] if with_user_ids else None,
        )

        return_data = {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "skill_hypergraph": skill_hypergraph,
            "hetero_graph": hetero_graph,
            "question_skill_matrix": question_skill_matrix,
            "skill_ids_per_question": skill_ids_per_question,
        }

        return return_data

    def build_difficulty_weighted_hypergraph(
        self,
        edge_type: tuple[str, str, str],
        vertex_type: str = None,
        num_difficulty_clusters: int = 3,
        use_difficulty_clustering: bool = True,
        use_edge_weights: bool = True,
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
            use_difficulty_clustering: 是否进行难度感知聚类（消融开关）；
                关闭时每个技能退化为一条普通超边，权重取该技能下题目平均难度
            use_edge_weights: 是否启用超边难度权重（消融开关）；
                关闭时所有超边权重为 1

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
        import numpy as np
        from dhg import Hypergraph
        from sklearn.cluster import KMeans
        from tqdm import tqdm

        vertex_node_type, _, hyperedge_node_type = edge_type

        if vertex_type is None:
            vertex_type = vertex_node_type

        difficulty_scores = self.calculate_question_difficulty()

        H = self.build_relationship_matrix(edge_type, value_type="binary")
        num_vertices = H.shape[0]

        rows, cols = np.nonzero(H)
        edge_dict = {}
        for vertex_idx, hyperedge_idx in zip(rows, cols):
            if hyperedge_idx not in edge_dict:
                edge_dict[hyperedge_idx] = []
            edge_dict[hyperedge_idx].append(int(vertex_idx))

        # Cluster questions within each hyperedge (e.g. skill) by difficulty
        e_list = []
        edge_weights = []

        logger.debug(
            f"Building difficulty-weighted {hyperedge_node_type} hypergraph..."
        )

        if not use_difficulty_clustering:
            # Ablation w/o clustering: one plain hyperedge per skill, weighted by
            # the mean difficulty of its questions (all-1 when weights are off).
            e_list = [vertices for vertices in edge_dict.values() if len(vertices) > 0]
            if use_edge_weights:
                edge_weights = [
                    float(np.mean([difficulty_scores.get(v, 0.5) for v in vertices]))
                    for vertices in e_list
                ]
            else:
                edge_weights = [1.0] * len(e_list)
        else:
            for hyperedge_idx, vertices in tqdm(
                edge_dict.items(),
                desc=f"Clustering {hyperedge_node_type} by difficulty",
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
                        edge_weights.append(1.0)
                    continue

                try:
                    kmeans = KMeans(
                        n_clusters=min(num_difficulty_clusters, len(vertices)),
                        random_state=42,
                        n_init=10,
                    )
                    cluster_labels = kmeans.fit_predict(difficulties)

                    # Collect cluster info for sorting by difficulty
                    current_skill_clusters = []

                    for cluster_id in range(kmeans.n_clusters):
                        cluster_vertices = [
                            vertices[i]
                            for i in range(len(vertices))
                            if cluster_labels[i] == cluster_id
                        ]

                        if len(cluster_vertices) == 0:
                            continue

                        # Edge weight = mean difficulty of questions in this cluster
                        cluster_difficulties = difficulties[
                            cluster_labels == cluster_id
                        ]
                        avg_difficulty = float(np.mean(cluster_difficulties))

                        current_skill_clusters.append(
                            {
                                "vertices": cluster_vertices,
                                "weight": avg_difficulty,
                                "avg_difficulty": avg_difficulty,
                            }
                        )

                    current_skill_clusters.sort(key=lambda x: x["avg_difficulty"])

                    for cluster in current_skill_clusters:
                        e_list.append(cluster["vertices"])
                        edge_weights.append(cluster["weight"])

                except Exception:
                    e_list.append(vertices)
                    edge_weights.append(1.0)

        if len(e_list) == 0:
            logger.warning("No hyperedges found. Creating self-loop hypergraph.")
            e_list = [[i] for i in range(num_vertices)]
            edge_weights = [1.0] * num_vertices

        assert len(e_list) == len(edge_weights), (
            f"Mismatch: {len(e_list)} edges but {len(edge_weights)} weights"
        )

        hypergraph = Hypergraph(
            num_v=num_vertices,
            e_list=e_list,
            e_weight=edge_weights if use_edge_weights else None,
        )

        logger.debug(
            f"Difficulty-weighted {hyperedge_node_type.capitalize()} Hypergraph constructed:"
        )
        logger.debug(f"  - Number of vertices ({vertex_type}s): {hypergraph.num_v}")
        logger.debug(
            f"  - Number of hyperedges (difficulty clusters): {hypergraph.num_e}"
        )

        return hypergraph
