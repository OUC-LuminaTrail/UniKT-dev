"""
ABKT 模型数据准备

负责将 kt-exp-graph 框架的标准数据格式转换为 ABKT 所需的格式。
"""

import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class ABKTModelData(QuestionModelData):
    """
    ABKT 模型数据准备类

    负责:
    1. 构建 Q-Matrix (题目-技能关联矩阵)
    2. 将框架数据转换为 ABKT 所需的序列格式
    3. 构建学生-题目二部图的归一化邻接矩阵
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    def prepare_data(self, args) -> dict:
        """
        准备 ABKT 所需的全部数据。

        参数:
            args: 命令行参数。

        返回:
            dict: {
                'Q_matrix': Tensor [num_items, num_skills],
                'train_sequences': dict {user_id: [[item_seq], [correct_seq]]},
                'test_triplets': list [[user, item, correct], ...],
                'train_users': list [user_id, ...],
                'test_users': list [user_id, ...],
                'num_users': int,
                'num_items': int,
                'num_skills': int,
                'num_records': int,
            }
        """
        fold_idx = args.fold if hasattr(args, "fold") and args.fold >= 0 else None

        num_users = self.data_src.get_metadata("num_users")
        num_items = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        self.logger.info(
            f"Data statistics: users={num_users}, items={num_items}, skills={num_skills}"
        )

        self.logger.info("Building Q-Matrix...")
        Q_matrix = self._build_q_matrix(num_items, num_skills)

        self.logger.info(f"Building ABKT sequences (fold={fold_idx})...")
        train_sequences, test_triplets, train_users, test_users, num_records = (
            self._build_abkt_sequences(fold_idx)
        )

        self.logger.info(
            f"Train users: {len(train_users)}, Test users: {len(test_users)}"
        )
        self.logger.info(
            f"Test triplets: {len(test_triplets)}, Total records: {num_records}"
        )

        return {
            "Q_matrix": Q_matrix,
            "train_sequences": train_sequences,
            "test_triplets": test_triplets,
            "train_users": train_users,
            "test_users": test_users,
            "num_users": num_users,
            "num_items": num_items,
            "num_skills": num_skills,
            "num_records": num_records,
        }

    def _build_q_matrix(self, num_items: int, num_skills: int) -> torch.Tensor:
        """构建 Q-Matrix（题目-技能关联矩阵），返回 [num_items, num_skills]。"""
        q_matrix_np = self.build_relationship_matrix(
            edge_type=("question", "has", "skill"),
            value_type="binary",
        )

        Q_matrix = torch.from_numpy(q_matrix_np).float()

        self.logger.info(f"Q-Matrix shape: {Q_matrix.shape}")
        self.logger.info(f"Q-Matrix density: {Q_matrix.sum() / Q_matrix.numel():.4f}")

        return Q_matrix

    def _build_abkt_sequences(
        self, fold_idx: int | None = None
    ) -> tuple[dict, list, list, list, int]:
        """将框架数据转换为 ABKT 的全数据 next-item holdout 划分。

        ABKT 是直推式模型（协同矩阵分解 + 学生-题目二部图），需要利用整个数据集：
        每个用户既贡献训练前缀（``seq[:-1]``）、又贡献一个测试三元组（最后一项）。
        这也是 ``_compute_boosting_residuals`` 的前提——测试用户的最终知识状态由其
        训练前缀计算得到，因此测试用户必须同时出现在训练集中，不能按用户划分 train/test。

        ``fold_idx`` 仅为兼容调用签名而保留，ABKT 不使用框架的 fold 划分。
        """
        if fold_idx is not None:
            self.logger.warning(
                "ABKT uses full-data next-item holdout (covers all users); "
                f"framework fold={fold_idx} is ignored."
            )

        data = self.data_src.get_sequence_data()

        # 按用户聚合序列
        all_sequences = {}
        for row in tqdm(
            data.iter_rows(named=True),
            total=len(data),
            desc="Aggregating user sequences",
        ):
            user_id = row["user"]
            item_id = row["question"]
            correct = row["label"]

            if user_id not in all_sequences:
                all_sequences[user_id] = {"items": [], "corrects": []}
            all_sequences[user_id]["items"].append(item_id)
            all_sequences[user_id]["corrects"].append(correct)

        train_sequences = {}
        test_triplets = []
        train_users = []
        test_users_list = []
        num_records = 0

        for user_id, seq_data in tqdm(
            all_sequences.items(), desc="Building train/test split"
        ):
            items = seq_data["items"]
            corrects = seq_data["corrects"]
            seq_len = len(items)
            train_len = seq_len - 1

            if train_len > 0:
                train_sequences[user_id] = [
                    [items[:train_len]],
                    [corrects[:train_len]],
                ]
                train_users.append(user_id)
                num_records += seq_len

            test_triplets.append([user_id, items[-1], corrects[-1]])
            if user_id not in test_users_list:
                test_users_list.append(user_id)

        return train_sequences, test_triplets, train_users, test_users_list, num_records

    def build_normalized_adj_matrix(
        self,
        train_sequences: dict,
        num_users: int,
        num_items: int,
        symmetric: bool = True,
    ) -> torch.sparse.Tensor:
        """
        构建学生-题目二部图的归一化邻接矩阵

        邻接矩阵结构:
        A = [[0, R], [R^T, 0]]
        其中 R 是用户-题目交互矩阵

        归一化方式:
        - symmetric=True: D^{-1/2} A D^{-1/2} (对称归一化)
        - symmetric=False: D^{-1} A (行归一化)

        参数:
            train_sequences: 训练序列字典
            num_users: 用户数量
            num_items: 题目数量
            symmetric: 是否使用对称归一化

        返回:
            归一化的稀疏邻接矩阵 [num_users + num_items, num_users + num_items]
        """
        self.logger.info("Building normalized adjacency matrix...")

        rows = []
        cols = []

        for user_id, seq_data in tqdm(
            train_sequences.items(),
            desc="Collecting edges",
        ):
            items = seq_data[0][0]  # [[item_ids]] -> [item_ids]
            for item_id in items:
                # 用户 -> 题目，题目 -> 用户（无向图）
                rows.append(user_id)
                cols.append(item_id + num_users)
                rows.append(item_id + num_users)
                cols.append(user_id)

        data = np.ones(len(rows))
        total_nodes = num_users + num_items
        adj_matrix = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(total_nodes, total_nodes),
        ).tocsr()

        # 添加自环
        adj_matrix = adj_matrix + sparse.eye(total_nodes)

        # 归一化
        self.logger.info("Normalizing adjacency matrix...")
        if symmetric:
            # 对称归一化: D^{-1/2} A D^{-1/2}
            d = sparse.diags(
                np.power(np.array(adj_matrix.sum(1)).flatten() + 1e-8, -0.5)
            )
            adj_norm = adj_matrix.dot(d).T.dot(d).tocoo()
        else:
            # 行归一化: D^{-1} A
            d = sparse.diags(np.power(np.array(adj_matrix.sum(1)).flatten() + 1e-8, -1))
            adj_norm = d.dot(adj_matrix).tocoo()

        indices = torch.tensor(
            np.vstack([adj_norm.row, adj_norm.col]),
            dtype=torch.long,
        )
        values = torch.tensor(adj_norm.data, dtype=torch.float32)
        shape = adj_norm.shape

        adj_sparse = torch.sparse_coo_tensor(indices, values, shape).coalesce()

        self.logger.info(f"Adjacency matrix shape: {shape}")
        self.logger.info(f"Number of edges: {len(values)}")

        return adj_sparse


__all__ = ["ABKTModelData"]
