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
from utils.net_data import BaseModelData

logger = get_logger(__name__)


class ABKTModelData(BaseModelData):
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
        准备 ABKT 所需的全部数据

        参数:
            args: 命令行参数，需包含 fold 字段

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

        # 获取元数据
        num_users = self.data_src.get_metadata("num_users")
        num_items = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        self.logger.info(
            f"Data statistics: users={num_users}, items={num_items}, "
            f"skills={num_skills}"
        )

        # 1. 构建 Q-Matrix
        self.logger.info("Building Q-Matrix...")
        Q_matrix = self._build_q_matrix(num_items, num_skills)

        # 2. 构建训练序列和测试三元组
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
        """
        构建 Q-Matrix (题目-技能关联矩阵)

        使用框架提供的 build_relationship_matrix 方法

        返回:
            Q_matrix: Tensor [num_items, num_skills]
        """
        # 使用框架方法构建关系矩阵
        q_matrix_np = self.build_relationship_matrix(
            edge_type=("question", "has", "skill"),
            value_type="binary",
        )

        # 转换为 PyTorch 张量
        Q_matrix = torch.from_numpy(q_matrix_np).float()

        self.logger.info(f"Q-Matrix shape: {Q_matrix.shape}")
        self.logger.info(f"Q-Matrix density: {Q_matrix.sum() / Q_matrix.numel():.4f}")

        return Q_matrix

    def _build_abkt_sequences(
        self, fold_idx: int | None
    ) -> tuple[dict, list, list, list, int]:
        """
        将框架数据转换为 ABKT 所需的序列格式

        ABKT 的数据格式:
        - train_sequences: {user_id: [[item_ids], [corrects]]}
            每个用户的训练序列（去掉最后一题）
        - test_triplets: [[user, item, correct], ...]
            测试集三元组（每个测试用户的最后一题）

        参数:
            fold_idx: K-Fold 的 fold 索引，None 表示不使用 K-Fold

        返回:
            train_sequences: dict
            test_triplets: list
            train_users: list (训练序列中的用户ID列表)
            test_users: list (测试三元组中的用户ID列表)
            num_records: int (总记录数)
        """
        data = self.data_src.get_sequence_data()

        # 确定测试用户集合
        if fold_idx is not None and "fold" in data.columns:
            # K-Fold 模式：指定 fold 的用户作为测试集
            test_user_set = set(data[data["fold"] == fold_idx]["user"].unique())
            self.logger.info(f"Using K-Fold mode: fold {fold_idx} as test set")
        else:
            # 非 K-Fold 模式：所有用户都参与训练，最后一题作为测试
            test_user_set = set(data["user"].unique())
            self.logger.info("Using non-K-Fold mode: all users' last item as test")

        # 按用户聚合数据
        all_sequences = {}
        for row in tqdm(
            data.itertuples(),
            total=len(data),
            desc="Aggregating user sequences",
        ):
            user_id = row.user
            item_id = row.question
            correct = row.label

            if user_id not in all_sequences:
                all_sequences[user_id] = {"items": [], "corrects": []}
            all_sequences[user_id]["items"].append(item_id)
            all_sequences[user_id]["corrects"].append(correct)

        # 构建训练序列和测试三元组
        train_sequences = {}
        test_triplets = []
        train_users = []
        test_users_list = []
        num_records = 0

        for user_id, seq_data in tqdm(
            all_sequences.items(),
            desc="Building train/test split",
        ):
            items = seq_data["items"]
            corrects = seq_data["corrects"]
            seq_len = len(items)
            num_records += seq_len

            # 训练序列：去掉最后一题
            train_len = seq_len - 1
            if train_len > 0:
                train_sequences[user_id] = [
                    [items[:train_len]],
                    [corrects[:train_len]],
                ]
                train_users.append(user_id)

            # 测试三元组：仅测试用户的最后一题
            if user_id in test_user_set:
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

        # 收集所有边
        rows = []
        cols = []

        for user_id, seq_data in tqdm(
            train_sequences.items(),
            desc="Collecting edges",
        ):
            items = seq_data[0][0]  # [[item_ids]] -> [item_ids]
            for item_id in items:
                # 用户 -> 题目
                rows.append(user_id)
                cols.append(item_id + num_users)
                # 题目 -> 用户 (无向图)
                rows.append(item_id + num_users)
                cols.append(user_id)

        # 构建稀疏邻接矩阵
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

        # 转换为 PyTorch 稀疏张量
        indices = torch.tensor(
            np.vstack([adj_norm.row, adj_norm.col]),
            dtype=torch.long,
        )
        values = torch.tensor(adj_norm.data, dtype=torch.float32)
        shape = adj_norm.shape

        adj_sparse = torch.sparse_coo_tensor(indices, values, shape)

        self.logger.info(f"Adjacency matrix shape: {shape}")
        self.logger.info(f"Number of edges: {len(values)}")

        return adj_sparse


__all__ = ["ABKTModelData"]
