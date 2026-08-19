"""DGEKT 模型数据处理模块。

- 题目 / 答案 / 掩码序列（question 级）
- 三张静态图，节点空间统一为 2Q 个交互节点（答对题 q = q，答错题 q = q+Q）：
  超图拉普拉斯 ``hyper_g``（题目-概念关联矩阵的块对角展开）、出 / 入转移
  邻接 ``adj_out`` / ``adj_in``（训练折相邻交互计数 + 自环行归一化）
"""

from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


def _safe_inv_sqrt(values: np.ndarray) -> np.ndarray:
    """Element-wise ``x^-0.5`` with zeros kept as zeros."""
    inv_sqrt = np.zeros_like(values, dtype=np.float64)
    nz = values > 0
    inv_sqrt[nz] = values[nz] ** -0.5
    return inv_sqrt


class DGEKTDataset(Dataset):
    """DGEKT 数据集。

    每个样本返回 ``(question, response, mask)``：
        question: 题目序列 [S]
        response: 答案序列 [S]
        mask: 有效位置掩码 [S]
    """

    def __init__(self, questions, responses, masks):
        self.questions = questions
        self.responses = responses
        self.masks = masks

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.questions[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
        )


class DGEKTModelData(QuestionModelData):
    """DGEKT 模型数据加载器。"""

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据与模型所需的图结构。

        Returns:
            (train_dataset, val_dataset, test_dataset, info) 元组，其中 ``info``
            为含 ``num_questions``、``num_skills``、``max_seq_len``、
            ``hyper_g``、``adj_out``、``adj_in`` 的字典。
        """
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")
        max_seq_len = self.data_src.get_metadata("max_seq_len")

        user_sequence, user_response, user_mask, _ = self.load_sequence_data()

        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        if fold_idx is None:
            raise ValueError("K-fold cross-validation is not enabled (fold < 0).")

        train_slices, val_slices, test_slices = self.split_kfold_data(
            user_sequence, user_response, user_mask, fold_idx=fold_idx
        )
        (tr_q, tr_a, tr_mask) = train_slices
        (va_q, va_a, va_mask) = val_slices
        (te_q, te_a, te_mask) = test_slices

        q_matrix = self.build_relationship_matrix(("question", "has", "skill"))
        hyper_g = self._build_hyper_graph(q_matrix)
        # Transition counts are collected from the training fold only.
        adj_out, adj_in = self._build_transition_adj(tr_q, tr_a, tr_mask, num_questions)
        logger.info(
            f"DGEKT graphs: hyper_g nnz={hyper_g._nnz()}, "
            f"adj_out nnz={adj_out._nnz()}, adj_in nnz={adj_in._nnz()}"
        )
        logger.info(
            f"DGEKT data: num_questions={num_questions}, num_skills={num_skills}, "
            f"max_seq_len={max_seq_len}"
        )

        train_dataset = DGEKTDataset(tr_q, tr_a, tr_mask)
        val_dataset = DGEKTDataset(va_q, va_a, va_mask)
        test_dataset = DGEKTDataset(te_q, te_a, te_mask)

        logger.info(
            f"DGEKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        info = {
            "num_questions": num_questions,
            "num_skills": num_skills,
            "max_seq_len": max_seq_len,
            "hyper_g": hyper_g,
            "adj_out": adj_out,
            "adj_in": adj_in,
        }
        return train_dataset, val_dataset, test_dataset, info

    def _build_hyper_graph(self, q_matrix: np.ndarray) -> torch.Tensor:
        """超图拉普拉斯 ``G = Dv^-1/2 · H · De^-1 · H^T · Dv^-1/2``。

        关联矩阵 H 为块对角 ``[[Qm, 0], [0, Qm]]``（2Q 交互节点 × 2S 超边），
        使答对 / 答错节点空间各自聚合概念超边。
        """
        qm = sparse.coo_matrix(q_matrix)
        H = sparse.block_diag([qm, qm]).tocoo()

        node_deg = np.asarray(H.sum(axis=1)).ravel()
        edge_deg = np.asarray(H.sum(axis=0)).ravel()
        inv_sqrt_node = _safe_inv_sqrt(node_deg)
        inv_edge = np.zeros_like(edge_deg, dtype=np.float64)
        nz = edge_deg > 0
        inv_edge[nz] = 1.0 / edge_deg[nz]

        G = (
            sparse.diags(inv_sqrt_node)
            @ H
            @ sparse.diags(inv_edge)
            @ H.T
            @ sparse.diags(inv_sqrt_node)
        )
        return self._to_torch_sparse(G.tocoo())

    def _build_transition_adj(
        self,
        questions: np.ndarray,
        responses: np.ndarray,
        masks: np.ndarray,
        num_questions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """出 / 入转移邻接：相邻交互计数加自环后行归一化。"""
        inter = questions + (1 - responses) * num_questions
        pair_valid = (masks[:, :-1] & masks[:, 1:]).astype(bool)
        src = inter[:, :-1][pair_valid]
        dst = inter[:, 1:][pair_valid]
        size = 2 * num_questions
        # Duplicate (src, dst) entries are summed on CSR conversion.
        counts = sparse.coo_matrix(
            (np.ones(src.size), (src, dst)), shape=(size, size)
        ).tocsr()
        # Transpose before adding self-loops so each direction is normalized
        # independently.
        return (
            self._row_normalize(counts + sparse.eye(size)),
            self._row_normalize(counts.T + sparse.eye(size)),
        )

    def _row_normalize(self, mat: sparse.spmatrix) -> torch.Tensor:
        """Row-stochastic 归一化，零行保持为零。"""
        rowsum = np.asarray(mat.sum(axis=1)).ravel()
        r_inv = np.zeros_like(rowsum, dtype=np.float64)
        nz = rowsum > 0
        r_inv[nz] = 1.0 / rowsum[nz]
        return self._to_torch_sparse((sparse.diags(r_inv) @ mat).tocoo())

    @staticmethod
    def _to_torch_sparse(mat: sparse.coo_matrix) -> torch.Tensor:
        indices = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        values = torch.from_numpy(mat.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, mat.shape).coalesce()
