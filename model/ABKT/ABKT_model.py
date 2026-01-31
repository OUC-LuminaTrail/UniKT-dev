"""
ABKT 模型架构定义

包含:
- IRT_2: 项目反应理论响应函数
- K_CMF: 知识模块 (Knowledge Module) - 基于矩阵分解的知识追踪
- GMF: 能力模块 (Ability Module) - 基于图矩阵分解的能力建模
"""

import torch
import torch.nn as nn

from utils.core import get_logger

logger = get_logger(__name__)


def IRT_2(
    user_k: torch.Tensor,
    item_k: torch.Tensor,
    item_q: torch.Tensor,
    guess: float,
) -> torch.Tensor:
    """
    项目反应理论 (IRT) 响应函数

    计算学生回答正确的概率：
    P = guess + (1 - guess) / (1 + exp(-d * r))
    其中 r = sum((user_k - item_k) * item_q) / sum(item_q)

    参数:
        user_k: 学生知识状态 [batch, num_skills] 或 [seq_len, num_skills]
        item_k: 题目难度参数 [batch, num_skills] 或 [seq_len, num_skills]
        item_q: 题目的 Q-Matrix 行 [batch, num_skills] 或 [seq_len, num_skills]
        guess: 猜测参数 (0-1)

    返回:
        预测的答题正确概率 [batch] 或 [seq_len]
    """
    d = 1.702  # IRT 常数
    # 计算能力与难度的差异，仅考虑相关技能
    r = torch.sum((user_k - item_k) * item_q, dim=-1) / (
        torch.sum(item_q, dim=-1) + 1e-8
    )
    p = guess + (1 - guess) / (1 + torch.exp(-d * r))
    return p


class K_CMF(nn.Module):
    """
    知识模块 (Knowledge Module)

    基于协同矩阵分解的知识追踪模型。
    通过学生初始知识状态和学习过程中的知识增长来建模学生能力。

    参数:
        k_hidden_size: 知识增长的隐藏维度 (k_K)
        skill_num: 技能数量
        user_num: 学生数量
        item_num: 题目数量
        Q_matrix: 题目-技能关联矩阵 [item_num, skill_num]
    """

    def __init__(
        self,
        k_hidden_size: int,
        skill_num: int,
        user_num: int,
        item_num: int,
        Q_matrix: torch.Tensor,
    ):
        super().__init__()

        self.k_hidden_size = k_hidden_size
        self.skill_num = skill_num
        self.user_num = user_num
        self.item_num = item_num

        # Q_matrix 扩展用于知识增长计算 [item_num, skill_num, k_hidden_size]
        self.register_buffer(
            "Q_matrix_m", Q_matrix.unsqueeze(2).repeat(1, 1, k_hidden_size)
        )
        self.register_buffer("Q_matrix", Q_matrix)

        # 学生初始知识状态 [user_num, skill_num]
        self.user_initial_k = nn.Parameter(torch.zeros((user_num, skill_num)) * 0.01)

        # 题目难度参数 [item_num, skill_num]
        self.item_k = nn.Parameter(torch.rand((item_num, skill_num)) * 0.01)

        # 学生知识增长参数 [user_num, skill_num, k_hidden_size]
        self.user_improving_k = nn.Parameter(
            torch.ones((user_num, skill_num, k_hidden_size)) * 0.01
        )

        # 题目知识增长参数 [item_num, skill_num, k_hidden_size]
        self.item_improving_k = nn.Parameter(
            torch.ones((item_num, skill_num, k_hidden_size)) * 0.01
        )

        # 使用 Q_matrix 掩码初始化题目知识增长参数
        with torch.no_grad():
            self.item_improving_k.data = (
                self.item_improving_k.data * self.Q_matrix_m.cpu()
            )

        logger.debug(
            f"K_CMF initialized: users={user_num}, items={item_num}, "
            f"skills={skill_num}, k_hidden={k_hidden_size}"
        )

    def forward(
        self, user: int, sq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：计算学生在答题序列中的知识状态演变

        参数:
            user: 学生 ID (标量)
            sq: 答题序列 (题目 ID 列表) [seq_len]

        返回:
            out: 知识状态序列 [seq_len + 1, skill_num]
                 (包含初始状态，所以比输入长1)
            ui_norm_k: 用于正则化的范数 (当前返回0)
            _: 占位符 (当前返回0)
        """
        length = len(sq)

        # 获取学生初始知识状态
        temp_k = self.user_initial_k[user, :]

        # 获取学生知识增长参数，扩展到序列长度
        # [seq_len, skill_num, k_hidden_size]
        user_improving_k_sq = (
            self.user_improving_k[user, :].unsqueeze(0).repeat(length, 1, 1)
        )

        # 存储知识状态序列
        sequence_k = [temp_k.unsqueeze(0)]

        # 获取题目知识增长参数 [seq_len, skill_num, k_hidden_size]
        item_improving_k_sq = self.item_improving_k[sq, :]

        # 计算知识增长量 [seq_len, skill_num]
        improves = torch.sum(user_improving_k_sq * item_improving_k_sq, dim=2)

        # 非负约束：知识只能增长不能减少
        improves_k = torch.relu(improves)

        # 迭代更新知识状态
        for i in range(length):
            improve_k = improves_k[i, :]
            temp_k = temp_k + improve_k
            sequence_k.append(temp_k.unsqueeze(0))

        # 拼接并通过 sigmoid 映射到 [0, 1]
        out = torch.sigmoid(torch.cat(sequence_k, dim=0))

        # 返回: [seq_len + 1, skill_num], 正则化项(未使用), 占位符
        ui_norm_k = torch.tensor(0.0, device=out.device)
        return out, ui_norm_k, ui_norm_k


class GMF(nn.Module):
    """
    能力模块 (Ability Module)

    基于图矩阵分解的能力建模，使用学生-题目二部图进行特征聚合。

    参数:
        n_users: 学生数量
        n_items: 题目数量
        embedding_k: 嵌入维度 (k_A)
        aj_norm: 归一化的邻接矩阵 [n_users + n_items, n_users + n_items]
        adj: 是否使用可学习的邻接权重
        layer: GNN 层数 (0-3)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_k: int,
        aj_norm: torch.Tensor,
        adj: bool = True,
        layer: int = 1,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_k = embedding_k
        self.adj_tag = adj
        self.layer = layer

        # 确保邻接矩阵为 float32
        aj_norm = aj_norm.to(torch.float32)

        # 节点嵌入 [n_users + n_items, embedding_k]
        self.embeddings = nn.Parameter(
            torch.randn((n_users + n_items, embedding_k)) * 0.01
        )

        # 预计算多跳邻接矩阵
        if layer >= 1:
            self.register_buffer("aj_norm_1", aj_norm.to_dense())
        if layer >= 2:
            aj_dense = aj_norm.to_dense()
            self.register_buffer("aj_norm_2", self.aj_norm_1.mm(aj_dense))
        if layer >= 3:
            aj_dense = aj_norm.to_dense()
            self.register_buffer("aj_norm_3", self.aj_norm_2.mm(aj_dense))

        # 可学习的邻接权重矩阵
        if adj:
            self.adj = nn.Parameter(torch.ones(aj_norm.shape))

        # 全局效应偏置
        self.user_GE = nn.Parameter(torch.zeros(n_users))
        self.item_GE = nn.Parameter(torch.zeros(n_items))

        logger.debug(
            f"GMF initialized: users={n_users}, items={n_items}, "
            f"embedding_k={embedding_k}, layer={layer}"
        )

    def forward(
        self, user_index: torch.Tensor, item_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：计算学生-题目对的预测分数

        参数:
            user_index: 学生索引 [batch]
            item_index: 题目索引 [batch]

        返回:
            pred_batch: 预测分数 [batch]
            u_norm: 学生嵌入范数 (用于正则化)
            i_norm: 题目嵌入范数 (用于正则化)
        """
        # 计算图卷积后的嵌入
        if self.adj_tag:
            # 使用可学习的邻接权重
            if self.layer == 0:
                G_embeddings = self.embeddings
            elif self.layer == 1:
                G_embeddings = (self.aj_norm_1 * self.adj).mm(self.embeddings)
            elif self.layer == 2:
                G_embeddings = (self.aj_norm_2 * self.adj).mm(self.embeddings)
            elif self.layer == 3:
                G_embeddings = (self.aj_norm_3 * self.adj).mm(self.embeddings)
            else:
                raise ValueError(f"GMF layer must be in [0, 1, 2, 3], got {self.layer}")
        else:
            # 不使用可学习权重
            if self.layer == 0:
                G_embeddings = self.embeddings
            elif self.layer == 1:
                G_embeddings = self.aj_norm_1.mm(self.embeddings)
            elif self.layer == 2:
                G_embeddings = self.aj_norm_2.mm(self.embeddings)
            elif self.layer == 3:
                G_embeddings = self.aj_norm_3.mm(self.embeddings)
            else:
                raise ValueError(f"GMF layer must be in [0, 1, 2, 3], got {self.layer}")

        # 获取学生和题目的嵌入
        user_embeddings_batch = G_embeddings[user_index, :]
        user_GE_batch = self.user_GE[user_index]

        item_embeddings_batch = G_embeddings[item_index + self.n_users, :]
        item_GE_batch = self.item_GE[item_index]

        # 计算预测分数: 内积 + 全局效应
        pred_batch = (
            (user_embeddings_batch * item_embeddings_batch).sum(dim=1)
            + user_GE_batch
            + item_GE_batch
        )

        # 计算正则化项
        u_norm = torch.mean(torch.pow(user_embeddings_batch, 2))
        i_norm = torch.mean(torch.pow(item_embeddings_batch, 2))

        return pred_batch, u_norm, i_norm


__all__ = ["IRT_2", "K_CMF", "GMF"]
