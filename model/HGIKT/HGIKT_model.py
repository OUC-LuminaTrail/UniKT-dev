from typing import Any

import torch
import torch.nn as nn
from dhg.nn import HGNNConv
from torch.nn import functional as F
from torch_geometric.nn import HGTConv, Linear

from utils.core import register_model

from ..layers import GeneralInteraction, HistoryRecap


class HeteroGNN(nn.Module):
    """基于 HGT 的异质图神经网络模块。

    使用 Heterogeneous Graph Transformer 进行多层异构图聚合。

    Args:
        embedding_dim: 节点嵌入维度
        n_hop: GNN 层数
        heads: 注意力头数
        dropout: Dropout 概率
        metadata: 异构图元数据

    Example:
        >>> gnn = HeteroGNN(embedding_dim=128, n_hop=2, heads=4, dropout=0.2, metadata=metadata)
        >>> output = gnn(x_dict, edge_index_dict)
    """

    def __init__(
        self,
        embedding_dim: int,
        n_hop: int,
        heads: int,
        dropout: float,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
    ) -> None:
        super().__init__()
        self.n_hop = n_hop
        self.heads = heads
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        for _ in range(n_hop):
            conv = HGTConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                metadata=metadata,
                heads=heads,
            )
            self.convs.append(conv)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x_dict: 节点特征字典
            edge_index_dict: 边索引字典

        Returns:
            聚合后的节点表示字典
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # 对每个节点类型的输出应用激活和 dropout
            new_x_dict = {}
            for node_type, x in x_dict.items():
                if x is not None:
                    x = F.gelu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                new_x_dict[node_type] = x
            x_dict = new_x_dict
        return x_dict


class HyperGNN(nn.Module):
    """基于dhg框架的超图神经网络

    使用dhg.nn.HGNNConv实现双层超图卷积，支持加权超图。

    数学公式：
        X' = σ(D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ)

    其中：
        - X 是输入顶点特征矩阵
        - H 是超图关联矩阵
        - W_e 是超边权重对角矩阵（可自定义或默认为单位矩阵）
        - D_v 是顶点度数对角矩阵
        - D_e 是超边度数对角矩阵
        - Θ 是可学习参数
    """

    def __init__(
        self,
        in_ch: int,
        n_hid: int,
        n_class: int,
        dropout: float = 0.0,
        use_edge_weights: bool = True,
    ) -> None:
        super().__init__()
        self.use_edge_weights = use_edge_weights

        # 第一层卷积：聚合直接邻居问题特征
        # is_last=False 表示使用激活函数和dropout
        self.hgc1 = HGNNConv(
            in_ch, n_hid, bias=True, use_bn=False, drop_rate=dropout, is_last=False
        )
        # 第二层卷积：聚合间接邻居的问题特征
        # is_last=True 表示跳过内置激活和dropout，在forward中手动应用ReLU（与原实现一致）
        self.hgc2 = HGNNConv(
            n_hid, n_class, bias=True, use_bn=False, drop_rate=dropout, is_last=True
        )

    def forward(self, x: torch.Tensor, hg: Any) -> torch.Tensor:  # dhg.Hypergraph
        """前向传播。

        Args:
            x: 输入特征矩阵 [num_vertices, in_ch]
            hg: dhg.Hypergraph 超图

        Returns:
            输出特征矩阵 [num_vertices, n_class]
        """
        x1 = self.hgc1(x, hg)
        x2 = F.relu(self.hgc2(x1, hg))

        return x2


class MoEFusion(nn.Module):
    """混合专家融合 (Mixture-of-Experts Fusion)。

    将两个视图的特征处理视为不同的"专家"。
    引入一个共享专家(Shared Expert)捕获共性。
    使用门控网络(Router)动态分配权重。
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim

        # 专家网络:
        # Expert 1: 处理 View 1
        # Expert 2: 处理 View 2
        # Expert 3: 处理 View 1 + View 2 (Shared)
        self.expert1 = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.expert2 = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.expert_shared = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout)
        )

        # 门控网络: 输入两个视图，输出专家的权重
        self.router = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 3),  # 3个专家
            nn.Softmax(dim=-1),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        # 展平批次维度以适应 Linear
        B_shape = view1.shape[:-1]
        v1_flat = view1.reshape(-1, self.dim)
        v2_flat = view2.reshape(-1, self.dim)

        e1 = self.expert1(v1_flat)
        e2 = self.expert2(v2_flat)
        e_shared = self.expert_shared(torch.cat([v1_flat, v2_flat], dim=-1))

        # 堆叠专家输出: [N, 3, D]
        experts = torch.stack([e1, e2, e_shared], dim=1)

        # 计算路由权重
        combined = torch.cat([v1_flat, v2_flat], dim=-1)
        weights = self.router(combined)  # [N, 3]

        # 加权求和
        # weights: [N, 3] -> [N, 3, 1]
        # experts: [N, 3, D]
        fused = torch.sum(experts * weights.unsqueeze(-1), dim=1)

        return self.norm(fused).reshape(*B_shape, self.dim)


@register_model("HGIKT")
class HGIKT(nn.Module):
    """HGIKT 主模型。

    层次化图知识追踪模型，融合异构图和超图进行预测。

    Args:
        args: 模型参数配置
        data_metadata: 数据集元数据
        hetero_metadata: 异构图元数据
        **kwargs: 额外的关键字参数

    Example:
        >>> model = HGIKT(args, data_metadata, hetero_metadata)
        >>> logits = model(user_sequence, user_response, user_mask, hetero_graph, hypergraph, question_skill_matrix)
    """

    def __init__(
        self,
        args: Any,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # 保存参数
        self.args = args
        # 元数据
        self.data_metadata = data_metadata

        # 模型参数
        self.hidden_dim = args.hidden_dim  # 隐藏层维度
        self.lstm_layers = args.lstm_layers  # LSTM层数
        self.dropout = args.dropout  # Dropout概率，所有层共享

        # Embedding
        self.question_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.question_embedding_hyper = torch.nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.skill_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_skills"],
            embedding_dim=self.hidden_dim,
        )
        self.assignment_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_assignments"],
            embedding_dim=self.hidden_dim,
        )
        self.template_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_templates"],
            embedding_dim=self.hidden_dim,
        )
        self.answer_embedding = torch.nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.hidden_dim,
        )
        self.embedding_dropout = torch.nn.Dropout(p=self.dropout)

        # 异质图模块
        self.hetero_conv = HeteroGNN(
            embedding_dim=self.hidden_dim,
            n_hop=args.n_hop,
            heads=args.heads,
            dropout=self.dropout,
            metadata=hetero_metadata,
        )

        # 超图模块
        self.hgnn_conv = HyperGNN(
            in_ch=self.hidden_dim,  # 输入通道数
            n_hid=self.hidden_dim,  # 隐藏层通道数
            n_class=self.hidden_dim,  # 输出通道数
            dropout=self.dropout,  # Dropout概率
        )

        # 融合模块
        self.fuse = MoEFusion(dim=self.hidden_dim, dropout=self.dropout)

        # 全连接层，将练习嵌入投影到隐藏维度
        self.fc_exercise = Linear(
            self.hidden_dim * 2, self.hidden_dim, weight_initializer="uniform"
        )

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # 历史回顾模块
        self.history_review = HistoryRecap(
            hist_neighbor_num=args.history_neighbour,
            att_bound=args.att_bound,
        )

        # 广义交互模块
        self.general_interaction = GeneralInteraction(hidden_dim=self.hidden_dim)

    def forward(
        self,
        user_sequence: torch.Tensor,  # [B, S]
        user_response: torch.Tensor,  # [B, S]
        user_mask: torch.Tensor,  # [B, S]
        hetero_graph: Any,  # HeteroData
        hypergraph: Any,  # dhg.Hypergraph
        question_skill_matrix: torch.Tensor,  # [Q, K]
    ) -> torch.Tensor:  # [B, S]
        """前向传播。

        Args:
            user_sequence: 用户问题序列 [B, S]
            user_response: 用户回答序列 [B, S]
            user_mask: 有效位置掩码 [B, S]
            hetero_graph: 异构图数据
            hypergraph: 超图数据
            question_skill_matrix: 问题-技能关联矩阵 [Q, K]

        Returns:
            预测 logits [B, S]
        """
        # 批量大小
        B, _ = user_sequence.size()

        # 获取用户答题序列的回复嵌入
        # [B, S, embedding_dim]
        answers_embedding: torch.Tensor = self.answer_embedding(user_response)

        # 加权超图卷积
        question_hyper_conv: torch.Tensor = self.hgnn_conv(
            self.question_embedding_hyper.weight, hypergraph
        )

        # 异构图卷积
        conv = self.hetero_conv(
            {
                "question": self.question_embedding.weight,
                "skill": self.skill_embedding.weight,
                "assignment": self.assignment_embedding.weight,
                "template": self.template_embedding.weight,
            },
            hetero_graph.edge_index_dict,
        )

        # 图卷积得到的问题嵌入 [num_questions, embedding_dim]
        question_hetero_conv: torch.Tensor = conv["question"]
        # 图卷积得到的技能嵌入 [num_skills, embedding_dim]
        skill_hetero_conv: torch.Tensor = conv["skill"]

        # 融合异构图与超图的题目表示
        # question_hyper_conv: [num_questions, embedding_dim]
        # question_hetero_conv: [num_questions, embedding_dim]
        question_conv_fused = self.fuse(
            question_hetero_conv, question_hyper_conv
        )  # [num_questions, embedding_dim]

        # 按照用户序列索引获取对应的问题的嵌入表示
        # user_sequence: [B, S], question_conv: [num_questions, embedding_dim]
        # question_embedding_sequence: [B, S, embedding_dim]
        question_embedding_sequence = question_conv_fused[user_sequence]

        # 组合问题和答案嵌入得到练习嵌入
        exercise_emb = torch.cat(
            [question_embedding_sequence, answers_embedding], dim=-1
        )  # [B, S, 2*E]

        # 将练习嵌入投影到隐藏维度
        exercise_emb = F.relu(self.fc_exercise(exercise_emb))  # [B, S, H]
        exercise_emb = self.embedding_dropout(exercise_emb)

        # lstm_output [B, S, H]
        lstm_output, _ = self.lstm(exercise_emb)

        # 获取下一题问题序列，最后一个时间步用零向量占位
        next_user_sequence = torch.zeros_like(user_sequence)  # [B, S]
        if user_sequence.size(1) > 1:
            # 将 user_sequence 向左移动一位，最后一位用0填充
            next_user_sequence[:, :-1] = user_sequence[:, 1:]
            next_user_sequence[:, -1] = 0

        # 获取下一题的问题序列嵌入
        # next_question_embedding: [B, S, embedding_dim]
        next_question_embedding: torch.Tensor = question_conv_fused[next_user_sequence]

        # 历史回顾模块
        # question_embedding_sequence: [B, S, E]
        # next_question_embedding: [B, S, E]
        # lstm_output: [B, S, H]
        history_question_neighbors = self.history_review(
            question_embedding_sequence,
            next_question_embedding,
            exercise_emb,
            user_mask,
        )  # history_question_neighbors: [B, S, M, H]

        # 构造学生相关状态集合：LSTM 输出 + 历史邻居
        # lstm_output: [B, S, H] -> [B, S, 1, H]
        # history_question_neighbors: [B, S, M, H]
        # student_status: [B, S, M+1, H]
        student_status = torch.cat(
            [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
        )  # [B, S, M+1, H]

        # 构建知识相关状态集合：下一题特征 + 相关知识点特征
        # 获取每个问题关联的技能
        # 此处得到的每一个问题关联的0/1向量表示其关联的技能
        # next_user_sequence: [B, S] 每个元素是问题ID
        q_skill_vectors = question_skill_matrix[
            next_user_sequence
        ]  # [B, S, num_skills]

        # 通过对二值向量降序排序，将值为1的技能索引排在前面
        sorted_skill_indices = torch.argsort(
            q_skill_vectors, dim=-1, descending=True
        )  # [B, S, num_skills]

        # 计算每行最大关联技能数
        max_skills_per_question = int(q_skill_vectors.sum(dim=-1).max().item())
        # 计算每个位置实际关联的技能数量（1的个数）
        skill_counts = q_skill_vectors.sum(dim=-1).long()  # [B, S]

        # 选取每个位置前 K 个技能索引（K=max_skills_per_q）
        related_skill_ids = sorted_skill_indices[
            ..., :max_skills_per_question
        ].clone()  # [B, S, K]

        # 不足 K 个技能的位置使用 padding 索引填充
        device = next_user_sequence.device
        pos = torch.arange(max_skills_per_question, device=device).view(
            1, 1, -1
        )  # [1,1,K]
        valid_pos_mask = pos < skill_counts.unsqueeze(-1)  # [B, S, K]

        padding_index = skill_hetero_conv.size(0)  # 额外的零向量行索引
        padding_ids = torch.full_like(related_skill_ids, padding_index)
        related_skill_ids = torch.where(
            valid_pos_mask, related_skill_ids, padding_ids
        )  # [B, S, K]

        # 为 padding 索引追加一行零向量
        skill_conv_padded = torch.cat(
            [
                skill_hetero_conv,
                torch.zeros(
                    1, self.hidden_dim, device=device, dtype=skill_hetero_conv.dtype
                ),
            ],
            dim=0,
        )  # [num_skills+1, embedding_dim]

        # 获取相关知识点嵌入
        # 从 skill_conv_padded 中取出对应的技能嵌入
        # related_skill_ids: [B, S, max_skills_per_question]
        # related_skill_embs: [B, S, max_skills_per_question, embedding_dim]
        related_skill_embs = skill_conv_padded[related_skill_ids]

        # 拼接得到知识相关状态集合
        # next_question_embedding: [B, S, embedding_dim] -> [B, S, 1, embedding_dim]
        # related_skill_embs: [B, S, max_skills_per_question, embedding_dim]
        knowledge_status = torch.cat(
            [next_question_embedding.unsqueeze(2), related_skill_embs],
            dim=2,
        )  # [B, S, max_skills_per_question+1, embedding_dim]

        # 广义交互模块
        logits = self.general_interaction(
            student_status, knowledge_status, user_mask
        )  # [B, S]

        return logits  # [B, S]
