"""HGIKT variant with hypergraph branch only (unweighted)."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.nn import HGNNConv
from torch_geometric.nn import Linear

from model.layers import GeneralInteraction, HistoryRecap
from utils.core import register_model


class HyperGNN(nn.Module):
    """Based on HGIKT HyperGNN module."""

    def __init__(
        self,
        in_ch: int,
        n_hid: int,
        n_class: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hgc1 = HGNNConv(
            in_ch, n_hid, bias=True, use_bn=False, drop_rate=dropout, is_last=False
        )
        self.hgc2 = HGNNConv(
            n_hid, n_class, bias=True, use_bn=False, drop_rate=dropout, is_last=True
        )

    def forward(self, x: torch.Tensor, hg: Any) -> torch.Tensor:
        """Forward pass. Note: DHG's HGNNConv handles unweighted hypergraphs if no weights provided."""
        x1 = F.relu(self.hgc1(x, hg))
        x2 = self.hgc2(x1, hg)
        return x2


@register_model("HGIKT_Hyper_Only_Unweighted")
class HGIKT_Hyper_Only_Unweighted(nn.Module):
    """HGIKT variant with only hypergraph (unweighted).

    The heterogeneous graph branch is removed.
    This variant uses a standard (unweighted) hypergraph.
    """

    def __init__(
        self,
        args: Any,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.args = args
        self.data_metadata = data_metadata

        self.hidden_dim = args.hidden_dim
        self.lstm_layers = args.lstm_layers
        self.dropout = args.dropout
        self.num_skills = data_metadata["num_skills"]
        # 获取聚类数量，默认为 3 (与 HGIKT_data.py 一致)
        self.num_clusters = getattr(args, "num_difficulty_clusters", 3)

        self.question_embedding = nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.skill_embedding = nn.Embedding(
            num_embeddings=self.num_skills,
            embedding_dim=self.hidden_dim,
        )
        self.answer_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.hidden_dim,
        )
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        # Hypergraph module (unweighted)
        self.hgnn_conv = HyperGNN(
            in_ch=self.hidden_dim,
            n_hid=self.hidden_dim,
            n_class=self.hidden_dim,
            dropout=self.dropout,
        )

        # Full connected layer
        self.fc_exercise = Linear(
            self.hidden_dim * 2, self.hidden_dim, weight_initializer="uniform"
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # History recap module
        self.history_review = HistoryRecap(
            hist_neighbor_num=args.history_neighbour,
            att_bound=args.att_bound,
        )

        # General interaction module
        self.general_interaction = GeneralInteraction(hidden_dim=self.hidden_dim)

    def forward(
        self,
        user_sequence: torch.Tensor,
        user_response: torch.Tensor,
        user_mask: torch.Tensor,
        hetero_graph: Any,
        hypergraph: Any,
        question_skill_matrix: torch.Tensor,
    ) -> torch.Tensor:
        B, _ = user_sequence.size()
        device = user_sequence.device
        answers_embedding = self.answer_embedding(user_response)

        # 【核心消融点1：无边权】强制清除超图边权，确保仅利用聚类结构拓扑
        if hasattr(hypergraph, "edge_weight"):
            hypergraph.edge_weight = None

        # 【核心消融点2：超图提供嵌入层】
        # 使用 HGNN 演化题目特征
        question_hyper_conv = self.hgnn_conv(self.question_embedding.weight, hypergraph)
        
        # 使用 v2e 聚合得到技能（难度簇）特征
        # 注意：此处 raw_e_features 包含 num_skills * num_clusters 个特征
        raw_e_features = hypergraph.v2e(question_hyper_conv, aggr="mean")
        
        # 将聚类簇特征聚合回原始技能 ID
        # 映射规律：E_idx = skill_id * num_clusters + cluster_id
        if raw_e_features.size(0) >= self.num_skills * self.num_clusters:
            skill_hyper_conv = raw_e_features.view(self.num_skills, self.num_clusters, self.hidden_dim).mean(dim=1)
        else:
            # 异常处理：如果不满足聚类倍数，尝试安全对准
            skill_hyper_conv = torch.zeros(self.num_skills, self.hidden_dim, device=device)
            min_size = min(self.num_skills, raw_e_features.size(0))
            skill_hyper_conv[:min_size] = raw_e_features[:min_size]

        # 【保持一致性】叠加技能原始 Embedding。
        # 这样确保了在去掉异构图模块后，原本属于技能 ID 的参数化表征依然存在。
        skill_hyper_conv = skill_hyper_conv + self.skill_embedding.weight

        # 3. 构造练习序列特征
        question_embedding_sequence = question_hyper_conv[user_sequence]
        exercise_emb = torch.cat(
            [question_embedding_sequence, answers_embedding], dim=-1
        )
        exercise_emb = F.relu(self.fc_exercise(exercise_emb))
        exercise_emb = self.embedding_dropout(exercise_emb)

        # 4. LSTM 演化
        lstm_output, _ = self.lstm(exercise_emb)

        # 5. 获取下一题和历史邻居
        next_user_sequence = torch.zeros_like(user_sequence)
        if user_sequence.size(1) > 1:
            next_user_sequence[:, :-1] = user_sequence[:, 1:]
            next_user_sequence[:, -1] = 0

        next_question_embedding = question_hyper_conv[next_user_sequence]

        history_question_neighbors = self.history_review(
            question_embedding_sequence,
            next_question_embedding,
            exercise_emb,
            user_mask,
        )

        # 构造学生状态集合：[B, S, 1 + K, H]
        student_status = torch.cat(
            [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
        )

        # 6. 构造知识状态集合 (Knowledge Status)
        # 从 question_skill_matrix 获取下一题对应的技能 IDs
        q_skill_vectors = question_skill_matrix[next_user_sequence] # [B, S, num_skills]
        
        # 获取最大关联技能数以便 Padding
        max_skills_per_question = int(q_skill_vectors.sum(dim=-1).max().item())
        max_skills_per_question = max(1, max_skills_per_question)
        
        skill_counts = q_skill_vectors.sum(dim=-1).long()
        # 找到非零索引 (即关联的技能 ID)
        sorted_skill_indices = torch.argsort(q_skill_vectors, dim=-1, descending=True)
        related_skill_ids = sorted_skill_indices[..., :max_skills_per_question].clone()

        # 处理 Padding
        pos = torch.arange(max_skills_per_question, device=device).view(1, 1, -1)
        valid_pos_mask = pos < skill_counts.unsqueeze(-1)
        
        padding_index = self.num_skills # 使用总技能数作为 padding 索引
        padding_ids = torch.full_like(related_skill_ids, padding_index)
        related_skill_ids = torch.where(valid_pos_mask, related_skill_ids, padding_ids)

        # 构造带全零 Padding 的特征矩阵
        skill_conv_padded = torch.cat(
            [
                skill_hyper_conv,
                torch.zeros(1, self.hidden_dim, device=device, dtype=skill_hyper_conv.dtype),
            ],
            dim=0,
        )
        related_skill_embs = skill_conv_padded[related_skill_ids]

        knowledge_status = torch.cat(
            [next_question_embedding.unsqueeze(2), related_skill_embs], dim=2
        )

        return self.general_interaction(student_status, knowledge_status, user_mask)
