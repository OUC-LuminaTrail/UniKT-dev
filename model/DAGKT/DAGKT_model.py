from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import Linear

from ..layers import GNN_QS, GeneralInteraction, HistoryRecap


class Autoencoder(nn.Module):
    """自编码器模块，用于难度和尝试次数的特征编码。

    架构: 1 → ae_hidden_dim → embedding_dim → ae_hidden_dim → 1
    编码器和解码器均使用 Sigmoid 激活函数。

    Args:
        input_dim: 输入维度（默认 1，对应标量输入）
        ae_hidden_dim: 自编码器隐藏层维度
        embedding_dim: 编码后的输出维度
    """

    def __init__(
        self,
        input_dim: int = 1,
        ae_hidden_dim: int = 50,
        embedding_dim: int = 100,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ae_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(ae_hidden_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, ae_hidden_dim),
            nn.Sigmoid(),
            nn.Linear(ae_hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            x: 输入张量 [..., input_dim]

        Returns:
            (encoded, reconstruction_loss) 元组:
                - encoded: 编码后的表示 [..., embedding_dim]
                - reconstruction_loss: 标量 MSE 重建损失
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        recon_loss = F.mse_loss(decoded, x)
        return encoded, recon_loss


class DAGKT(nn.Module):
    """DAGKT 主模型。

    在 GIKT 基础上增加难度模块和尝试次数模块：
    - 难度模块: 用自编码器编码题目正确率，与题目嵌入融合
    - 尝试次数模块: 用自编码器编码学生对每题的尝试次数，与答案嵌入融合

    总损失 = 预测损失(BCE) + 难度自编码器重建损失 + 尝试次数自编码器重建损失

    Args:
        args: 模型参数配置
        data_metadata: 数据集元数据，包含 num_questions 和 num_skills
        question_difficulty: 题目正确率张量 [num_questions, 1]
        **kwargs: 额外的关键字参数
    """

    def __init__(
        self,
        args: Any,
        data_metadata: dict[str, Any],
        question_difficulty: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.args = args
        self.data_metadata = data_metadata

        # 模型参数
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.lstm_layers = args.lstm_layers
        self.dropout = args.dropout
        ae_hidden_dim = getattr(args, "ae_hidden_dim", 50)

        num_questions = data_metadata["num_questions"]
        num_skills = data_metadata["num_skills"]

        # 题目正确率（作为 buffer 而非可训练参数）
        if question_difficulty is not None:
            self.register_buffer("difficulty_rates", question_difficulty.float())
        else:
            self.register_buffer("difficulty_rates", torch.zeros(num_questions, 1))

        # Embedding 层（同 GIKT）
        self.question_embedding = nn.Embedding(
            num_embeddings=num_questions,
            embedding_dim=self.embedding_dim,
        )
        self.skill_embedding = nn.Embedding(
            num_embeddings=num_skills,
            embedding_dim=self.embedding_dim,
        )
        self.answer_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_dropout = nn.Dropout(p=self.dropout)

        # GNN 层（同 GIKT，复用 GNN_QS）
        self.conv = GNN_QS(
            embedding_dim=self.embedding_dim,
            n_hop=args.n_hop,
            heads=args.heads,
            dropout=self.dropout,
        )

        # 难度自编码器
        self.difficulty_ae = Autoencoder(
            input_dim=1,
            ae_hidden_dim=ae_hidden_dim,
            embedding_dim=self.embedding_dim,
        )

        # 尝试次数自编码器
        self.attempt_ae = Autoencoder(
            input_dim=1,
            ae_hidden_dim=ae_hidden_dim,
            embedding_dim=self.embedding_dim,
        )

        # 难度融合层: concat(q_emb, diff_emb) → embedding_dim
        self.transdiff = nn.Linear(2 * self.embedding_dim, self.embedding_dim)

        # 尝试融合层: concat(answer_emb, attempt_emb) → embedding_dim
        self.trans_answer = nn.Linear(2 * self.embedding_dim, self.embedding_dim)

        # 全连接层: concat(q_fused, ans_fused) → hidden_dim
        self.fc_exercise = Linear(
            2 * self.embedding_dim, self.hidden_dim, weight_initializer="uniform"
        )

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # 历史回顾模块（同 GIKT）
        self.history_review = HistoryRecap(
            hist_neighbor_num=args.history_neighbour,
            att_bound=args.att_bound,
        )

        # 广义交互模块（同 GIKT）
        self.general_interaction = GeneralInteraction(hidden_dim=self.hidden_dim)

    def forward(
        self,
        user_sequence: torch.Tensor,  # [B, S]
        user_response: torch.Tensor,  # [B, S]
        user_mask: torch.Tensor,  # [B, S]
        graph: Any,
        question_skill_matrix: torch.Tensor,  # [Q, K]
        attempt_counts: torch.Tensor,  # [B, S]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            user_sequence: 用户问题序列 [B, S]
            user_response: 用户回答序列 [B, S]
            user_mask: 有效位置掩码 [B, S]
            graph: 问题-技能异构图
            question_skill_matrix: 问题-技能关联矩阵 [Q, K]
            attempt_counts: 每个学生对每题的累计尝试次数 [B, S]

        Returns:
            (logits, loss_diff, loss_attempt) 元组
        """
        B, S = user_sequence.size()

        # ================================================================
        # 1. 难度编码：将题目正确率通过自编码器编码为嵌入
        # ================================================================
        # difficulty_rates: [num_questions, 1] → diff_emb: [num_questions, E]
        diff_emb, loss_diff = self.difficulty_ae(self.difficulty_rates)

        # ================================================================
        # 2. 图卷积（同 GIKT）
        # ================================================================
        conv = self.conv(
            {
                "question": self.question_embedding.weight,
                "skill": self.skill_embedding.weight,
            },
            graph.edge_index_dict,
        )
        question_conv: torch.Tensor = conv["question"]  # [num_questions, E]
        skill_conv: torch.Tensor = conv["skill"]  # [num_skills, E]

        # ================================================================
        # 3. 当前题嵌入 + 难度融合
        # ================================================================
        q_emb = question_conv[user_sequence]  # [B, S, E]
        cur_diff_emb = diff_emb[user_sequence]  # [B, S, E]
        q_fused = torch.relu(
            self.transdiff(torch.cat([q_emb, cur_diff_emb], dim=-1))
        )  # [B, S, E]

        # ================================================================
        # 4. 尝试次数编码：将尝试次数通过自编码器编码为嵌入
        # ================================================================
        # attempt_counts: [B, S] → reshape [B*S, 1]
        attempt_input = attempt_counts.reshape(-1, 1)  # [B*S, 1]
        attempt_encoded, loss_attempt = self.attempt_ae(attempt_input)
        # [B*S, E] → [B, S, E]
        attempt_emb = attempt_encoded.reshape(B, S, self.embedding_dim)

        # ================================================================
        # 5. 答案嵌入 + 尝试融合
        # ================================================================
        ans_emb = self.answer_embedding(user_response)  # [B, S, E]
        ans_fused = torch.relu(
            self.trans_answer(torch.cat([ans_emb, attempt_emb], dim=-1))
        )  # [B, S, E]

        # ================================================================
        # 6. 练习嵌入: concat(q_fused, ans_fused) → hidden_dim
        # ================================================================
        exercise_emb = torch.cat([q_fused, ans_fused], dim=-1)  # [B, S, 2E]
        exercise_emb = torch.relu(self.fc_exercise(exercise_emb))  # [B, S, H]
        exercise_emb = self.embedding_dropout(exercise_emb)

        # ================================================================
        # 7. LSTM 处理（同 GIKT）
        # ================================================================
        lstm_output, _ = self.lstm(exercise_emb)  # [B, S, H]

        # ================================================================
        # 8. 下一题处理（同 GIKT 但加入难度融合）
        # ================================================================
        next_user_sequence = torch.zeros_like(user_sequence)  # [B, S]
        if S > 1:
            next_user_sequence[:, :-1] = user_sequence[:, 1:]
            next_user_sequence[:, -1] = 0

        # 下一题图卷积嵌入 + 难度融合
        next_q_emb = question_conv[next_user_sequence]  # [B, S, E]
        next_diff_emb = diff_emb[next_user_sequence]  # [B, S, E]
        next_q_fused = torch.relu(
            self.transdiff(torch.cat([next_q_emb, next_diff_emb], dim=-1))
        )  # [B, S, E]

        # 下一题答案嵌入（用于 GeneralInteraction 中的 knowledge_status 构造）
        next_user_response = torch.zeros_like(user_response)  # [B, S]
        if S > 1:
            next_user_response[:, :-1] = user_response[:, 1:]
            next_user_response[:, -1] = 0

        # ================================================================
        # 9. 历史回顾模块（同 GIKT）
        # ================================================================
        history_question_neighbors = self.history_review(
            q_emb,
            next_q_emb,
            exercise_emb,
            user_mask,
        )  # [B, S, M, H]

        # ================================================================
        # 10. 构造学生状态集合（同 GIKT）
        # ================================================================
        student_status = torch.cat(
            [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
        )  # [B, S, M+1, H]

        # ================================================================
        # 11. 构建知识状态集合（同 GIKT）
        # ================================================================
        q_skill_vectors = question_skill_matrix[
            next_user_sequence
        ]  # [B, S, num_skills]

        sorted_skill_indices = torch.argsort(
            q_skill_vectors, dim=-1, descending=True
        )  # [B, S, num_skills]

        max_skills_per_question = int(q_skill_vectors.sum(dim=-1).max().item())
        skill_counts = q_skill_vectors.sum(dim=-1).long()  # [B, S]

        related_skill_ids = sorted_skill_indices[
            ..., :max_skills_per_question
        ].clone()  # [B, S, K]

        device = next_user_sequence.device
        pos = torch.arange(max_skills_per_question, device=device).view(
            1, 1, -1
        )  # [1, 1, K]
        valid_pos_mask = pos < skill_counts.unsqueeze(-1)  # [B, S, K]

        padding_index = skill_conv.size(0)
        padding_ids = torch.full_like(related_skill_ids, padding_index)
        related_skill_ids = torch.where(
            valid_pos_mask, related_skill_ids, padding_ids
        )  # [B, S, K]

        skill_conv_padded = torch.cat(
            [
                skill_conv,
                torch.zeros(1, self.hidden_dim, device=device, dtype=skill_conv.dtype),
            ],
            dim=0,
        )  # [num_skills+1, H]

        related_skill_embs = skill_conv_padded[related_skill_ids]

        # knowledge_status 使用 next_q_fused（已融合难度）作为问题表示
        knowledge_status = torch.cat(
            [next_q_fused.unsqueeze(2), related_skill_embs],
            dim=2,
        )  # [B, S, max_skills_per_question+1, E/H]

        # ================================================================
        # 12. 广义交互模块（同 GIKT）
        # ================================================================
        logits = self.general_interaction(
            student_status, knowledge_status, user_mask
        )  # [B, S]

        return logits, loss_diff, loss_attempt
