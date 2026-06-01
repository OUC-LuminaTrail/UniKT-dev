"""QIKT 模型实现

QIKT (Question-centric Knowledge Tracing): 双路径 LSTM 模型，
通过问题级和概念级预测的融合进行知识追踪。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.core import register_model


class MLP(nn.Module):
    """MLP 分类解码器"""

    def __init__(self, n_layer, hidden_dim, output_dim, dropout):
        super().__init__()
        self.lins = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer)]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


@register_model("QIKT")
class QIKT(nn.Module):
    """QIKT 模型

    使用双路径 LSTM 分别处理问题和概念序列，通过多任务预测头
    融合问题级和概念级预测。

    Args:
        num_questions: 问题数量
        num_skills: 技能/概念数量
        emb_size: 嵌入维度
        max_concepts: 每个问题最大技能数
        dropout: Dropout 概率
        mlp_layer_num: MLP 预测头的层数
    """

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        emb_size: int = 64,
        max_concepts: int = 4,
        dropout: float = 0.1,
        mlp_layer_num: int = 1,
    ):
        super().__init__()
        self.num_q = num_questions
        self.num_c = num_skills
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.max_concepts = max_concepts

        # 嵌入层
        self.question_emb = nn.Embedding(num_questions, emb_size)
        self.concept_emb = nn.Parameter(torch.randn(num_skills, emb_size))
        self.que_c_linear = nn.Linear(2 * emb_size, emb_size)

        # 双路径 LSTM
        self.que_lstm = nn.LSTM(emb_size * 4, emb_size, batch_first=True)
        self.concept_lstm = nn.LSTM(emb_size * 2, emb_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)

        # 问题预测头
        self.out_question_next = MLP(mlp_layer_num, emb_size * 3, 1, dropout)
        self.out_question_all = MLP(mlp_layer_num, emb_size, num_questions, dropout)

        # 概念预测头
        self.out_concept_next = MLP(mlp_layer_num, emb_size * 3, num_skills, dropout)
        self.out_concept_all = MLP(mlp_layer_num, emb_size, num_skills, dropout)

    def _get_avg_skill_emb(self, c):
        """计算多概念问题的平均技能嵌入

        Args:
            c: 技能ID [B, S, max_concepts]，-1 为填充

        Returns:
            平均技能嵌入 [B, S, emb_size]
        """
        # 在概念嵌入前添加零向量行，用于处理 -1 填充
        concept_emb_padded = torch.cat(
            [torch.zeros(1, self.emb_size, device=c.device), self.concept_emb],
            dim=0,
        )

        related_concepts = (c + 1).long()
        concept_emb_sum = concept_emb_padded[related_concepts].sum(dim=-2)

        concept_num = (related_concepts != 0).sum(dim=-1).unsqueeze(-1).float()
        concept_num = torch.where(
            concept_num == 0, torch.ones_like(concept_num), concept_num
        )
        return concept_emb_sum / concept_num

    def _avg_fusion_concepts(self, y_concept, c_shift):
        """将概念级预测融合为多概念问题的标量预测

        对每个问题的多个关联技能的预测值取平均。

        Args:
            y_concept: 概念预测 [B, S-1, num_c]
            c_shift: 移位后的技能ID [B, S-1, max_concepts]

        Returns:
            融合后的预测 [B, S-1]
        """
        max_num_concept = c_shift.shape[-1]
        concept_mask = c_shift.long() != -1
        concept_index = F.one_hot(
            torch.where(c_shift != -1, c_shift, 0).long(), self.num_c
        ).float()

        concept_sum = (
            y_concept.unsqueeze(2).expand(-1, -1, max_num_concept, -1) * concept_index
        ).sum(-1)
        concept_sum = concept_sum * concept_mask.float()

        denom = concept_mask.sum(dim=-1).float()
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        return concept_sum.sum(-1) / denom

    def forward(self, question, response, mask, skills):
        """前向传播

        模型在时刻 t 的输出基于 question[0:t] 和 response[0:t-1]
        预测 response[t]，输出长度为 S-1，填充为 [B, S]。

        Args:
            question: 问题ID序列 [B, S]
            response: 响应序列 [B, S]
            mask: 有效位置掩码 [B, S]
            skills: 多概念技能ID序列 [B, S, max_concepts]

        Returns:
            各预测头的字典，所有值形状为 [B, S]（首列为填充）
        """
        B, S = question.shape

        # 嵌入
        emb_q = self.question_emb(question)
        emb_c = self._get_avg_skill_emb(skills)
        emb_qc = torch.cat([emb_q, emb_c], dim=-1)

        # 响应条件嵌入
        r = response.float().unsqueeze(-1)
        emb_qca = torch.cat(
            [
                emb_qc * (1 - r).expand_as(emb_qc),
                emb_qc * r.expand_as(emb_qc),
            ],
            dim=-1,
        )

        # 移位嵌入
        emb_qc_shift = emb_qc[:, 1:, :]
        emb_qca_current = emb_qca[:, :-1, :]

        # 问题 LSTM 路径
        que_h = self.dropout_layer(self.que_lstm(emb_qca_current)[0])

        h_next_q = torch.cat([emb_qc_shift, que_h], dim=-1)
        y_question_next = torch.sigmoid(self.out_question_next(h_next_q)).squeeze(-1)

        y_question_all_raw = torch.sigmoid(self.out_question_all(que_h))
        q_shift = question[:, 1:]
        y_question_all = (y_question_all_raw * F.one_hot(q_shift, self.num_q)).sum(-1)

        # 概念 LSTM 路径
        emb_ca = torch.cat(
            [
                emb_c * (1 - r).expand_as(emb_c),
                emb_c * r.expand_as(emb_c),
            ],
            dim=-1,
        )
        emb_ca_current = emb_ca[:, :-1, :]
        concept_h = self.dropout_layer(self.concept_lstm(emb_ca_current)[0])

        h_next_c = torch.cat([emb_qc_shift, concept_h], dim=-1)
        y_concept_next_raw = torch.sigmoid(self.out_concept_next(h_next_c))
        y_concept_all_raw = torch.sigmoid(self.out_concept_all(concept_h))

        c_shift = skills[:, 1:, :]
        y_concept_next = self._avg_fusion_concepts(y_concept_next_raw, c_shift)
        y_concept_all = self._avg_fusion_concepts(y_concept_all_raw, c_shift)

        # 从 [B, S-1] 填充到 [B, S]，末尾补零
        # 约定：position t 预测 response[t+1]，配合 skip_first=True 使用
        dummy = torch.zeros(B, 1, device=question.device)
        return {
            "y_question_next": torch.cat([y_question_next, dummy], dim=1),
            "y_question_all": torch.cat([y_question_all, dummy], dim=1),
            "y_concept_next": torch.cat([y_concept_next, dummy], dim=1),
            "y_concept_all": torch.cat([y_concept_all, dummy], dim=1),
        }
