import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class IEKTMLP(nn.Module):
    """多层感知机头

    ``n_layer`` 个同维隐藏层（ReLU）+ 一个输出层，输出前施加 Dropout。
    当 ``n_layer == 0`` 时退化为单层线性映射。
    """

    def __init__(self, n_layer: int, hidden_dim: int, output_dim: int, dropout=0.0):
        super().__init__()
        self.lins = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layer)]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


class IEKTGRUCell(nn.Module):
    """单步 GRU 单元。"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.g_ir = nn.Linear(input_dim, hidden_dim)
        self.g_iz = nn.Linear(input_dim, hidden_dim)
        self.g_in = nn.Linear(input_dim, hidden_dim)
        self.g_hr = nn.Linear(hidden_dim, hidden_dim)
        self.g_hz = nn.Linear(hidden_dim, hidden_dim)
        self.g_hn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h):
        r_t = torch.sigmoid(self.g_ir(x) + self.g_hr(h))
        z_t = torch.sigmoid(self.g_iz(x) + self.g_hz(h))
        n_t = torch.tanh(self.g_in(x) + self.g_hn(h) * r_t)
        h_t = (1 - z_t) * n_t + z_t * h
        return h_t


class IEKT(nn.Module):
    """IEKT 模型

    每个时间步 t：
      1. 认知策略 ``pi_cog`` 根据题目表征与历史隐状态 ``ques_h=[v,h]`` 采样认知级别 m_t；
      2. 用 ``[h, v, m_t]`` 预测当前题目作答 logits；
      3. 敏感性策略 ``pi_sens`` 根据（真值+预测）拼接的 ``out_x`` 采样获取级别 s_t；
      4. 用 ``[v, s_t]`` 与真值标签更新隐状态 h（GRU）。

    Args:
        num_questions: 问题数量
        num_skills: 技能/概念数量
        emb_size: 嵌入维度
        max_concepts: 每个问题最大技能数
        lamb: 强化学习损失的权重
        n_layer: 预测/策略头 MLP 的隐藏层数
        cog_levels: 认知估计离散级别数（m_t 动作空间）
        acq_levels: 知识获取敏感性离散级别数（s_t 动作空间）
        dropout: Dropout 概率
        gamma: 策略梯度的奖励折扣因子
    """

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        emb_size: int = 64,
        max_concepts: int = 4,
        lamb: int = 40,
        n_layer: int = 1,
        cog_levels: int = 10,
        acq_levels: int = 10,
        dropout: float = 0.0,
        gamma: float = 0.93,
    ):
        super().__init__()
        self.num_q = num_questions
        self.num_c = num_skills
        self.emb_size = emb_size
        self.max_concepts = max_concepts
        self.lamb = lamb
        self.gamma = gamma
        self.cog_levels = cog_levels
        self.acq_levels = acq_levels

        # 问题/知识点嵌入
        self.question_emb = nn.Embedding(num_questions, emb_size)
        self.concept_emb = nn.Parameter(torch.randn(num_skills, emb_size))

        # 预测头：输入 [h, v, m_t] = emb + 2*emb + 2*emb = 5*emb
        self.predictor = IEKTMLP(n_layer, emb_size * 5, 1, dropout)
        # 认知 / 获取级别嵌入矩阵（动作查询表）
        self.cog_matrix = nn.Parameter(torch.randn(cog_levels, emb_size * 2))
        self.acq_matrix = nn.Parameter(torch.randn(acq_levels, emb_size * 2))
        # 认知策略头：输入 ques_h=[v, h] = 2*emb + emb = 3*emb
        self.select_preemb = IEKTMLP(n_layer, emb_size * 3, cog_levels, dropout)
        # 敏感性策略头：输入 out_x = 12*emb
        self.checker_emb = IEKTMLP(n_layer, emb_size * 12, acq_levels, dropout)
        # 知识状态 GRU：输入 [v_cat + e_cat] = 4*emb，隐状态 emb
        # self.gru_h = IEKTGRUCell(emb_size * 4, emb_size)
        self.gru_h = nn.GRUCell(emb_size * 4, emb_size)

    def _get_avg_skill_emb(self, c):
        """多概念问题的平均技能嵌入。

        Args:
            c: 技能 ID [..., max_concepts]，-1 为填充

        Returns:
            平均技能嵌入 [..., emb_size]
        """
        concept_emb_padded = torch.cat(
            [torch.zeros(1, self.emb_size, device=c.device), self.concept_emb], dim=0
        )
        related = (c + 1).long()  # -1 -> 0（零向量行），用于填充
        emb_sum = concept_emb_padded[related].sum(dim=-2)
        concept_num = (related != 0).sum(dim=-1).unsqueeze(-1).float()
        concept_num = torch.where(
            concept_num == 0, torch.ones_like(concept_num), concept_num
        )
        return emb_sum / concept_num

    def get_ques_representation(self, q, c):
        """题目表征 v = concat(avg_skill_emb, question_emb)，维度 2*emb_size。

        支持任意前导维度（单步 ``[B]`` 或整序列 ``[B, S]``）。
        """
        concept_avg = self._get_avg_skill_emb(c)
        que_emb = self.question_emb(q)
        return torch.cat([concept_avg, que_emb], dim=-1)

    def pi_cog(self, ques_h):
        """认知策略分布（softmax over cog_levels）。"""
        return F.softmax(self.select_preemb(ques_h), dim=-1)

    def pi_sens(self, out_x):
        """敏感性策略分布（softmax over acq_levels）。"""
        return F.softmax(self.checker_emb(out_x), dim=-1)

    def _sample_action(self, probs):
        """训练时采样，评估时取 argmax。"""
        if self.training:
            return Categorical(probs).sample()
        return probs.argmax(dim=-1)

    def forward(self, question, response, mask, skills):
        """前向传播：逐时间步采样并预测。

        Args:
            question: 问题ID序列 [B, S]
            response: 响应序列 [B, S]
            mask: 有效位置掩码 [B, S]
            skills: 多概念技能ID序列 [B, S, max_concepts]

        Returns:
            dict:
              ``logits`` [B, S]            每步原始 logits（predict_current）
              ``p_actions`` [B, S]         认知动作 m_t 索引
              ``emb_actions`` [B, S]       敏感性动作 s_t 索引
              ``pre_states`` [B, S, 3*emb] 认知策略输入状态 ques_h
              ``states`` [B, S, 12*emb]    敏感性策略输入状态 out_x
        """
        B, S = question.shape
        device = question.device

        # 一次性计算整序列题目表征，循环内按位索引
        v_seq = self.get_ques_representation(question, skills)  # [B, S, 2*emb]

        h = torch.zeros(B, self.emb_size, device=device)
        logits_list, p_action_list, emb_action_list = [], [], []
        pre_state_list, state_list = [], []

        for t in range(S):
            v = v_seq[:, t]  # [B, 2*emb]
            ques_h = torch.cat([v, h], dim=1)  # [B, 3*emb]   equation 4

            # 认知动作 m_t
            cog_probs = self.pi_cog(ques_h)  # [B, cog_levels]
            cog_action = self._sample_action(cog_probs)
            m_t = self.cog_matrix[cog_action]  # [B, 2*emb]

            # 预测
            h_v = torch.cat([h, v], dim=1)  # [B, 3*emb]
            logits = self.predictor(torch.cat([h_v, m_t], dim=1)).squeeze(-1)  # [B]

            # 敏感性动作 s_t
            gt = response[:, t].float().unsqueeze(-1)  # [B, 1]
            out_x_gt = torch.cat(
                [h_v * gt, h_v * (1 - gt)], dim=1
            )  # [B, 6*emb]   equation 9
            pred01 = (logits > 0).float().unsqueeze(-1)
            out_x_pred = torch.cat(
                [h_v * pred01, h_v * (1 - pred01)], dim=1
            )  # [B, 6*emb]
            out_x = torch.cat([out_x_gt, out_x_pred], dim=1)  # [B, 12*emb]

            sens_probs = self.pi_sens(out_x)  # [B, acq_levels]
            sens_action = self._sample_action(sens_probs)
            s_t = self.acq_matrix[sens_action]  # [B, 2*emb]

            # 状态更新
            v_cat = torch.cat([v * gt, v * (1 - gt)], dim=1)  # [B, 4*emb]
            e_cat = torch.cat([s_t * (1 - gt), s_t * gt], dim=1)  # [B, 4*emb]
            gru_in = v_cat + e_cat  # [B, 4*emb]
            h = self.gru_h(gru_in, h)  # [B, emb]

            logits_list.append(logits)
            p_action_list.append(cog_action)
            emb_action_list.append(sens_action)
            pre_state_list.append(ques_h)
            state_list.append(out_x)

        return {
            "logits": torch.stack(logits_list, dim=1),  # [B, S]
            "p_actions": torch.stack(p_action_list, dim=1),  # [B, S]
            "emb_actions": torch.stack(emb_action_list, dim=1),  # [B, S]
            "pre_states": torch.stack(pre_state_list, dim=1),  # [B, S, 3*emb]
            "states": torch.stack(state_list, dim=1),  # [B, S, 12*emb]
        }
