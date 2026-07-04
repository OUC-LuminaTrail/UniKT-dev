"""DIMKT 模型"""

import torch
import torch.nn as nn


class DIMKT(nn.Module):
    """DIMKT 模型。

    Args:
        num_q: 题目（question）数量（用于 q_emb/qd 查找前的题目嵌入）。
        num_c: 技能（concept/skill）数量。
        dropout: dropout 概率。
        emb_size: 嵌入维度。
        batch_size: 批大小。
        difficult_levels: 难度离散化等级数 D；sd/qd 取值范围为 [1, D+1]。
        skill_diff_table: 全局技能难度查表（long tensor, shape=[num_skills]）。
        question_diff_table: 全局题目难度查表（long tensor, shape=[num_questions]）。
    """

    def __init__(
        self,
        num_q,
        num_c,
        dropout,
        emb_size,
        batch_size,
        difficult_levels,
        skill_diff_table,
        question_diff_table,
    ):
        super().__init__()
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.difficult_levels = difficult_levels
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        # 初始知识状态 k
        self.knowledge = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, self.emb_size)), requires_grad=True
        )

        # padding_idx=0；调用方需保证传入的真实 id >= 1
        self.q_emb = nn.Embedding(self.num_q + 1, self.emb_size, padding_idx=0)
        self.c_emb = nn.Embedding(self.num_c + 1, self.emb_size, padding_idx=0)
        self.sd_emb = nn.Embedding(
            self.difficult_levels + 2, self.emb_size, padding_idx=0
        )
        self.qd_emb = nn.Embedding(
            self.difficult_levels + 2, self.emb_size, padding_idx=0
        )
        self.a_emb = nn.Embedding(2, self.emb_size)

        self.linear_1 = nn.Linear(4 * self.emb_size, self.emb_size)
        self.linear_2 = nn.Linear(1 * self.emb_size, self.emb_size)
        self.linear_3 = nn.Linear(1 * self.emb_size, self.emb_size)
        self.linear_4 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.linear_5 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.linear_6 = nn.Linear(4 * self.emb_size, self.emb_size)

        # 全局难度查表
        self.register_buffer("skill_diff_table", skill_diff_table)
        self.register_buffer("question_diff_table", question_diff_table)

    def forward(self, skill, question, response, mask):
        c_full = (skill + 1).masked_fill(~mask, 0)
        q_full = (question + 1).masked_fill(~mask, 0)
        sd_full = self.skill_diff_table[skill].masked_fill(~mask, 0)
        qd_full = self.question_diff_table[question].masked_fill(~mask, 0)
        a_full = response

        # next-item：当前时刻 [:-1]，目标时刻 [1:]
        q = q_full[:, :-1]
        c = c_full[:, :-1]
        sd = sd_full[:, :-1]
        qd = qd_full[:, :-1]
        a = a_full[:, :-1]
        qshft = q_full[:, 1:]
        cshft = c_full[:, 1:]
        sdshft = sd_full[:, 1:]
        qdshft = qd_full[:, 1:]

        if self.batch_size != len(q):
            self.batch_size = len(q)

        q_emb = self.q_emb(q)
        c_emb = self.c_emb(c)
        sd_emb = self.sd_emb(sd)
        qd_emb = self.qd_emb(qd)
        a_emb = self.a_emb(a)

        target_q = self.q_emb(qshft)
        target_c = self.c_emb(cshft)
        target_sd = self.sd_emb(sdshft)
        target_qd = self.qd_emb(qdshft)

        # 当前时刻输入投影
        input_data = torch.cat((q_emb, c_emb, sd_emb, qd_emb), -1)
        input_data = self.linear_1(input_data)

        target_data = torch.cat((target_q, target_c, target_sd, target_qd), -1)
        target_data = self.linear_1(target_data)

        k = self.knowledge.repeat(self.batch_size, 1)

        outputs = []
        seqlen = q.size(1)
        for t in range(seqlen):
            sd_1 = sd_emb[:, t]
            a_1 = a_emb[:, t]
            qd_1 = qd_emb[:, t]
            input_data_1 = input_data[:, t]

            qq = k - input_data_1

            gates_SDF = self.sigmoid(self.linear_2(qq))
            SDFt = self.dropout(self.tanh(self.linear_3(qq)))
            SDFt = gates_SDF * SDFt

            x = torch.cat((SDFt, a_1), -1)
            gates_PKA = self.sigmoid(self.linear_4(x))
            PKAt = gates_PKA * self.tanh(self.linear_5(x))

            ins = torch.cat((k, a_1, sd_1, qd_1), -1)
            gates_KSU = self.sigmoid(self.linear_6(ins))
            k = gates_KSU * k + (1.0 - gates_KSU) * PKAt

            outputs.append(k)

        output = torch.stack(outputs, dim=1)
        logits = torch.sum(target_data * output, dim=-1)
        y = self.sigmoid(logits)

        return y
