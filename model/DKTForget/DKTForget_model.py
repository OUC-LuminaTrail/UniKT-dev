"""DKT-Forget 模型实现

在 DKT 的基础上额外引入三种"遗忘"相关特征，通过 CIntegration 模块注入：
- rgap (repeated gap): 距离上次练习同一技能的时间间隔（log2 分钟）
- sgap (sequence gap): 距离上一次任意交互的时间间隔（log2 分钟）
- pcount (past count): 该技能历史练习次数（log2）
"""

import torch
from torch import nn


class CIntegration(nn.Module):
    """将遗忘特征（rgap/sgap/pcount）整合进表示向量。

    对三种特征分别做 one-hot，拼接后过无偏置线性层得到 c，与输入 vt 逐元素相乘，
    再把原始 one-hot 拼接特征 concat 到末尾，得到 theta = [vt ⊙ c ; ct]。

    Args:
        num_rgap: repeated-gap 词表大小
        num_sgap: sequence-gap 词表大小
        num_pcount: past-count 词表大小
        emb_dim: 输入/输出表示维度
    """

    def __init__(self, num_rgap, num_sgap, num_pcount, emb_dim):
        super().__init__()
        self.register_buffer("rgap_eye", torch.eye(num_rgap))
        self.register_buffer("sgap_eye", torch.eye(num_sgap))
        self.register_buffer("pcount_eye", torch.eye(num_pcount))
        ntotal = num_rgap + num_sgap + num_pcount
        self.cemb = nn.Linear(ntotal, emb_dim, bias=False)

    def forward(self, vt, rgap, sgap, pcount):
        rgap = self.rgap_eye[rgap]  # [B, S, num_rgap]
        sgap = self.sgap_eye[sgap]  # [B, S, num_sgap]
        pcount = self.pcount_eye[pcount]  # [B, S, num_pcount]
        ct = torch.cat((rgap, sgap, pcount), -1)  # [B, S, ntotal]
        cct = self.cemb(ct)  # [B, S, emb_dim]
        theta = torch.mul(vt, cct)  # 元素级相乘
        theta = torch.cat((theta, ct), -1)  # [B, S, emb_dim + ntotal]
        return theta


class DKTForget(nn.Module):
    """DKT-Forget 模型

    Args:
        num_c: 技能（概念）数量
        num_rgap: repeated-gap 词表大小
        num_sgap: sequence-gap 词表大小
        num_pcount: past-count 词表大小
        emb_size: 嵌入与 LSTM 隐藏维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        num_c,
        num_rgap,
        num_sgap,
        num_pcount,
        emb_size,
        dropout=0.1,
    ):
        super().__init__()
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size

        self.interaction_emb = nn.Embedding(num_c * 2, self.emb_size)
        self.c_integration = CIntegration(num_rgap, num_sgap, num_pcount, self.emb_size)
        ntotal = num_rgap + num_sgap + num_pcount
        self.lstm_layer = nn.LSTM(
            self.emb_size + ntotal, self.hidden_size, batch_first=True
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.hidden_size + ntotal, num_c)

    def forward(self, q, r, rgaps, sgaps, pcounts):
        """前向传播

        Args:
            q: 技能ID序列 [B, S]
            r: 响应序列 [B, S]
            rgaps/sgaps/pcounts: 遗忘特征序列 [B, S]
        """
        x = q + self.num_c * r
        xemb = self.interaction_emb(x)  # [B, S, emb_size]
        theta_in = self.c_integration(xemb, rgaps, sgaps, pcounts)
        h, _ = self.lstm_layer(theta_in)  # h[:, t] = 交互 0..t 后的状态

        h_pred = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        theta_out = self.c_integration(h_pred, rgaps, sgaps, pcounts)
        theta_out = self.dropout_layer(theta_out)
        logits = self.out_layer(theta_out)  # [B, S, num_c]

        y = logits.gather(-1, q.long().unsqueeze(-1)).squeeze(-1)  # [B, S]
        y = torch.sigmoid(y)
        return y
