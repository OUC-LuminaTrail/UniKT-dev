"""DKVMN (Dynamic Key-Value Memory Networks) 模型实现

论文: Dynamic Key-Value Memory Networks for Knowledge Tracing (Zhang et al., 2017)
"""

import torch
from torch import nn
from torch.nn.init import kaiming_normal_


class DKVMN(nn.Module):
    """DKVMN 模型

    使用键值记忆网络追踪学生知识状态。

    Args:
        num_c: 概念数量
        dim_s: 状态向量维度
        size_m: 记忆槽位数量
        dropout: Dropout概率
    """

    def __init__(self, num_c: int, dim_s: int, size_m: int, dropout: float = 0.2):
        super().__init__()
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m

        # 概念嵌入层（键嵌入）
        self.k_emb_layer = nn.Embedding(self.num_c, self.dim_s)
        # 交互嵌入层（值嵌入）：概念+响应组合
        self.v_emb_layer = nn.Embedding(self.num_c * 2, self.dim_s)

        # 记忆矩阵
        self.Mk = nn.Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = nn.Parameter(torch.Tensor(self.size_m, self.dim_s))
        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        # 记忆写入层
        self.e_layer = nn.Linear(self.dim_s, self.dim_s)
        self.a_layer = nn.Linear(self.dim_s, self.dim_s)

        # 预测层
        self.f_layer = nn.Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = nn.Dropout(dropout)
        self.p_layer = nn.Linear(self.dim_s, 1)

    def forward(
        self, sequence: torch.Tensor, response: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            sequence: 概念ID序列，形状为 [batch_size, seq_len]
            response: 响应序列，形状为 [batch_size, seq_len]
            mask: 有效位置掩码，形状为 [batch_size, seq_len]

        Returns:
            预测结果，形状为 [batch_size, seq_len]
            p[:, t] 使用历史 0..t-1 预测位置 t 的响应
        """
        batch_size = sequence.shape[0]

        # 嵌入
        x = sequence + self.num_c * response
        k = self.k_emb_layer(sequence)  # [B, S, dim_s]
        v = self.v_emb_layer(x)  # [B, S, dim_s]

        # 初始化记忆
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, size_m, dim_s]
        Mv = [Mvt]

        # 注意力权重
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)  # [B, S, size_m]

        # 写入过程：逐时间步更新记忆
        e = torch.sigmoid(self.e_layer(v))  # [B, S, dim_s]
        a = torch.tanh(self.a_layer(v))  # [B, S, dim_s]

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (
                wt.unsqueeze(-1) * at.unsqueeze(1)
            )
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)  # [B, S+1, size_m, dim_s]

        # 读取过程：从记忆中读取并预测
        read_content = (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2)  # [B, S, dim_s]
        f = torch.tanh(
            self.f_layer(torch.cat([read_content, k], dim=-1))
        )  # [B, S, dim_s]
        p = torch.sigmoid(self.p_layer(self.dropout_layer(f))).squeeze(-1)  # [B, S]

        return p
