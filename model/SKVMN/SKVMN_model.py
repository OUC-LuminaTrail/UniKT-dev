"""SKVMN (Sparse Key-Value Memory Network) 模型实现

论文: Sparse Knowledge Tracing: Efficient and Interpretable Knowledge
Tracking with Sparse Memory Networks (Abdelrahman et al., 2023)

在 DKVMN 的键值记忆机制之上引入 Hop-LSTM：将每个时间步的相关权重
阈值化为 identity 向量，检索序列内最近的相同 identity 时刻 t-λ，
在 LSTM 递推时把 hidden/cell state 恢复为 t-λ 时刻的状态，从而
跳过冗余的中间步骤。

迁移自 pykt-toolkit (pykt/models/skvmn.py)，主要改动：
- 移除模块级全局 device，改为跟随输入张量的设备
- triangular_layer 消除逐行 Python 拼接，改为 reshape + clamp
- Hop-LSTM 消除 batch 维 Python 双重循环，改为 gather + where
- 修复原实现别名 bug：原版 hop 替换时 ``hx[j, :] = ...`` 原地写入
  上一时间步 append 进 hidden_state 的张量（cx 有 clone 保护而 hx
  遗漏），会污染历史 hidden_state 并影响最终预测；此处改为非原地替换
"""

import torch
from torch import nn
from torch.nn.init import kaiming_normal_


class SKVMN(nn.Module):
    """SKVMN 模型

    Args:
        num_c: 概念数量
        dim_s: 状态向量维度
        size_m: 记忆槽位数量
        dropout: Dropout 概率
        use_onehot: 写入记忆的交互表示使用 one-hot 向量而非交互嵌入
        tri_a/tri_b/tri_c: 三角隶属度函数的左界、峰值、右界
        id_weak/id_strong: identity 阈值化的一档/二档阈值
    """

    def __init__(
        self,
        num_c: int,
        dim_s: int,
        size_m: int,
        dropout: float = 0.2,
        use_onehot: bool = False,
        tri_a: float = 0.075,
        tri_b: float = 0.088,
        tri_c: float = 1.00,
        id_weak: float = 0.1,
        id_strong: float = 0.6,
    ):
        super().__init__()
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.use_onehot = use_onehot
        self.tri_a = tri_a
        self.tri_b = tri_b
        self.tri_c = tri_c
        self.id_weak = id_weak
        self.id_strong = id_strong

        self.k_emb_layer = nn.Embedding(self.num_c, self.dim_s)
        self.x_emb_layer = nn.Embedding(2 * self.num_c + 1, self.dim_s)
        self.Mk = nn.Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = nn.Parameter(torch.Tensor(self.size_m, self.dim_s))
        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        # 记忆写入门（erase/add），作用于 [f_t, y_t] 融合后的交互表示
        self.a_embed = nn.Linear(
            self.num_c + self.dim_s if use_onehot else self.dim_s * 2, self.dim_s
        )
        self.erase_layer = nn.Linear(self.dim_s, self.dim_s, bias=True)
        self.add_layer = nn.Linear(self.dim_s, self.dim_s, bias=True)
        kaiming_normal_(self.erase_layer.weight)
        kaiming_normal_(self.add_layer.weight)
        nn.init.constant_(self.erase_layer.bias, 0)
        nn.init.constant_(self.add_layer.bias, 0)

        self.f_layer = nn.Linear(self.dim_s * 2, self.dim_s)
        self.hx = nn.Parameter(torch.Tensor(1, self.dim_s))
        self.cx = nn.Parameter(torch.Tensor(1, self.dim_s))
        kaiming_normal_(self.hx)
        kaiming_normal_(self.cx)
        self.dropout_layer = nn.Dropout(dropout)
        self.p_layer = nn.Linear(self.dim_s, 1)
        self.lstm_cell = nn.LSTMCell(self.dim_s, self.dim_s)

    def _hop_sources(self, correlation_weight: torch.Tensor) -> torch.Tensor:
        """计算每个时间步的 hop 源索引

        将相关权重经三角隶属度函数阈值化为 identity 向量，在序列内
        检索最近的相同 identity 时刻 t-λ（仅严格过去）。

        Args:
            correlation_weight: 相关权重，形状为 [batch_size, seq_len, size_m]

        Returns:
            hop 源索引矩阵，形状为 [batch_size, seq_len]，-1 表示该步无 hop
        """
        batch_size, seq_len, _ = correlation_weight.shape
        device = correlation_weight.device
        neg = torch.tensor(-1e32, device=device)

        # 三角隶属度: min((w-a)/(b-a), (c-w)/(c-b)) 再与 0 取 max
        w = correlation_weight.reshape(-1)
        tri = torch.stack(
            [
                (w - self.tri_a) / (self.tri_b - self.tri_a),
                (self.tri_c - w) / (self.tri_c - self.tri_b),
            ]
        )
        tri, _ = torch.min(tri, dim=0)
        tri = torch.relu(tri)

        # identity 向量: <weak -> 0, [weak, strong) -> 1, >=strong -> 2
        identity = torch.where(
            tri.ge(self.id_strong), 2.0, torch.where(tri.ge(self.id_weak), 1.0, 0.0)
        ).view(batch_size, seq_len, -1)

        # identity 向量间的平方欧氏距离，0 即 identity 相同
        norm = torch.sum(identity * identity, dim=2, keepdim=True)  # [B, S, 1]
        distances = (
            norm
            + norm.transpose(1, 2)
            - 2 * torch.bmm(identity, identity.transpose(1, 2))
        )
        # 正距离（不同 identity）置 -inf 候补，仅保留严格过去（下三角）
        distances = torch.where(distances > 0.0, neg, distances)
        future_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=0
        )
        distances = distances.masked_fill(future_mask, neg)
        # 加列索引后在候补中取最大：优先距离 0，平局时取最近的过去时刻
        distances = distances + torch.arange(seq_len, device=device).view(1, 1, seq_len)
        values, indices = torch.topk(distances, 1, dim=2, largest=True)

        # 有效的 hop：topk 值 >= 0（找到相同 identity 的历史时刻）
        return torch.where(values.squeeze(-1) >= 0, indices.squeeze(-1), -1)

    def forward(
        self, sequence: torch.Tensor, response: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            sequence: 概念 ID 序列，形状为 [batch_size, seq_len]
            response: 响应序列，形状为 [batch_size, seq_len]
            mask: 有效位置掩码，形状为 [batch_size, seq_len]（不参与计算，
                对齐由 trainer 负责，与 pykt 行为一致）

        Returns:
            预测结果，形状为 [batch_size, seq_len]
            p[:, t] 使用历史 0..t-1 预测位置 t 的响应
        """
        batch_size, seq_len = sequence.shape

        x = sequence + self.num_c * response  # [B, S]
        k = self.k_emb_layer(sequence)  # [B, S, dim_s]

        if self.use_onehot:
            # 每题的 one-hot 响应向量：在概念 q 位置填 r
            q_col = sequence.reshape(batch_size * seq_len, 1)
            r_expand = (
                response.unsqueeze(2)
                .expand(-1, -1, self.num_c)
                .reshape(batch_size * seq_len, self.num_c)
            )
            r_onehot = torch.zeros(
                batch_size * seq_len,
                self.num_c,
                device=sequence.device,
                dtype=response.dtype,
            )
            y = (
                r_onehot.scatter(1, q_col, r_expand)
                .reshape(batch_size, seq_len, -1)
                .float()
            )
        else:
            y = self.x_emb_layer(x)  # [B, S, dim_s]

        # 相关权重一次性计算（键记忆固定不变）
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)  # [B, S, size_m]

        # 逐步读-算-写：f_t 依赖写入前的记忆，写入门输入依赖 f_t，故无法全向量化
        mem_value = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, size_m, dim_s]
        ft = []
        for t in range(seq_len):
            wt = w[:, t]  # [B, size_m]
            read_content = (wt.unsqueeze(-1) * mem_value).sum(dim=1)  # [B, dim_s]
            f = torch.tanh(
                self.f_layer(torch.cat([read_content, k[:, t]], dim=-1))
            )  # [B, dim_s]
            ft.append(f)

            write_embed = self.a_embed(torch.cat([f, y[:, t]], dim=-1))  # [B, dim_s]
            erase = torch.sigmoid(self.erase_layer(write_embed)).unsqueeze(
                1
            )  # [B, 1, dim_s]
            add = torch.tanh(self.add_layer(write_embed)).unsqueeze(1)  # [B, 1, dim_s]
            w_col = wt.unsqueeze(-1)  # [B, size_m, 1]
            mem_value = mem_value * (1 - w_col * erase) + w_col * add

        ft = torch.stack(ft, dim=0)  # [S, B, dim_s]

        # Hop-LSTM：相同 identity 时刻恢复其 hidden/cell state 后再递推
        hop_src = self._hop_sources(w)  # [B, S]
        has_hop = hop_src >= 0
        src = hop_src.clamp(min=0)
        batch_idx = torch.arange(batch_size, device=ft.device)

        h_hist = ft.new_empty(seq_len, batch_size, self.dim_s)
        c_hist = ft.new_empty(seq_len, batch_size, self.dim_s)
        hx = self.hx.repeat(batch_size, 1)  # [B, dim_s]
        cx = self.cx.repeat(batch_size, 1)  # [B, dim_s]
        # 单次同步判定整批是否有 hop；逐步 any() 会每步触发 GPU->CPU 同步
        any_hop = bool(has_hop.any())
        for t in range(seq_len):
            if any_hop:
                # 非原地替换，避免污染已存入 *_hist 的历史状态；
                # 无 hop 的行 where 掩码为 False，不会传播未初始化的 gather 值
                hop = has_hop[:, t].unsqueeze(-1)
                hx = torch.where(hop, h_hist[src[:, t], batch_idx], hx)
                cx = torch.where(hop, c_hist[src[:, t], batch_idx], cx)
            hx, cx = self.lstm_cell(ft[t], (hx, cx))
            h_hist[t] = hx
            c_hist[t] = cx

        hidden_state = h_hist.permute(1, 0, 2)  # [B, S, dim_s]
        p = torch.sigmoid(self.p_layer(self.dropout_layer(hidden_state))).squeeze(
            -1
        )  # [B, S]

        return p
