"""DKT (Deep Knowledge Tracing) 模型实现"""

import torch
from torch import nn


class DKT(nn.Module):
    """DKT 模型

    Args:
        num_c: 概念数量
        emb_size: 嵌入维度
        dropout: Dropout概率
    """

    def __init__(self, num_c: int, emb_size: int, dropout: float = 0.1):
        super().__init__()
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size

        # 交互嵌入层：将概念ID和响应组合编码
        self.interaction_emb = nn.Embedding(self.num_c * 2, self.emb_size)

        # LSTM层：处理序列数据
        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        # Dropout层：防止过拟合
        self.dropout_layer = nn.Dropout(dropout)
        # 输出层：预测每个概念的掌握概率
        self.out_layer = nn.Linear(self.hidden_size, self.num_c)

    def forward(
        self, sequence: torch.Tensor, response: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        Args:
            sequence: 概念ID序列，形状为 [batch_size, sequence_length]
            response: 响应序列，形状为 [batch_size, sequence_length]
            mask: 有效位置掩码，形状为 [batch_size, sequence_length]

        Returns:
            预测结果，形状为 [batch_size, sequence_length]
            在时刻 t 的输出预测的是 t+1 的标签
        """
        from torch.nn.functional import one_hot

        # 生成交互嵌入
        # 将概念ID和响应组合：c + num_c * r
        x = sequence + self.num_c * response
        xemb = self.interaction_emb(x)

        # LSTM前向传播
        h, _ = self.lstm_layer(xemb)
        # Dropout
        h = self.dropout_layer(h)
        # 线性层输出
        y = self.out_layer(h)

        # Sigmoid激活函数，将输出转换为概率
        y = torch.sigmoid(y)

        # 提取下一个时间步对应概念的预测
        target_concepts = sequence[:, 1:]
        # 使用 one-hot 编码选择对应概念的预测
        y = (y[:, :-1] * one_hot(target_concepts.long(), self.num_c)).sum(-1)
        # 约定：位置 t 预测 response[t+1]，末尾补零对齐到 [B, S]
        y = torch.cat([y, torch.zeros_like(y[:, :1])], dim=1)

        return y
