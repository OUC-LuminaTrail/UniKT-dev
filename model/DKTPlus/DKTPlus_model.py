"""DKT+ 模型实现"""

import torch
import torch.nn.functional as F
from torch import nn


class DKTPlus(nn.Module):
    """DKT+ 模型

    Args:
        num_c: 概念数量
        emb_size: 嵌入维度
        lambda_r: loss_r 权重
        lambda_w1: loss_w1 权重
        lambda_w2: loss_w2 权重
        dropout: Dropout 概率
    """

    def __init__(
        self,
        num_c: int,
        emb_size: int,
        lambda_r: float,
        lambda_w1: float,
        lambda_w2: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2

        self.interaction_emb = nn.Embedding(self.num_c * 2, self.emb_size)
        self.lstm_layer = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.hidden_size, self.num_c)

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            sequence: 概念ID序列 [B, S]
            response: 响应序列 [B, S]
            mask: 有效位置掩码 [B, S]

        Returns:
            y_hat: next-item 同位置预测 [B, S]，``y_hat[:, t]`` 预测 ``response[t]``
            reg_loss: 已加权的 DKT+ 三项正则化损失标量
        """
        # 交互嵌入：c + num_c * r
        x = sequence + self.num_c * response
        xemb = self.interaction_emb(x)

        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = torch.sigmoid(self.out_layer(h))

        y_hat = self._next_item_predict(y, sequence)  # [B, S]
        reg_loss = self._regularization_loss(y, sequence, response, mask)
        return y_hat, reg_loss

    def _next_item_predict(
        self, y: torch.Tensor, sequence: torch.Tensor
    ) -> torch.Tensor:
        """从全概念矩阵 y [B, S, num_c] 取 next-item 预测 [B, S]。"""
        target = sequence[:, 1:].long().unsqueeze(-1)  # [B, S-1, 1]
        y_next = y[:, :-1].gather(-1, target).squeeze(-1)  # [B, S-1]
        return torch.cat([torch.zeros_like(y_next[:, :1]), y_next], dim=1)  # [B, S]

    def _regularization_loss(
        self,
        y: torch.Tensor,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """DKT+ 三项正则化损失。"""
        num_c = self.num_c
        y_seq = y[:, :-1]  # [B, S-1, num_c]
        valid = mask[:, :-1] & mask[:, 1:]  # [B, S-1]

        y_curr = y_seq.gather(-1, sequence[:, :-1].long().unsqueeze(-1)).squeeze(
            -1
        )  # [B, S-1]
        r_curr = response.float()[:, :-1]
        loss_r = F.binary_cross_entropy(
            torch.masked_select(y_curr, valid).double(),
            torch.masked_select(r_curr, valid).double(),
        )

        # loss_w1 / loss_w2 为相邻时刻输出矩阵差
        diff = y_seq[:, 1:] - y_seq[:, :-1]  # [B, S-2, num_c]
        smooth_mask = valid[:, 1:]  # [B, S-2]
        w1 = torch.masked_select(diff.abs().sum(-1), smooth_mask).mean() / num_c
        w2 = torch.masked_select(diff.pow(2).sum(-1), smooth_mask).mean() / num_c

        return self.lambda_r * loss_r + self.lambda_w1 * w1 + self.lambda_w2 * w2
