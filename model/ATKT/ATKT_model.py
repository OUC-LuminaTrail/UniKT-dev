"""ATKT (Adversarial Training based Knowledge Tracing) 模型实现

原始论文: Gupta et al., "Attentive Knowledge Tracing", AAAI 2021
"""

import torch
from torch import nn


class ATKT(nn.Module):
    """ATKT 模型

    LSTM + 累积注意力骨干，对交互嵌入施加 L2 归一化的对抗扰动以提升泛化性。

    Args:
        num_c: 概念（技能）数量
        skill_emb_dim: 概念嵌入维度
        answer_emb_dim: 响应嵌入维度
        hidden_dim: LSTM 隐藏维度
        attention_dim: 注意力中间层维度
    """

    def __init__(
        self,
        num_c: int,
        skill_emb_dim: int = 256,
        answer_emb_dim: int = 96,
        hidden_dim: int = 80,
        attention_dim: int = 80,
    ):
        super().__init__()
        self.num_c = num_c
        self.hidden_dim = hidden_dim

        self.skill_emb = nn.Embedding(num_c, skill_emb_dim)
        self.answer_emb = nn.Embedding(2, answer_emb_dim)

        self.rnn = nn.LSTM(skill_emb_dim + answer_emb_dim, hidden_dim, batch_first=True)

        self.mlp = nn.Linear(hidden_dim, attention_dim)
        self.similarity = nn.Linear(attention_dim, 1, bias=False)

        self.fc = nn.Linear(hidden_dim * 2, num_c)
        self.sig = nn.Sigmoid()

    def attention_module(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """累积注意力：位置 t 仅聚合 t 之前（不含 t）的注意力历史，防止泄漏。

        Args:
            lstm_output: LSTM 输出 [B, S, hidden_dim]

        Returns:
            拼接 (历史注意力累积, LSTM 输出) 的张量 [B, S, hidden_dim*2]
        """
        att_w = torch.tanh(self.mlp(lstm_output))
        att_w = self.similarity(att_w)

        alphas = torch.softmax(att_w, dim=1)

        attn_output = alphas * lstm_output
        attn_output_cum = torch.cumsum(attn_output, dim=1)
        attn_output_cum_1 = attn_output_cum - attn_output

        return torch.cat((attn_output_cum_1, lstm_output), 2)

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        perturbation: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        响应为 0/1 时交换技能与响应嵌入的拼接顺序，区分对错两类交互。

        Args:
            sequence: 概念ID序列，形状为 [batch_size, sequence_length]
            response: 响应序列，形状为 [batch_size, sequence_length]
            perturbation: 交互嵌入上的对抗扰动 [batch_size, sequence_length,
                skill_emb_dim + answer_dim]（可选，仅对抗训练分支使用）

        Returns:
            (preds, features):
                preds: 预测结果 [batch_size, sequence_length]，
                    preds[:, t] 基于 0..t-1 的交互预测 response[t]（同位对齐）
                features: 未加扰动的交互嵌入，作为对抗梯度的锚点
        """
        skill_embedding = self.skill_emb(sequence)
        answer_embedding = self.answer_emb(response)

        skill_answer = torch.cat((skill_embedding, answer_embedding), 2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), 2)

        answer = response.unsqueeze(2).expand_as(skill_answer)
        features = torch.where(answer == 1, skill_answer, answer_skill)

        rnn_input = features if perturbation is None else features + perturbation

        out, _ = self.rnn(rnn_input)
        out = self.attention_module(out)
        res = self.sig(self.fc(out))  # [B, S, num_c]

        # Next-skill prediction: mastery at t dot one-hot of skill at t+1
        pred_next = res[:, :-1].gather(-1, sequence[:, 1:].unsqueeze(-1)).squeeze(-1)
        # Pad a leading placeholder so preds[:, t] predicts response[t]
        preds = torch.cat([torch.zeros_like(pred_next[:, :1]), pred_next], dim=1)

        return preds, features
