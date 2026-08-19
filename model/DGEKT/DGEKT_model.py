"""DGEKT 模型定义。

实现自论文：
    "DGEKT: A Dual Graph Ensemble Learning Method for Knowledge Tracing"。

交互节点空间为 2Q（答对题 q = q，答错题 q = q+Q），冻结的随机初始特征经
两条图卷积分支增强后查表作为 GRU 输入：
    P = Dv^-1/2·H·De^-1, Q = H^T·Dv^-1/2   超图拉普拉斯因子，G = P·Q
    ques_h = P @ (Q @ ...)                 概念关联超图分支
    ques_d = [GCN(ques, adj_out); GCN(ques, adj_in)] 有向转移图分支
    logit_c = fc_c(GRU_c(lookup(ques_h)))
    logit_t = fc_t(GRU_t(lookup(ques_d)))
    θ       = σ(w1·out_c + w2·out_t)
    logit_e = fc_ensemble([θ·out_t, (1-θ)·out_c])   门控融合 ensemble 路
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class _GraphConvolution(nn.Module):
    """有向 GCN 层：``adj @ (x W) + b``。"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        stdv = 1.0 / math.sqrt(out_features)
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj, x.matmul(self.weight)) + self.bias


class _HypergraphConvolution(nn.Module):
    """HGNN 层：``P @ (Q @ (x W + b))``"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        stdv = 1.0 / math.sqrt(out_features)
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(
        self, x: torch.Tensor, p: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        return torch.sparse.mm(p, torch.sparse.mm(q, x.matmul(self.weight) + self.bias))


class _GCN(nn.Module):
    """两层 GCN，层间与末层 ReLU。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.gc1 = _GraphConvolution(in_dim, hidden_dim)
        self.gc2 = _GraphConvolution(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gc2(F.relu(self.gc1(x, adj)), adj))


class _HGNN(nn.Module):
    """两层 HGNN，层间与末层 ReLU。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.hgc1 = _HypergraphConvolution(in_dim, hidden_dim)
        self.hgc2 = _HypergraphConvolution(hidden_dim, out_dim)

    def forward(
        self, x: torch.Tensor, p: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        return F.relu(self.hgc2(F.relu(self.hgc1(x, p, q)), p, q))


class DGEKT(nn.Module):
    """双图集成知识追踪网络。

    Args:
        num_questions: 题目数 Q。
        emb_dim: 交互嵌入维度（须为偶数，转移分支各 GCN 输出一半）。
        hidden_dim: GRU 隐藏维度。
        num_layers: GRU 层数。
        hyper_factors: 超图拉普拉斯因子 (P, Q)，P = Dv^-1/2·H·De^-1
            [2Q, 2S]，Q = H^T·Dv^-1/2 [2S, 2Q]，G = P·Q。
        adj_out: 出转移邻接稀疏张量 [2Q, 2Q]。
        adj_in: 入转移邻接稀疏张量 [2Q, 2Q]。
    """

    def __init__(
        self,
        num_questions: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        hyper_factors: tuple[torch.Tensor, torch.Tensor],
        adj_out: torch.Tensor,
        adj_in: torch.Tensor,
    ):
        super().__init__()
        if emb_dim % 2 != 0:
            raise ValueError(f"emb_dim must be even, got {emb_dim}")
        self.num_questions = num_questions
        # Frozen random interaction features; persisted so checkpoints stay
        # self-contained (rebuilding draws a different random matrix).
        self.register_buffer(
            "interaction_feat", torch.randn(2 * num_questions, emb_dim)
        )
        hyper_p, hyper_q = hyper_factors
        # Static graphs are deterministically rebuilt from data each run.
        self.register_buffer("hyper_p", hyper_p, persistent=False)
        self.register_buffer("hyper_q", hyper_q, persistent=False)
        self.register_buffer("adj_out", adj_out, persistent=False)
        self.register_buffer("adj_in", adj_in, persistent=False)
        self.hgnn = _HGNN(emb_dim, emb_dim, emb_dim)
        self.gcn_out = _GCN(emb_dim, emb_dim, emb_dim // 2)
        self.gcn_in = _GCN(emb_dim, emb_dim, emb_dim // 2)
        self.rnn_c = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.rnn_t = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_c = nn.Linear(hidden_dim, num_questions)
        self.fc_t = nn.Linear(hidden_dim, num_questions)
        self.fc_ensemble = nn.Linear(2 * hidden_dim, num_questions)
        self.gate_w1 = nn.Linear(hidden_dim, hidden_dim)
        self.gate_w2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, question: torch.Tensor, response: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播至隐状态层。

        Args:
            question: 题目序列 [B, S]。
            response: 答案序列 [B, S]。
            mask: 有效位置掩码 [B, S]。

        Returns:
            (out_c, out_t, gate_in)：concept / transition 路 GRU 隐状态
            [B, S, hidden_dim] 与 ensemble 路门控拼接特征
            [B, S, 2*hidden_dim]；输出头由 :meth:`head_logits` /
            :meth:`target_logits` 在选定步上应用。
        """
        # Graph convolutions rerun every forward so gradients reach their weights.
        ques_h = self.hgnn(self.interaction_feat, self.hyper_p, self.hyper_q)
        ques_d = torch.cat(
            [
                self.gcn_out(self.interaction_feat, self.adj_out),
                self.gcn_in(self.interaction_feat, self.adj_in),
            ],
            dim=-1,
        )

        inter_id = question + (1 - response) * self.num_questions
        keep = mask.unsqueeze(-1).to(ques_h.dtype)  # padded steps feed zero vectors
        x_c = F.embedding(inter_id, ques_h) * keep
        x_t = F.embedding(inter_id, ques_d) * keep

        out_c, _ = self.rnn_c(x_c)
        out_t, _ = self.rnn_t(x_t)
        theta = torch.sigmoid(self.gate_w1(out_c) + self.gate_w2(out_t))
        gate_in = torch.cat([theta * out_t, (1 - theta) * out_c], dim=-1)
        return out_c, out_t, gate_in

    def head_logits(
        self,
        h_c: torch.Tensor,
        h_t: torch.Tensor,
        h_e: torch.Tensor,
        rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """选定步上的全词表三路 logits。

        Args:
            h_c / h_t: 隐状态 [B, S, hidden_dim]。
            h_e: 门控拼接特征 [B, S, 2*hidden_dim]。
            rows: 步骤选择掩码 [B, S]。

        Returns:
            (logit_c, logit_t, logit_e)，各 [V, num_questions]。
        """
        return (
            self.fc_c(h_c[rows]),
            self.fc_t(h_t[rows]),
            self.fc_ensemble(h_e[rows]),
        )

    def target_logits(
        self,
        h_c: torch.Tensor,
        h_t: torch.Tensor,
        h_e: torch.Tensor,
        q_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """选定步上目标题目的三路 logit。

        每行与其目标题的头权重行点积，等价于从全词表 logits 中
        gather 目标列。

        Args:
            h_c / h_t: 选定步隐状态 [V, hidden_dim]。
            h_e: 选定步门控特征 [V, 2*hidden_dim]。
            q_idx: 各步目标题索引 [V]。

        Returns:
            (logit_c, logit_t, logit_e)，各 [V]。
        """
        return (
            (h_c * self.fc_c.weight[q_idx]).sum(dim=-1) + self.fc_c.bias[q_idx],
            (h_t * self.fc_t.weight[q_idx]).sum(dim=-1) + self.fc_t.bias[q_idx],
            (h_e * self.fc_ensemble.weight[q_idx]).sum(dim=-1)
            + self.fc_ensemble.bias[q_idx],
        )
