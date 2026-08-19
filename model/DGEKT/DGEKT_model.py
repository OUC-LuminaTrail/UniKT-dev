"""DGEKT 模型定义。

实现自论文：
    "DGEKT: A Dual Graph Ensemble Learning Method for Knowledge Tracing"。

交互节点空间为 2Q（答对题 q = q，答错题 q = q+Q），冻结的随机初始特征经
两条图卷积分支增强后查表作为 GRU 输入：
    ques_h = HGNN(ques, G)                          概念关联超图分支
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
    """HGNN 层：``G @ (x W + b)``。"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        stdv = 1.0 / math.sqrt(out_features)
        nn.init.uniform_(self.weight, -stdv, stdv)
        nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(G, x.matmul(self.weight) + self.bias)


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

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return F.relu(self.hgc2(F.relu(self.hgc1(x, G)), G))


class DGEKT(nn.Module):
    """双图集成知识追踪网络。

    Args:
        num_questions: 题目数 Q。
        emb_dim: 交互嵌入维度（须为偶数，转移分支各 GCN 输出一半）。
        hidden_dim: GRU 隐藏维度。
        num_layers: GRU 层数。
        hyper_g: 超图拉普拉斯稀疏张量 [2Q, 2Q]。
        adj_out: 出转移邻接稀疏张量 [2Q, 2Q]。
        adj_in: 入转移邻接稀疏张量 [2Q, 2Q]。
    """

    def __init__(
        self,
        num_questions: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        hyper_g: torch.Tensor,
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
        # Static graphs are deterministically rebuilt from data each run.
        self.register_buffer("hyper_g", hyper_g, persistent=False)
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
        """前向传播。

        Args:
            question: 题目序列 [B, S]。
            response: 答案序列 [B, S]。
            mask: 有效位置掩码 [B, S]。

        Returns:
            (logit_c, logit_t, logit_e) 三路 logits，各 [B, S, Q]。
        """
        # Graph convolutions rerun every forward so gradients reach their weights.
        ques_h = self.hgnn(self.interaction_feat, self.hyper_g)
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
        logit_c = self.fc_c(out_c)
        logit_t = self.fc_t(out_t)
        theta = torch.sigmoid(self.gate_w1(out_c) + self.gate_w2(out_t))
        logit_e = self.fc_ensemble(
            torch.cat([theta * out_t, (1 - theta) * out_c], dim=-1)
        )
        return logit_c, logit_t, logit_e
