"""SKT 模型定义。

实现自论文：
    "Structure-based Knowledge Tracing: An Influence Propagation View"。

每道题目维护独立隐状态 [B, Q, H]，逐步经自影响 GRU 刷新当前题状态，再通过
同步（无向相似邻居 + 反射）与传播（有向后继）两种影响机制扩散到全题空间：
    s'_q = GRU_self(resp_emb(q, a), s_q)             当前题状态刷新
    sync = nb_mask · f_sync([S; broadcast(s'_q)])     相似邻居同步
    sync += self_mask · Σ_q sync                      当前点反射
    prop = sc_mask · f_prop(s'_q - s_q)               后继传播
    S'   = GRU(f_agg(α·sync + (1-α)·prop), S)         逐节点状态转移
    logit = out(S')                                   全题掌握度
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class SKT(nn.Module):
    """结构化知识追踪网络。

    Args:
        num_questions: 题目数量 Q。
        hidden_dim: 单题隐状态维度 H。
        latent_dim: 响应嵌入维度。
        concept_dim: 概念嵌入维度。
        alpha: 同步与传播影响的聚合权重。
        neighbor_adj: 无向相似图邻接 [Q, Q]（bool），同步作用范围。
        successor_adj: 有向转移图邻接 [Q, Q]（bool），传播作用范围。
        dropout: 输出层前的 dropout 概率。
        self_dropout: 自影响更新后的 dropout 概率。
        use_checkpoint: 逐步计算是否启用 gradient checkpointing。
    """

    def __init__(
        self,
        num_questions: int,
        hidden_dim: int,
        latent_dim: int,
        concept_dim: int,
        alpha: float,
        neighbor_adj: torch.Tensor,
        successor_adj: torch.Tensor,
        dropout: float = 0.0,
        self_dropout: float = 0.5,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.num_questions = num_questions
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.use_checkpoint = use_checkpoint

        # Index q * 2 + a encodes (question, correctness) jointly
        self.response_embedding = nn.Embedding(2 * num_questions, latent_dim)
        self.concept_embedding = nn.Embedding(num_questions, concept_dim)
        # Self influence: refresh the answered question's own state
        self.f_self = nn.GRUCell(latent_dim, hidden_dim)
        # Per-node state transition applied to every question slot
        self.rnn = nn.GRUCell(hidden_dim + concept_dim, hidden_dim)
        self.f_sync = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU())
        self.f_prop = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.f_agg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.self_dropout = nn.Dropout(self_dropout)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

        self.register_buffer("neighbor_adj", neighbor_adj.bool(), persistent=False)
        self.register_buffer("successor_adj", successor_adj.bool(), persistent=False)

    def _step(
        self, states: torch.Tensor, question: torch.Tensor, response: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单步影响传播。

        Args:
            states: 题目隐状态 [B, Q, H]。
            question: 当前题目 id [B]。
            response: 当前答案 [B]。

        Returns:
            (next_states [B, Q, H], 掌握度 logits [B, Q])。
        """
        batch_size, _, h = states.shape
        q_num = self.num_questions

        # Self influence on the answered question
        batch_idx = torch.arange(batch_size, device=states.device)
        self_state = states[batch_idx, question]  # [B, H]
        resp_emb = self.response_embedding(question * 2 + response)  # [B, latent]
        next_self = self.self_dropout(self.f_self(resp_emb, self_state))  # [B, H]

        self_mask = F.one_hot(question, q_num).to(states.dtype)  # [B, Q]
        nb_mask = self.neighbor_adj[question].to(states.dtype)  # [B, Q]
        sc_mask = self.successor_adj[question].to(states.dtype)  # [B, Q]

        # Synchronization across similar questions, then reflection on the vertex
        sync_diff = torch.cat(
            [states, next_self.unsqueeze(1).expand(batch_size, q_num, h)], dim=-1
        )  # [B, Q, 2H]
        sync_inf = nb_mask.unsqueeze(-1) * self.f_sync(sync_diff)  # [B, Q, H]
        reflection = sync_inf.sum(dim=1, keepdim=True)  # [B, 1, H]
        sync_inf = sync_inf + self_mask.unsqueeze(-1) * reflection

        # Propagation of the state change to successor questions
        prop_inf = self.f_prop(next_self - self_state)  # [B, H]
        prop_inf = sc_mask.unsqueeze(-1) * prop_inf.unsqueeze(1)  # [B, Q, H]

        # Aggregate both influence channels and update every question state
        inf = self.f_agg(self.alpha * sync_inf + (1.0 - self.alpha) * prop_inf)
        concept = self.concept_embedding.weight.unsqueeze(0).expand(
            batch_size, q_num, -1
        )
        rnn_in = torch.cat([inf, concept], dim=-1).reshape(batch_size * q_num, -1)
        next_states = self.rnn(rnn_in, states.reshape(batch_size * q_num, h))
        next_states = next_states.reshape(batch_size, q_num, h)

        logits = self.out(self.dropout(next_states)).squeeze(-1)  # [B, Q]
        return next_states, logits

    def forward(
        self, questions: torch.Tensor, responses: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """SKT 前向传播。

        Args:
            questions: 题目序列 [B, S]。
            responses: 答案序列 [B, S]。
            mask: 有效位置掩码 [B, S]（循环内不参与计算，由训练侧对齐过滤）。

        Returns:
            全题掌握度 logits [B, S, Q]，``logits[:, t, q]`` 为看完前 t+1 个
            交互后对题目 q 的预测（next-item 约定）。
        """
        seq_len = questions.shape[1]
        device = questions.device
        states = torch.zeros(
            questions.size(0), self.num_questions, self.hidden_dim, device=device
        )

        outputs = []
        for t in range(seq_len):
            step = self._step
            args = (states, questions[:, t], responses[:, t])
            if self.use_checkpoint and self.training:
                states, logits = checkpoint(step, *args, use_reentrant=False)
            else:
                states, logits = step(*args)
            outputs.append(logits)

        return torch.stack(outputs, dim=1)  # [B, S, Q]
