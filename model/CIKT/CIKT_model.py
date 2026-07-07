"""CIKT: Disentangling Response Sequences with Causal Invariance for Knowledge Tracing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """单层图卷积：output = LayerNorm(adj @ (x W))."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.layer_norm = nn.LayerNorm(out_features, elementwise_affine=False)

    def forward(self, input, adj):
        support = self.weight(input)
        output = torch.bmm(adj, support)
        return self.layer_norm(output)


class GCN(nn.Module):
    """多层 GCN，每层接 ReLU + Dropout。"""

    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers > 0:
            self.layers.append(GraphConvolution(input_size, hidden_size))
            for _ in range(num_layers - 1):
                self.layers.append(GraphConvolution(hidden_size, hidden_size))
            self.layers.append(GraphConvolution(hidden_size, num_classes))
        else:
            self.layers.append(GraphConvolution(input_size, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        for layer in self.layers:
            x = self.dropout(F.relu(layer(x, adj)))
        return x


class AttentionScoreQcWeight(nn.Module):
    """问题-概念联合权重：attention = (Q·K^T)(Q_c·K_c^T)，缩放后逐元素相乘。"""

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_q1 = nn.Linear(hidden_size, hidden_size)
        self.linear_k1 = nn.Linear(hidden_size, hidden_size)
        self.linear_q2 = nn.Linear(hidden_size, hidden_size)
        self.linear_k2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, ques_state, conc_state):
        q1 = self.linear_q1(ques_state[:, 1:].contiguous())
        k1 = self.linear_k1(ques_state[:, :-1].contiguous())
        q2 = self.linear_q2(conc_state[:, 1:].contiguous())
        k2 = self.linear_k2(conc_state[:, :-1].contiguous())
        scaling = self.hidden_size**-0.5
        attention1 = torch.bmm(q1, k1.transpose(-2, -1)) * scaling
        attention2 = torch.bmm(q2, k2.transpose(-2, -1)) * scaling
        return attention1 * attention2


class AttentionScoreCausal(nn.Module):
    """因果注意力分数（标准 softmax）。"""

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=-1)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, attn_mask, key_padding_mask, qc_score):
        q = self.linear_q(q)
        k = self.linear_k(k)
        scaling = self.hidden_size**-0.5
        attention = torch.bmm(q, k.transpose(-2, -1)) * scaling
        attention = attention * qc_score
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            attention = attention.masked_fill(
                key_padding_mask.unsqueeze(1), float("-inf")
            )
        return self.softmax(attention)


class AttentionScoreTrivial(nn.Module):
    """平凡注意力分数：1 - sigmoid(score)，再 softmax。含可学习缩放 a/b。"""

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=-1)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.FloatTensor([10]))
        self.b = nn.Parameter(torch.FloatTensor([0.1]))

    def forward(self, q, k, attn_mask, key_padding_mask, qc_score):
        q = self.linear_q(q)
        k = self.linear_k(k)
        scaling = self.hidden_size**-0.5
        attention = torch.bmm(q, k.transpose(-2, -1)) * scaling
        attention = attention * qc_score
        attention = 1 - torch.sigmoid(attention)
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            attention = attention.masked_fill(
                key_padding_mask.unsqueeze(1), float("-inf")
            )
        return self.softmax(attention)


class DisentangleCausal(nn.Module):
    """因果/平凡解耦：对每个 (query, key) 用 Gumbel-softmax 二选一分配权重。"""

    def __init__(self, hidden_size, seq_len, tau=1, is_hard=True, dropout_p=0):
        super().__init__()
        self.tau = tau
        self.is_hard = is_hard
        self.causal_att = AttentionScoreCausal(hidden_size, dropout_p)
        self.trivial_att = AttentionScoreTrivial(hidden_size, dropout_p)
        self.qc_causal = AttentionScoreQcWeight(hidden_size, dropout_p)
        self.qc_trivial = AttentionScoreQcWeight(hidden_size, dropout_p)

    def forward(
        self, q_state, x_state, attn_mask, key_padding_mask, ques_state, conc_state
    ):
        qc_causal_score = self.qc_causal(ques_state, conc_state)
        qc_trivial_score = self.qc_trivial(ques_state, conc_state)
        causal_score = self.causal_att(
            q_state, x_state, attn_mask, key_padding_mask, qc_causal_score
        )
        trivial_score = self.trivial_att(
            q_state, x_state, attn_mask, key_padding_mask, qc_trivial_score
        )
        score = torch.cat(
            (causal_score.unsqueeze(2), trivial_score.unsqueeze(2)), dim=2
        )
        score = F.gumbel_softmax(score, tau=self.tau, hard=self.is_hard, dim=2)
        causal_mask = score[:, :, 0, :].masked_fill(attn_mask, 0.0)
        trivial_mask = score[:, :, 1, :].masked_fill(attn_mask, 0.0)
        return causal_mask, trivial_mask


class EncoderEmbedding(nn.Module):
    """问题 / 作答 / 概念联合嵌入，按 pattern 输出不同视图。"""

    def __init__(self, q_num, concept_num, d_model):
        super().__init__()
        self.exercise_embed = nn.Embedding(q_num, d_model)
        self.response_embed = nn.Embedding(2, d_model)
        self.concept_embed = nn.Embedding(concept_num, d_model)

    def forward(
        self,
        exercises,
        pattern,
        response=None,
        concept=None,
        causal_mask=None,
        trivial_mask=None,
        qr_response=None,
    ):
        e = self.exercise_embed(exercises)
        if pattern == "only_q":
            return e + self.concept_embed(concept)
        if pattern == "x_state":
            return e + self.response_embed(response) + self.concept_embed(concept)
        if pattern == "qc_embed":
            return torch.cat([e, self.concept_embed(concept)], dim=-2)
        if pattern == "x_reversal":
            e = e + self.concept_embed(concept)
            r = self.response_embed(response)
            r2 = self.response_embed(1 - response)
            return (
                e.unsqueeze(1)
                + r.unsqueeze(1) * causal_mask.unsqueeze(-1)
                + r2.unsqueeze(1) * trivial_mask.unsqueeze(-1)
            )
        if pattern == "q_replace":
            r = self.response_embed(response) + self.concept_embed(concept)
            e_replace = self.exercise_embed(qr_response)
            return (
                r.unsqueeze(1)
                + e.unsqueeze(1) * causal_mask.unsqueeze(-1)
                + e_replace.unsqueeze(1) * trivial_mask.unsqueeze(-1)
            )
        raise ValueError(f"Unknown embedding pattern: {pattern}")


class PredictHead(nn.Module):
    """二分类预测头：sigmoid(MLP(concat(X, Q)))。"""

    def __init__(self, d_model):
        super().__init__()
        self.out_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, q):
        return self.out_fc(torch.cat([x, q], dim=-1)).squeeze(-1)


class PredictHeadClassifier(nn.Module):
    """难度等级分类头：输出 nd 个类别 logits（用于平凡分支的干预监督）。"""

    def __init__(self, d_model, num_classes):
        super().__init__()
        self.out_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, num_classes),
        )

    def forward(self, x, q):
        return self.out_fc(torch.cat([x, q], dim=-1))


class CIKT(nn.Module):
    """CIKT 主模型。"""

    def __init__(
        self,
        num_questions,
        num_concepts,
        d_model,
        seq_len,
        dropout=0.5,
        num_difficulty_levels=10,
        difficulty_table=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.length = int(seq_len)
        self.seq_len = int(seq_len) - 1  # response window length
        self.num_difficulty_levels = num_difficulty_levels

        # static masks / adjacency
        fixed_edge = self._build_fixed_edge(self.length)
        self.register_buffer("fixed_edge", fixed_edge, persistent=False)
        att_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).to(
            dtype=torch.bool
        )
        self.register_buffer("att_mask", att_mask, persistent=False)
        att_mask_num = 1 - torch.triu(
            torch.ones(self.seq_len, self.seq_len), diagonal=1
        )
        self.register_buffer("att_mask_num", att_mask_num, persistent=False)
        one_mask = torch.triu(torch.ones(self.length, self.length), diagonal=1).to(
            dtype=torch.bool
        )
        all_mask = torch.cat([torch.cat([one_mask, one_mask], dim=-1)] * 2, dim=0)
        self.register_buffer("all_mask", all_mask, persistent=False)

        # modules
        self.gcn = GCN(d_model, d_model, d_model, num_layers=2, dropout=dropout)
        self.x_encoder_num_layers = 2
        self.q_encoder_num_layers = 2
        self.rnn_q = nn.LSTM(
            d_model, d_model, num_layers=self.q_encoder_num_layers, batch_first=True
        )
        self.rnn = nn.LSTM(
            d_model, d_model, num_layers=self.x_encoder_num_layers, batch_first=True
        )
        self.disentanglement = DisentangleCausal(
            hidden_size=d_model,
            seq_len=self.seq_len,
            tau=1,
            is_hard=True,
            dropout_p=0,
        )
        self.encoder_embedding = EncoderEmbedding(
            q_num=num_questions, concept_num=num_concepts, d_model=d_model
        )
        self.pred_causal = PredictHead(d_model)
        self.pred_trivial = PredictHeadClassifier(d_model, num_difficulty_levels)
        self.pred_intervention = PredictHead(d_model)
        self.pred_replace = PredictHead(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # data-derived lookup tables
        if difficulty_table is not None:
            self.register_buffer("difficulty_table", difficulty_table, persistent=False)

    @staticmethod
    def _build_fixed_edge(length):
        """构造 2L 节点的固定图邻接。

        节点布局：``[0, L)`` 题目节点，``[L, 2L)`` 概念节点。
        """
        q_src = list(range(0, length - 1))
        q_tgt = list(range(1, length))
        c_src = list(range(length, length + length - 1))
        c_tgt = list(range(length + 1, length + length))
        src = q_src + c_src + q_src + c_src
        tgt = q_tgt + c_tgt + c_tgt + q_tgt
        # to_undirected: 追加反向边
        all_src = src + tgt
        all_tgt = tgt + src
        adj = torch.zeros(2 * length, 2 * length, dtype=torch.float32)
        adj[all_src, all_tgt] = 1.0
        return adj

    def _zero_lstm_states(self, batch_size, device, dtype):
        """构造 LSTM 的零初始 (h_0, c_0)，形状 ``[num_layers, B, d]``。"""
        h = torch.zeros(
            self.x_encoder_num_layers,
            batch_size,
            self.d_model,
            device=device,
            dtype=dtype,
        )
        c = torch.zeros_like(h)
        return h, c

    def _zero_q_states(self, batch_size, device, dtype):
        h = torch.zeros(
            self.q_encoder_num_layers,
            batch_size,
            self.d_model,
            device=device,
            dtype=dtype,
        )
        c = torch.zeros_like(h)
        return h, c

    def _expand_and_average(self, embed_masked, batch_size):
        """对 ``[B, L-1, L-1, d]`` 的掩码嵌入跑 LSTM 后做因果窗口平均。

        Returns ``[B, L-1, d]``。
        """
        device, dtype = embed_masked.device, embed_masked.dtype
        seq_len = self.seq_len
        x = embed_masked.reshape(batch_size * seq_len, seq_len, self.d_model)
        h0, c0 = self._zero_lstm_states(batch_size * seq_len, device, dtype)
        x_state, _ = self.rnn(x, (h0, c0))
        x_state = x_state.view(batch_size, seq_len, seq_len, self.d_model)
        x_state = x_state * self.att_mask_num.unsqueeze(0).unsqueeze(-1)
        return torch.sum(x_state, dim=-2) / torch.sum(
            self.att_mask_num, dim=-1
        ).unsqueeze(-1)

    def forward(self, q, y, c, qr, mask):
        """前向传播。

        Args:
            q: ``[B, L]`` 题目 id。
            y: ``[B, L]`` 作答 (0/1)。
            c: ``[B, L]`` 概念 id。
            qr: ``[B, L]`` 同概念替换题目 id。
            mask: ``[B, L]`` 有效位置掩码。

        Returns:
            dict: ``y_pred`` (融合预测, ``[B, L-1]``), ``y_causal`` / ``y_intervention`` /
            ``y_replace`` (各 ``[B, L-1]``), ``y_trivial`` (``[B, L-1, nd]``)。
        """
        q = q.int()
        y = y.int()
        c = c.int()
        qr = qr.int()

        q_predict = q[:, 1:].contiguous()
        c_predict = c[:, 1:].contiguous()
        q_response = q[:, :-1].contiguous()
        y_response = y[:, :-1].contiguous()
        c_response = c[:, :-1].contiguous()
        qr_response = qr[:, :-1].contiguous()

        batch_size = q.size(0)
        padding_response = mask[:, :-1] == 0

        # Embedding
        x_embed = self.encoder_embedding(
            exercises=q_response,
            response=y_response,
            concept=c_response,
            pattern="x_state",
        )
        q_embed = self.encoder_embedding(
            exercises=q_predict, concept=c_predict, pattern="only_q"
        )

        # Graph node & edge
        c_node = torch.cat([c, c], dim=-1)
        dynamic_edge = (c_node.unsqueeze(-1) - c_node.unsqueeze(-2) == 0).to(
            dtype=torch.float32
        )
        graph_edge = self.fixed_edge + dynamic_edge
        graph_edge = graph_edge.masked_fill(self.all_mask, 0.0)
        qc_embed = self.encoder_embedding(exercises=q, concept=c, pattern="qc_embed")
        qc_state = self.gcn(qc_embed, graph_edge)
        ques_state = qc_state[:, : self.length, :]
        conc_state = qc_state[:, self.length :, :]

        # Attention / disentanglement
        q_state, _ = self.rnn_q(
            q_embed, self._zero_q_states(batch_size, q_embed.device, q_embed.dtype)
        )
        x_state, _ = self.rnn(
            x_embed, self._zero_lstm_states(batch_size, x_embed.device, x_embed.dtype)
        )
        causal_mask, trivial_mask = self.disentanglement(
            q_state=q_state,
            x_state=x_state,
            attn_mask=self.att_mask,
            key_padding_mask=padding_response,
            ques_state=ques_state,
            conc_state=conc_state,
        )

        # Causal encoding
        x_embed_causal = x_embed.unsqueeze(1) * causal_mask.unsqueeze(-1)
        x_causal_ave = self._expand_and_average(x_embed_causal, batch_size)

        # Intervention: remove
        x_embed_trivial = x_embed.unsqueeze(1) * trivial_mask.unsqueeze(-1)
        x_trivial_ave = self._expand_and_average(x_embed_trivial, batch_size)

        # Intervention: invert
        x_embed_intervention = self.encoder_embedding(
            exercises=q_response,
            response=y_response,
            concept=c_response,
            causal_mask=causal_mask,
            trivial_mask=trivial_mask,
            pattern="x_reversal",
        )
        x_intervention_ave = self._expand_and_average(x_embed_intervention, batch_size)

        # Intervention: replace
        x_embed_replace = self.encoder_embedding(
            exercises=q_response,
            response=y_response,
            concept=c_response,
            causal_mask=causal_mask,
            trivial_mask=trivial_mask,
            pattern="q_replace",
            qr_response=qr_response,
        )
        x_replace_ave = self._expand_and_average(x_embed_replace, batch_size)

        # Predict
        # 对 intervention/replace 复用 norm1，trivial 用 norm2。
        x_causal_ave = self.norm1(x_causal_ave)
        x_trivial_ave = self.norm2(x_trivial_ave)
        x_intervention_ave = self.norm1(x_intervention_ave)
        x_replace_ave = self.norm1(x_replace_ave)

        y_causal = self.pred_causal(x_causal_ave, q_state)
        y_trivial = self.pred_trivial(x_trivial_ave, q_state)
        y_intervention = self.pred_intervention(x_intervention_ave, q_state)
        y_replace = self.pred_replace(x_replace_ave, q_state)

        y_pred = (y_causal + y_intervention + y_replace) / 3
        return {
            "y_pred": y_pred,
            "y_causal": y_causal,
            "y_intervention": y_intervention,
            "y_replace": y_replace,
            "y_trivial": y_trivial,
        }
