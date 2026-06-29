"""GIKT: Yang et al. ECML-PKDD 2020。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import GeneralInteraction, HistoryRecap


class SumAggregator(nn.Module):
    """GraphSAGE mean 聚合：output = act(W·(self + mean(neighbors)) + b)。"""

    def __init__(self, dim, dropout=0.0, act=torch.tanh):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.act = act
        self.weights = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, self_vectors, neighbor_vectors, batch_size, seq_len):
        neighbors_agg = torch.mean(neighbor_vectors, dim=-2)
        output = (self_vectors + neighbors_agg).reshape(-1, self.dim)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = output @ self.weights + self.bias
        return self.act(output.reshape(batch_size, seq_len, -1, self.dim))


class ConcatAggregator(nn.Module):
    """GraphSAGE concat 聚合：output = act(W·concat(self, mean(neighbors)) + b)。"""

    def __init__(self, dim, dropout=0.0, act=torch.tanh):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.act = act
        self.weights = nn.Parameter(torch.empty(dim * 2, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, self_vectors, neighbor_vectors, batch_size, seq_len):
        neighbors_agg = torch.mean(neighbor_vectors, dim=-2)
        output = torch.cat([self_vectors, neighbors_agg], dim=-1).reshape(
            -1, self.dim * 2
        )
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = output @ self.weights + self.bias
        return self.act(output.reshape(batch_size, seq_len, -1, self.dim))


class GIKTGraphAggregator(nn.Module):
    """问题-技能图上的多跳采样聚合（get_neighbors / aggregate / next_neighbor_sampler）。"""

    def __init__(
        self,
        embedding_dim,
        question_neighbor_num,
        skill_neighbor_num,
        n_hop=3,
        dropout_probs=None,
        aggregator="sum",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.question_neighbor_num = question_neighbor_num
        self.skill_neighbor_num = skill_neighbor_num
        self.n_hop = n_hop
        self.dropout_gnn = (dropout_probs or [0.2, 0.2, 0.0])[1]

        if aggregator not in {"sum", "concat"}:
            raise ValueError("aggregator must be 'sum' or 'concat'")
        aggregator_cls = SumAggregator if aggregator == "sum" else ConcatAggregator

        # 每跳一个 aggregator，在其层内的内层 hop 中复用
        self.aggregators = nn.ModuleList(
            [
                aggregator_cls(self.embedding_dim, self.dropout_gnn, torch.tanh)
                for _ in range(n_hop)
            ]
        )

    def get_neighbors(self, n_hop, question_index, question_neighbors, skill_neighbors):
        """逐跳采样邻居 id。question_index: [B, S]（题目节点 id，已 +num_skills）。"""
        _, max_step = question_index.shape
        seeds = [question_index]
        for i in range(n_hop):
            # 偶数跳 题目→技能(question_neighbors)，奇数跳 技能→题目(skill_neighbors)
            table = question_neighbors if i % 2 == 0 else skill_neighbors
            num = self.question_neighbor_num if i % 2 == 0 else self.skill_neighbor_num
            # -1 维吸收多跳邻居乘积，与原 TF reshape 一致
            neighbor = table[seeds[i].reshape(-1)].reshape(-1, max_step, num)
            seeds.append(neighbor)
        return seeds

    def aggregate(self, input_neighbors, embedding_table):
        """由外向内逐层聚合。"""
        batch_size, max_step = input_neighbors[0].shape
        sq = [
            embedding_table[n.reshape(-1)].reshape(
                batch_size, max_step, -1, self.embedding_dim
            )
            for n in input_neighbors
        ]

        for i in range(self.n_hop):
            aggregator = self.aggregators[i]
            for hop in range(self.n_hop - i):
                num = (
                    self.question_neighbor_num
                    if hop % 2 == 0
                    else self.skill_neighbor_num
                )
                shape = [batch_size, max_step, -1, num, self.embedding_dim]
                sq[hop] = aggregator(
                    sq[hop], sq[hop + 1].reshape(shape), batch_size, max_step
                )
        return sq

    def sample_next_neighbors(self, next_aggregate_embedding, num_samples):
        """从下一题的 1 跳聚合邻居中随机采样 num_samples 个。"""
        temp_emb = next_aggregate_embedding[1]
        batch_size, seq_len, _, emb_dim = temp_emb.shape
        temp_emb = temp_emb.reshape(-1, self.question_neighbor_num, emb_dim).transpose(
            0, 1
        )
        temp_emb = temp_emb[
            torch.randperm(temp_emb.shape[0], device=temp_emb.device)
        ].transpose(0, 1)
        if self.question_neighbor_num >= num_samples:
            return temp_emb[:, :num_samples, :].reshape(
                batch_size, seq_len, num_samples, emb_dim
            )
        repeat = -(-num_samples // self.question_neighbor_num)
        return temp_emb.repeat(1, repeat, 1)[:, :num_samples, :].reshape(
            batch_size, seq_len, num_samples, emb_dim
        )

    def forward(self, question_indices, next_question_indices, graph_data):
        self.batch_size, self.max_step = question_indices.shape
        embedding_table = graph_data["feature_embedding"]
        aggregate_embedding = self.aggregate(
            self.get_neighbors(
                self.n_hop,
                question_indices,
                graph_data["question_neighbors"],
                graph_data["skill_neighbors"],
            ),
            embedding_table,
        )
        next_aggregate_embedding = self.aggregate(
            self.get_neighbors(
                self.n_hop,
                next_question_indices,
                graph_data["question_neighbors"],
                graph_data["skill_neighbors"],
            ),
            embedding_table,
        )
        return aggregate_embedding, next_aggregate_embedding


class GIKT(nn.Module):
    """GIKT 主模型。"""

    def __init__(self, args, data_metadata, **kwargs):
        super().__init__(**kwargs)
        self.num_skills = data_metadata["num_skills"]
        self.num_questions = data_metadata["num_questions"]
        self.embedding_dim = args.embedding_dim
        self.hidden_neurons = list(args.hidden_neurons)
        self.hidden_size = self.hidden_neurons[-1]
        self.dropout_prob = list(args.dropout_probs)[0]
        self.model_name = getattr(args, "variant", "hsei")
        self.sim_emb = getattr(args, "sim_emb", "question_emb")
        self.hist_neighbor_num = args.hist_neighbor_num
        self.next_neighbor_num = args.next_neighbor_num
        self.n_hop = args.n_hop

        # 内积要求 LSTM 末层输出 == 嵌入维度
        assert self.hidden_size == self.embedding_dim, (
            f"hidden_neurons[-1]({self.hidden_size}) must equal embedding_dim({self.embedding_dim})"
        )

        # 共享嵌入表：[0, num_skills) 技能，[num_skills, +num_questions) 题目，末 2 行作答
        self.feature_embedding = nn.Embedding(
            self.num_skills + self.num_questions + 2, self.embedding_dim
        )

        self.graph_aggregator = GIKTGraphAggregator(
            self.embedding_dim,
            args.question_neighbor_num,
            args.skill_neighbor_num,
            self.n_hop,
            list(args.dropout_probs),
            args.aggregator,
        )
        # feature_layer 当前题/下一题共用（对应原 TF reuse=True）
        self.feature_layer = nn.Linear(self.embedding_dim, self.hidden_size)
        self.feature_layer_act = nn.ReLU()
        self.input_trans_layer = nn.Linear(
            self.hidden_size + self.embedding_dim, self.hidden_size
        )

        # 链式 LSTM 实现 MultiRNNCell（各层尺寸可不同，如 [200,100]）
        sizes = [self.embedding_dim] + self.hidden_neurons
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(sizes[i], sizes[i + 1], batch_first=True)
                for i in range(len(self.hidden_neurons))
            ]
        )

        self.history_recap = HistoryRecap(
            self.hist_neighbor_num, getattr(args, "att_bound", 0.7)
        )
        self.general_interaction = GeneralInteraction(self.hidden_size)

    def _run_lstm(self, x):
        """逐层 LSTM，每层输出接 dropout（对应 DropoutWrapper）。"""
        drop_p = self.dropout_prob
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = F.dropout(x, p=drop_p, training=self.training)
        return x

    def _hist_neighbor_sampler(self, input_embedding, hist_neighbor_index, max_step):
        """同技能索引历史采样。hist_neighbor_index 取值 [0, max_step]，max_step 指向零向量 padding 行。"""
        B, _, H = input_embedding.shape
        emb = torch.cat(
            [
                input_embedding,
                torch.zeros(
                    B, 1, H, device=input_embedding.device, dtype=input_embedding.dtype
                ),
            ],
            dim=1,
        )
        idx = hist_neighbor_index.reshape(B, max_step * self.hist_neighbor_num)
        return torch.gather(emb, 1, idx.unsqueeze(-1).expand(-1, -1, H)).reshape(
            B, max_step, self.hist_neighbor_num, H
        )

    def forward(
        self,
        user_sequence,
        user_response,
        user_mask,
        skills,
        graph_data,
        hist_neighbor_index,
        return_states=False,
    ):
        max_step = user_sequence.size(1) - 1
        # 节点 id 布局：题目 +num_skills；作答 +num_skills+num_questions
        question_indices = user_sequence[:, :-1] + self.num_skills
        next_question_indices = user_sequence[:, 1:] + self.num_skills
        answer_indices = user_response[:, :-1] + self.num_skills + self.num_questions

        input_questions_embedding = self.feature_embedding(question_indices)
        next_questions_embedding = self.feature_embedding(next_question_indices)
        input_answers_embedding = self.feature_embedding(answer_indices)

        aggregate_embedding, next_aggregate_embedding = self.graph_aggregator(
            question_indices, next_question_indices, graph_data
        )

        feature_trans_embedding = self.feature_layer_act(
            self.feature_layer(aggregate_embedding[0].squeeze(2))
        )
        next_trans_embedding = self.feature_layer_act(
            self.feature_layer(next_aggregate_embedding[0].squeeze(2))
        )

        input_trans_embedding = self.input_trans_layer(
            torch.cat([feature_trans_embedding, input_answers_embedding], dim=-1)
        )
        output_series = self._run_lstm(input_trans_embedding)

        # 历史邻居：hssi/hsei 用同技能索引采样，ssei/dkt 用相似度 top-k
        if self.model_name in ("hssi", "hsei"):
            source = (
                output_series if self.model_name == "hssi" else input_trans_embedding
            )
            hist_neighbors_features = self._hist_neighbor_sampler(
                source, hist_neighbor_index, max_step
            )
        else:
            if self.sim_emb == "skill_emb":
                qe, nqe = (
                    self.feature_embedding(skills[:, :-1]),
                    self.feature_embedding(skills[:, 1:]),
                )
            elif self.sim_emb == "question_emb":
                qe, nqe = input_questions_embedding, next_questions_embedding
            else:
                qe, nqe = feature_trans_embedding, next_trans_embedding
            qa_source = (
                input_trans_embedding if self.model_name == "ssei" else output_series
            )
            hist_neighbors_features = self.history_recap(
                qe, nqe, qa_source, user_mask[:, :-1], hist_neighbor_index
            )

        # 下一题邻居：[下一题特征, 采样的图邻居]
        if self.next_neighbor_num != 0:
            Nn_sampled = self.graph_aggregator.sample_next_neighbors(
                next_aggregate_embedding, self.next_neighbor_num
            )
            Nn = torch.cat([next_trans_embedding.unsqueeze(2), Nn_sampled], dim=2)
        else:
            Nn = next_trans_embedding.unsqueeze(2)

        # 学生状态：[LSTM 输出, 历史邻居]
        if self.hist_neighbor_num != 0:
            Nh = torch.cat([output_series.unsqueeze(2), hist_neighbors_features], dim=2)
        else:
            Nh = output_series.unsqueeze(2)

        logits = self.general_interaction(Nh, Nn, user_mask[:, :-1])

        if return_states:
            return logits, output_series
        return logits
