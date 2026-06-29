"""SQGKT: student-question interaction graph-based knowledge tracing.

1. 边权 g_ij（§4.1 式 6）：g_ij = w_c·c_i + w_p·g_ij^p + w_n·g_ij^n
   - c_i：学生整体正确率（学习能力，式 1）
   - g_ij^p：基于 attempt_count 的泊松知识获取因子（式 2–3）
   - g_ij^n：基于 hint_count 的泊松知识获取因子（式 4–5）
   w_c/w_p/w_n 可学习；c/g^p/g^n 三分量在数据端预算，按位与采样学生对齐。
2. 图嵌入（§4.3 式 9–11）：把答过 q_j 的学生按边权 g_ij 加权聚合“进”问题，得到每题表示 q̃_j。
3. 融合（式 16）：q̂_j = w_q1·q̃_j + w_q2·q_j（q_j 来自问题-技能图），w_q1/w_q2 可学习。
4. q̂_t 同时用于 LSTM 输入（式 17）与预测交互项（式 20）。

修复的原作者 sqgkt.py bug：
- LSTMCell 未传上一时刻隐状态
- predict 末端多做一次 sigmoid 与 BCEWithLogitsLoss 冲突
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GIKT.GIKT_model import GIKTGraphAggregator

from ..layers import GeneralInteraction, HistoryRecap


class SQGKT(nn.Module):
    def __init__(self, args, data_metadata, **kwargs):
        super().__init__(**kwargs)
        self.num_skills = data_metadata["num_skills"]
        self.num_questions = data_metadata["num_questions"]
        self.num_users = data_metadata["num_users"]
        self.embedding_dim = args.embedding_dim
        self.hidden_neurons = list(args.hidden_neurons)
        self.hidden_size = self.hidden_neurons[-1]
        self.dropout_prob = args.dropout_probs[0]
        self.model_name = getattr(args, "variant", "hsei")
        self.sim_emb = getattr(args, "sim_emb", "question_emb")
        self.hist_neighbor_num = args.hist_neighbor_num
        self.next_neighbor_num = args.next_neighbor_num
        self.n_hop = args.n_hop

        assert self.hidden_size == self.embedding_dim, (
            f"hidden_neurons[-1]({self.hidden_size}) must equal embedding_dim({self.embedding_dim})"
        )

        self.feature_embedding = nn.Embedding(
            self.num_skills + self.num_questions + 2, self.embedding_dim
        )

        self.graph_aggregator = GIKTGraphAggregator(
            self.embedding_dim,
            args.question_neighbor_num,
            args.skill_neighbor_num,
            self.n_hop,
            args.dropout_probs,
            args.aggregator,
        )

        # 学生-问题图（§4.3 式 9–11）：学生节点嵌入 + 题目自身嵌入（该图专用）+ GCN 变换。
        # 把答过 q_j 的学生按边权 g_ij 加权聚合“进”该题，得到每题表示 q̃_j。
        self.emb_table_student = nn.Embedding(self.num_users, self.embedding_dim)
        self.emb_table_question_sq = nn.Embedding(
            self.num_questions, self.embedding_dim
        )
        self.sq_transform = nn.Linear(self.embedding_dim, self.embedding_dim)
        # g_ij 边权融合（式 6）与 q̂ 双图融合（式 16，w_q1/w_q2 均为自适应可学习参数）。
        self.w_c = nn.Parameter(torch.tensor(0.33))
        self.w_p = nn.Parameter(torch.tensor(0.33))
        self.w_n = nn.Parameter(torch.tensor(0.33))
        self.w_q1 = nn.Parameter(torch.tensor(0.5))
        self.w_q2 = nn.Parameter(torch.tensor(0.5))

        self.feature_layer = nn.Linear(self.embedding_dim, self.hidden_size)
        self.feature_layer_act = nn.ReLU()
        self.input_trans_layer = nn.Linear(
            self.hidden_size + self.embedding_dim, self.hidden_size
        )

        sizes = [self.embedding_dim] + self.hidden_neurons
        self.lstm_layers = nn.ModuleList(
            nn.LSTM(sizes[i], sizes[i + 1], batch_first=True)
            for i in range(len(self.hidden_neurons))
        )

        self.history_recap = HistoryRecap(
            self.hist_neighbor_num, getattr(args, "att_bound", 0.7)
        )
        self.general_interaction = GeneralInteraction(self.hidden_size)

    def _run_lstm(self, x):
        drop_p = self.dropout_prob
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = F.dropout(x, p=drop_p, training=self.training)
        return x

    def _hist_neighbor_sampler(self, input_embedding, hist_neighbor_index, max_step):
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

    def _aggregate_sq(self, questions):
        """学生-问题图 GCN（式 9–11）：把答过每题的学生按边权 g_ij 加权聚合进该题，返回 q̃ [B, S, d]。"""
        students = self.q_neighbors_2[questions]  # [B, S, K]
        stats = self.uq_stat_q[questions]  # [B, S, K, 3]
        g = (
            self.w_c * stats[..., 0]
            + self.w_p * stats[..., 1]
            + self.w_n * stats[..., 2]
        )  # [B, S, K]
        stu_emb = self.emb_table_student(students)  # [B, S, K, d]
        neighbor_msg = (stu_emb * g.unsqueeze(-1)).mean(dim=2)  # [B, S, d]
        self_msg = self.emb_table_question_sq(questions)  # [B, S, d]
        return self.feature_layer_act(self.sq_transform(neighbor_msg + self_msg))

    def forward(
        self,
        user_sequence,
        user_response,
        user_mask,
        user_ids,
        skills,
        graph_data,
        hist_neighbor_index,
    ):
        max_step = user_sequence.size(1) - 1
        question_indices = user_sequence[:, :-1] + self.num_skills
        next_question_indices = user_sequence[:, 1:] + self.num_skills
        answer_indices = user_response[:, :-1] + self.num_skills + self.num_questions

        input_questions_embedding = self.feature_embedding(question_indices)
        next_questions_embedding = self.feature_embedding(next_question_indices)
        input_answers_embedding = self.feature_embedding(answer_indices)

        aggregate_embedding, next_aggregate_embedding = self.graph_aggregator(
            question_indices, next_question_indices, graph_data
        )

        # 学生-问题图聚合 q̃，与问题-技能图 q 线性融合得 q̂（式 16）
        self.q_neighbors_2 = graph_data["q_neighbors_2"]
        self.uq_stat_q = graph_data["uq_stat_q"]
        q_tilde_in = self._aggregate_sq(user_sequence[:, :-1])
        q_tilde_next = self._aggregate_sq(user_sequence[:, 1:])
        qhat_in = self.w_q1 * q_tilde_in + self.w_q2 * aggregate_embedding[0].squeeze(2)
        qhat_next = self.w_q1 * q_tilde_next + self.w_q2 * next_aggregate_embedding[
            0
        ].squeeze(2)

        feature_trans_embedding = self.feature_layer_act(self.feature_layer(qhat_in))
        next_trans_embedding = self.feature_layer_act(self.feature_layer(qhat_next))

        input_trans_embedding = self.input_trans_layer(
            torch.cat([feature_trans_embedding, input_answers_embedding], dim=-1)
        )
        output_series = self._run_lstm(input_trans_embedding)

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

        if self.next_neighbor_num != 0:
            nn_sampled = self.graph_aggregator.sample_next_neighbors(
                next_aggregate_embedding, self.next_neighbor_num
            )
            nn_keys = torch.cat([next_trans_embedding.unsqueeze(2), nn_sampled], dim=2)
        else:
            nn_keys = next_trans_embedding.unsqueeze(2)

        if self.hist_neighbor_num != 0:
            nh_values = torch.cat(
                [output_series.unsqueeze(2), hist_neighbors_features], dim=2
            )
        else:
            nh_values = output_series.unsqueeze(2)

        return self.general_interaction(nh_values, nn_keys, user_mask[:, :-1])
