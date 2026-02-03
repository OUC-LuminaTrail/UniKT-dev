import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from utils.core import register_model


@register_model("SQGKT")
class SQGKT(nn.Module):
    def __init__(self, args, data_metadata, **kwargs):
        super(SQGKT, self).__init__(**kwargs)
        # 保存参数
        self.args = args

        # 元数据
        self.data_metadata = data_metadata
        # 模型参数
        self.dim_emb = args.dim_emb
        self.num_question, self.num_concept = (
            data_metadata["num_questions"],
            data_metadata["num_skills"],
        )
        self.num_user = data_metadata["num_users"]
        self.agg_hops = args.agg_hops
        self.dropout4lstm = args.dropout4lstm
        self.dropout4gnn = args.dropout4gnn
        self.rank_k = args.rank_k

        # 两个图使用独立的问题嵌入
        self.embed_question_qs = nn.Embedding(self.num_question, self.dim_emb)
        self.embed_question_uq = nn.Embedding(self.num_question, self.dim_emb)
        self.embed_concept = nn.Embedding(self.num_concept, self.dim_emb)
        self.embed_user = nn.Embedding(self.num_user, self.dim_emb)
        self.embed_correctness = nn.Embedding(2, self.dim_emb)

        # 问题嵌入的混合权重参数，对应论文公式16
        self.w1_q = nn.Parameter(torch.tensor(0.5))
        self.w2_q = nn.Parameter(torch.tensor(0.5))

        # 三个因子的融合权重，对应论文公式6
        self.w_c = nn.Parameter(torch.tensor(0.33))  # ability factor
        self.w_p = nn.Parameter(torch.tensor(0.33))  # attempt factor
        self.w_n = nn.Parameter(torch.tensor(0.33))  # hint factor

        # LSTM单元前的线性层，对应论文公式17
        self.lstm_pre_fc = nn.Linear(self.dim_emb * 2, self.dim_emb * 2)
        self.lstm1 = nn.LSTMCell(self.dim_emb * 2, self.dim_emb)
        self.lstm2 = nn.LSTMCell(self.dim_emb, self.dim_emb)

        # GCN参数 - 问题-技能图 (w1, b1 in paper Eq 9-11)
        self.mlp4agg_qs = nn.ModuleList(
            Linear(self.dim_emb, self.dim_emb) for _ in range(self.agg_hops)
        )
        self.MLP_AGG_last_qs = Linear(self.dim_emb, self.dim_emb)

        # GCN参数 - 学生-问题图 (w2, b2 in paper Eq 12-14)
        self.mlp4agg_uq = nn.ModuleList(
            Linear(self.dim_emb, self.dim_emb) for _ in range(self.agg_hops)
        )
        self.MLP_AGG_last_uq = Linear(self.dim_emb, self.dim_emb)
        self.dropout_lstm = nn.Dropout(self.dropout4lstm)
        self.dropout_gnn = nn.Dropout(self.dropout4gnn)
        self.MLP_query = Linear(self.dim_emb, self.dim_emb)
        self.MLP_key = Linear(self.dim_emb, self.dim_emb)
        # 公式10中的W
        self.MLP_W = Linear(2 * self.dim_emb, 1)

    def forward(
        self,
        user_seq,
        question_seq,
        correctness_seq,
        mask_seq,
        qs_table,
        q_neighbors_qs,  # q_neighbors_qs[question_id] -> 技能邻居
        c_neighbors_qs,  # c_neighbors_qs[skill_id] -> 问题邻居
        uq_table,
        u_neighbors_uq,  # u_neighbors_uq[user_id] -> 问题邻居
        q_neighbors_uq,  # q_neighbors_uq[question_id] -> 用户邻居
    ):
        dim_emb = self.dim_emb
        if not hasattr(self, "device"):
            self.device = question_seq.device

        batch_size, seq_len = question_seq.shape

        # 问题-技能图：q_neighbors_qs 每个问题有多少技能邻居，c_neighbors_qs 每个技能有多少问题邻居
        qs_q_neighbor_size = q_neighbors_qs.shape[1]  # 技能邻居数
        qs_c_neighbor_size = c_neighbors_qs.shape[1]  # 问题邻居数

        # 用户-问题图：u_neighbors_uq 每个用户有多少问题邻居，q_neighbors_uq 每个问题有多少用户邻居
        uq_u_neighbor_size = u_neighbors_uq.shape[1]  # 问题邻居数
        uq_q_neighbor_size = q_neighbors_uq.shape[1]  # 用户邻居数

        # 初始化LSTM隐藏状态和细胞状态
        h1 = torch.nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(
            self.device
        )
        c1 = torch.nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(
            self.device
        )
        h2 = torch.nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(
            self.device
        )
        c2 = torch.nn.init.xavier_uniform_(torch.zeros(batch_size, dim_emb)).to(
            self.device
        )

        state_history = torch.zeros(batch_size, seq_len, dim_emb).to(self.device)
        y_hat = torch.zeros(batch_size, seq_len).to(self.device)

        for t in range(seq_len - 1):
            user_t = user_seq[:, t]
            question_t = question_seq[:, t]
            response_t = correctness_seq[:, t]
            mask_t = torch.ne(mask_seq[:, t], 0)
            emb_response_t = self.embed_correctness(response_t)

            # ========== 问题-技能图的GNN聚合 ==========
            nodes_qs = [question_t[mask_t]]
            batch_size_qs = len(nodes_qs[0])
            for i in range(self.agg_hops):
                nodes_current = nodes_qs[-1].reshape(-1)
                # i=0: 问题->技能, i=1: 技能->问题, i=2: 问题->技能...
                neighbor_shape = [batch_size_qs] + [
                    (qs_q_neighbor_size if j % 2 == 0 else qs_c_neighbor_size)
                    for j in range(i + 1)
                ]
                if i % 2 == 0:
                    # 当前是问题ID，查找其技能邻居
                    nodes_qs.append(
                        q_neighbors_qs[nodes_current].reshape(neighbor_shape)
                    )
                else:
                    # 当前是技能ID，查找其问题邻居
                    nodes_qs.append(
                        c_neighbors_qs[nodes_current].reshape(neighbor_shape)
                    )

            # 嵌入：偶数层是问题，奇数层是技能
            emb_nodes_qs = []
            for i, nodes in enumerate(nodes_qs):
                if i % 2 == 0:
                    emb_nodes_qs.append(self.embed_question_qs(nodes))
                else:
                    emb_nodes_qs.append(self.embed_concept(nodes))

            emb_question_t_qs = self.aggregate_qs(emb_nodes_qs)
            qs_emb_reconstruct = torch.zeros(batch_size, dim_emb).to(self.device)
            qs_emb_reconstruct[mask_t] = emb_question_t_qs
            qs_emb_reconstruct[~mask_t] = self.embed_question_qs(question_t[~mask_t])

            # ========== 用户-问题图的GNN聚合 ==========
            # 起点是用户ID，交替访问：用户->问题->用户->...
            nodes_uq = [user_t[mask_t]]
            batch_size_uq = len(nodes_uq[0])
            for i in range(self.agg_hops):
                nodes_current = nodes_uq[-1].reshape(-1)
                # i=0: 用户->问题, i=1: 问题->用户, i=2: 用户->问题...
                neighbor_shape = [batch_size_uq] + [
                    (uq_u_neighbor_size if j % 2 == 0 else uq_q_neighbor_size)
                    for j in range(i + 1)
                ]
                if i % 2 == 0:
                    # 当前是用户ID，查找其问题邻居
                    nodes_uq.append(
                        u_neighbors_uq[nodes_current].reshape(neighbor_shape)
                    )
                else:
                    # 当前是问题ID，查找其用户邻居
                    nodes_uq.append(
                        q_neighbors_uq[nodes_current].reshape(neighbor_shape)
                    )

            # 嵌入：偶数层是用户，奇数层是问题
            emb_nodes_uq = []
            for i, nodes in enumerate(nodes_uq):
                if i % 2 == 0:
                    emb_nodes_uq.append(self.embed_user(nodes))
                else:
                    emb_nodes_uq.append(self.embed_question_uq(nodes))

            # 使用带权重聚合
            emb_question_t_uq = self.aggregate_uq(
                emb_nodes_uq, user_t[mask_t], uq_table
            )
            uq_emb_reconstruct = torch.zeros(batch_size, dim_emb).to(self.device)
            uq_emb_reconstruct[mask_t] = emb_question_t_uq
            uq_emb_reconstruct[~mask_t] = self.embed_question_uq(question_t[~mask_t])

            # 融合两个图的问题嵌入，公式16
            emb_hat_q = self.w1_q * qs_emb_reconstruct + self.w2_q * uq_emb_reconstruct

            # LSTM更新知识状态
            lstm1_input = torch.concat((emb_hat_q, emb_response_t), dim=1)
            h1, c1 = self.lstm1(lstm1_input, (h1, c1))
            h1 = self.dropout_lstm(h1)

            h2, c2 = self.lstm2(h1, (h2, c2))
            lstm2_output = self.dropout_lstm(h2)

            # 找t+1时刻习题对应的知识点
            question_next = question_seq[:, t + 1]
            correspond_concepts = qs_table[question_next]
            correspond_concepts_list = []
            max_concept = 1
            for i in range(batch_size):
                concepts_index = torch.nonzero(correspond_concepts[i] == 1).squeeze()
                if len(concepts_index.shape) == 0:
                    correspond_concepts_list.append(
                        torch.unsqueeze(self.embed_concept(concepts_index), dim=0)
                    )
                else:
                    if concepts_index.shape[0] > max_concept:
                        max_concept = concepts_index.shape[0]
                    correspond_concepts_list.append(self.embed_concept(concepts_index))
            # 将习题和对应知识点embedding拼接起来
            emb_question_next = self.embed_question_qs(question_next)
            question_concept = torch.zeros(batch_size, max_concept + 1, dim_emb).to(
                self.device
            )
            for b, emb_concepts in enumerate(correspond_concepts_list):
                num_qc = 1 + emb_concepts.shape[0]
                emb_next = torch.unsqueeze(emb_question_next[b], dim=0)
                question_concept[b, 0:num_qc] = torch.concat(
                    (emb_next, emb_concepts), dim=0
                )
            question_concept = question_concept.to(self.device)
            # recap选取历史状态
            current_state = lstm2_output.unsqueeze(dim=1)
            if t == 0:
                # t=0时，只有当前状态，没有历史状态
                current_history_state = current_state
            elif t <= self.rank_k:
                current_history_state = torch.concat(
                    (current_state, state_history[:, 0:t]), dim=1
                )
            else:
                Q = (
                    self.embed_question_qs(question_next)
                    .clone()
                    .detach()
                    .unsqueeze(dim=-1)
                )
                K = self.embed_question_qs(question_seq[:, 0:t]).clone().detach()
                product_score = torch.bmm(K, Q).squeeze(dim=-1)
                _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                select_history = torch.concat(
                    tuple(
                        state_history[i][indices[i]].unsqueeze(dim=0)
                        for i in range(batch_size)
                    ),
                    dim=0,
                )
                current_history_state = torch.concat(
                    (current_state, select_history), dim=1
                )
            # 标准KT对齐：y_hat[:, t] 存储用t时刻信息预测t+1时刻的结果
            # 这样 y_hat[:, t] 对应标签 correctness_seq[:, t+1]
            y_hat[:, t] = self.predict(question_concept, current_history_state)
            state_history[:, t] = lstm2_output
        return y_hat

    def aggregate_qs(self, emb_list):
        """问题-技能图的聚合 (Eq 9-11)"""
        agg_hops = self.agg_hops
        for i in range(agg_hops):
            for j in range(agg_hops - i):
                emb_list[j] = self.sum_aggregate_qs(emb_list[j], emb_list[j + 1], j)
        return torch.tanh(self.MLP_AGG_last_qs(emb_list[0]))

    def sum_aggregate_qs(self, emb_self, emb_neighbor, hop):
        """问题-技能图的单跳聚合"""
        emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2)
        emb_sum = emb_sum_neighbor + emb_self
        return torch.tanh(self.dropout_gnn(self.mlp4agg_qs[hop](emb_sum)))

    def aggregate_uq(self, emb_list, user_ids=None, uq_table=None):
        """学生-问题图的聚合 (Eq 12-14)，考虑 g_ij 权重"""
        agg_hops = self.agg_hops
        for i in range(agg_hops):
            for j in range(agg_hops - i):
                emb_list[j] = self.sum_aggregate_uq(
                    emb_list[j], emb_list[j + 1], j, user_ids, uq_table
                )
        return torch.tanh(self.MLP_AGG_last_uq(emb_list[0]))

    def sum_aggregate_uq(
        self, emb_self, emb_neighbor, hop, user_ids=None, uq_table=None
    ):
        """学生-问题图的单跳聚合"""
        if hop == 0 and user_ids is not None and uq_table is not None:
            # 只在第一跳（用户->问题）时进行加权
            # emb_self: [num_users_batch, emb_dim]
            # emb_neighbor: [num_users_batch, num_neighbors, emb_dim]
            # uq_table: [num_users, num_questions, 3]

            num_nodes = emb_self.size(0)
            emb_dim = emb_self.size(1)
            num_neighbors = emb_neighbor.size(1)

            # 获取用户的因子 [num_users_batch, num_questions, 3]
            user_factors = uq_table[user_ids]  # [batch, num_questions, 3]

            # 对所有问题取平均得到该用户的平均因子 [num_users_batch, 3]
            avg_factors = user_factors.mean(dim=1)  # [batch, 3]

            # 计算 g_ij = w_c * c_i + w_p * g_p + w_n * g_n
            # avg_factors: [batch, 3] where [:, 0]=c_i, [:, 1]=g_p, [:, 2]=g_n
            g_ij = (
                self.w_c * avg_factors[:, 0]
                + self.w_p * avg_factors[:, 1]
                + self.w_n * avg_factors[:, 2]
            )  # [batch]

            # 扩展权重以匹配邻居维度 [batch, num_neighbors, emb_dim]
            g_ij_expanded = g_ij.view(num_nodes, 1, 1).expand(
                num_nodes, num_neighbors, emb_dim
            )

            # 加权聚合
            weighted_neighbor = emb_neighbor * g_ij_expanded
            emb_sum_neighbor = torch.mean(weighted_neighbor, dim=-2)
            emb_sum = emb_sum_neighbor + emb_self
        else:
            # 其他跳使用标准平均聚合
            emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2)
            emb_sum = emb_sum_neighbor + emb_self

        return torch.tanh(self.dropout_gnn(self.mlp4agg_uq[hop](emb_sum)))

    def predict(self, question_concept, current_history_state):
        # question_concept: (batch_size, num_qc, dim_emb), current_history_state: (batch_size, num_state, dim_emb)
        output_g = torch.bmm(
            question_concept, torch.transpose(current_history_state, 1, 2)
        )

        num_qc, num_state = question_concept.shape[1], current_history_state.shape[1]
        states = torch.unsqueeze(
            current_history_state, dim=1
        )  # [batch_size, 1, num_state, dim_emb]
        states = states.repeat(
            1, num_qc, 1, 1
        )  # [batch_size, num_qc, num_state, dim_emb]
        question_concepts = torch.unsqueeze(
            question_concept, dim=2
        )  # [batch_size, num_qc, 1, dim_emb]
        question_concepts = question_concepts.repeat(
            1, 1, num_state, 1
        )  # [batch_size, num_qc, num_state, dim_emb]

        K = torch.tanh(
            self.MLP_query(states)
        )  # [batch_size, num_qc, num_state, dim_emb]
        Q = torch.tanh(
            self.MLP_key(question_concepts)
        )  # [batch_size, num_qc, num_state, dim_emb]
        tmp = self.MLP_W(
            torch.concat((Q, K), dim=-1)
        )  # [batch_size, num_qc, num_state, 1]
        tmp = torch.squeeze(tmp, dim=-1)  # [batch_size, num_qc, num_state]
        alpha = torch.softmax(tmp, dim=2)  # [batch_size, num_qc, num_state]
        p = torch.sum(torch.sum(alpha * output_g, dim=1), dim=1)  # [batch_size, 1]
        result = torch.squeeze(p, dim=-1)

        return result
