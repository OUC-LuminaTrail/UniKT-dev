import torch
from torch.nn import Module, Embedding, Linear, ModuleList, Dropout, LSTMCell
import torch.nn as nn
import torch.nn.functional as F
from utils.core import MODELS


@MODELS.register("SQGKT")
class SQGKT(Module):
    def __init__(
        self,
        args,
        data_metadata,
    ):
        super(SQGKT, self).__init__()
        # 保存参数
        self.args = args
        # 数据元数据
        self.num_user = data_metadata["num_users"]
        self.num_question = data_metadata["num_questions"]
        self.num_skill = data_metadata["num_skills"]

        # 其他参数
        self.agg_hops = args.n_hop  # GNN聚合层数
        self.emb_dim = args.embedding_dim  # 嵌入维度
        self.rank_k = args.rank_k  # 软回顾机制中选择的Top K

        self.question_embedding_1 = Embedding(self.num_question, self.emb_dim)
        self.question_embedding_2 = Embedding(self.num_question, self.emb_dim)
        self.skill_embedding = Embedding(self.num_skill, self.emb_dim)
        self.user_embedding = Embedding(self.num_user, self.emb_dim)
        self.answer_embedding = Embedding(2, self.emb_dim)

        self.w1_q = nn.Parameter(torch.tensor(0.5))
        self.w2_q = nn.Parameter(torch.tensor(0.5))

        self.w_c = nn.Parameter(torch.tensor(0.33))
        self.w_p = nn.Parameter(torch.tensor(0.33))
        self.w_n = nn.Parameter(torch.tensor(0.33))

        self.lstm_linear = Linear(self.emb_dim * 2, self.emb_dim * 2)
        self.lstm_cell = LSTMCell(input_size=self.emb_dim * 2, hidden_size=self.emb_dim)
        self.mlps4agg = ModuleList(
            Linear(self.emb_dim, self.emb_dim) for _ in range(self.agg_hops)
        )
        self.MLP_AGG_last = Linear(self.emb_dim, self.emb_dim)
        self.dropout_lstm = Dropout(args.dropout_lstm)
        self.dropout_gnn = Dropout(args.dropout_gnn)
        self.MLP_query = Linear(self.emb_dim, self.emb_dim)
        self.MLP_key = Linear(self.emb_dim, self.emb_dim)
        self.MLP_W = Linear(2 * self.emb_dim, 1)
        self.attention_weights = torch.nn.Parameter(torch.randn(3))
        self.attention_bias = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        user,
        question,
        response,
        mask,
        uq_matrix,
        qs_matrix,
        qs_q_neighbor_list,
        qs_s_neighbor_list,
        uq_u_neighbor_list,
        uq_q_neighbor_list,
    ):
        DEVICE = user.device

        # 静态数据
        self.uq_matrix = uq_matrix  # 学生-问题矩阵
        self.qs_matrix = qs_matrix  # 问题-技能矩阵
        qs_q_neighbor_list = qs_q_neighbor_list  # 问题技能邻居表
        qs_s_neighbor_list = qs_s_neighbor_list  # 技能问题邻居表
        uq_u_neighbor_list = uq_u_neighbor_list  # 用户问题邻居表
        uq_q_neighbor_list = uq_q_neighbor_list  # 问题用户邻居表

        batch_size, seq_len = question.shape
        q_neighbor_size, s_neighbor_size = (
            qs_q_neighbor_list.shape[1],
            qs_s_neighbor_list.shape[1],
        )
        u_neighbor_size, q_neighbor_size_2 = (
            uq_u_neighbor_list.shape[1],
            uq_q_neighbor_list.shape[1],
        )

        h1_pre = torch.nn.init.xavier_uniform_(
            torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1)
        )
        h2_pre = torch.nn.init.xavier_uniform_(
            torch.zeros(self.emb_dim, device=DEVICE).repeat(batch_size, 1)
        )
        state_history = torch.zeros(batch_size, seq_len, self.emb_dim, device=DEVICE)
        y_hat = torch.zeros(batch_size, seq_len, device=DEVICE)

        for t in range(seq_len - 1):
            user_t = user[:, t]
            question_t = question[:, t]
            response_t = response[:, t]
            mask_t = torch.eq(mask[:, t], torch.tensor(1))
            emb_response_t = self.answer_embedding(response_t)

            node_neighbors = [question_t[mask_t]]
            _batch_size = len(node_neighbors[0])
            for i in range(self.agg_hops):
                nodes_current = node_neighbors[-1].reshape(-1)
                neighbor_shape = [_batch_size] + [
                    (q_neighbor_size if j % 2 == 0 else s_neighbor_size)
                    for j in range(i + 1)
                ]
                if i % 2 == 0:
                    node_neighbors.append(
                        qs_q_neighbor_list[nodes_current].reshape(neighbor_shape)
                    )
                else:
                    node_neighbors.append(
                        qs_s_neighbor_list[nodes_current].reshape(neighbor_shape)
                    )
            emb_node_neighbor = []
            for i, nodes in enumerate(node_neighbors):
                if i % 2 == 0:
                    emb_node_neighbor.append(self.question_embedding_1(nodes))
                else:
                    emb_node_neighbor.append(self.skill_embedding(nodes))
            emb0_question_t = self.aggregate(emb_node_neighbor)
            emb_question_t = torch.zeros(
                batch_size, self.emb_dim, device=DEVICE, dtype=emb0_question_t.dtype
            )
            emb_question_t[mask_t] = emb0_question_t
            emb_question_t[~mask_t] = self.question_embedding_1(question_t[~mask_t]).to(
                emb_question_t.dtype
            )

            node_neighbors_2 = [user_t[mask_t]]
            _batch_size_2 = len(node_neighbors_2[0])
            for i in range(self.agg_hops):
                nodes_current_2 = node_neighbors_2[-1].reshape(-1)
                neighbor_shape_2 = [_batch_size_2] + [
                    (u_neighbor_size if j % 2 == 0 else q_neighbor_size_2)
                    for j in range(i + 1)
                ]
                if i % 2 == 0:
                    node_neighbors_2.append(
                        uq_u_neighbor_list[nodes_current_2].reshape(neighbor_shape_2)
                    )
                else:
                    node_neighbors_2.append(
                        uq_q_neighbor_list[nodes_current_2].reshape(neighbor_shape_2)
                    )
            emb_node_neighbor_2 = []
            for i, nodes in enumerate(node_neighbors_2):
                if i % 2 == 0:
                    emb_node_neighbor_2.append(self.user_embedding(nodes))
                else:
                    emb_node_neighbor_2.append(self.question_embedding_2(nodes))
            emb0_question_t_2 = self.aggregate_uq(emb_node_neighbor_2, node_neighbors_2)
            emb_question_t_2 = torch.zeros(
                batch_size, self.emb_dim, device=DEVICE, dtype=emb0_question_t_2.dtype
            )
            emb_question_t_2[mask_t] = emb0_question_t_2
            emb_question_t_2[~mask_t] = self.question_embedding_2(
                question_t[~mask_t]
            ).to(emb_question_t_2.dtype)

            emb_hat_q = self.w1_q * emb_question_t + self.w2_q * emb_question_t_2

            lstm_input = torch.cat((emb_hat_q, emb_response_t), dim=1)
            lstm_input = self.lstm_linear(lstm_input)
            lstm_input = F.relu(lstm_input)
            h1_pre, h2_pre = self.lstm_cell(lstm_input, (h1_pre, h2_pre))
            lstm_output = self.dropout_lstm(h1_pre)

            q_next = question[:, t + 1]
            skills_related = self.qs_matrix[q_next]
            skills_related_list = []
            max_num_skill = 1
            for i in range(batch_size):
                skills_index = torch.nonzero(skills_related[i]).squeeze()
                if len(skills_index.shape) == 0:
                    skills_related_list.append(
                        torch.unsqueeze(self.skill_embedding(skills_index), dim=0)
                    )
                else:
                    skills_related_list.append(self.skill_embedding(skills_index))
                    if skills_index.shape[0] > max_num_skill:
                        max_num_skill = skills_index.shape[0]

            emb_q_next = self.question_embedding_1(q_next)
            qs_concat = torch.zeros(
                batch_size,
                max_num_skill + 1,
                self.emb_dim,
                device=DEVICE,
                dtype=emb_q_next.dtype,
            )
            for i, emb_skills in enumerate(skills_related_list):
                num_qs = 1 + emb_skills.shape[0]
                emb_next = torch.unsqueeze(emb_q_next[i], dim=0)
                # emb_skills 与 emb_next 可能为半精度；确保与目标一致
                qs_concat[i, 0:num_qs] = torch.cat((emb_next, emb_skills), dim=0).to(
                    qs_concat.dtype
                )

            if t == 0:
                y_hat[:, 0] = torch.tensor(0.5, device=DEVICE, dtype=y_hat.dtype)
                y_hat[:, 1] = self.predict(
                    qs_concat, torch.unsqueeze(lstm_output, dim=1)
                ).to(y_hat.dtype)
                continue

            current_state = lstm_output.unsqueeze(dim=1)
            if t <= self.rank_k:
                current_history_state = torch.cat(
                    (current_state, state_history[:, 0:t]), dim=1
                )
            else:
                Q = self.question_embedding_1(q_next).clone().detach().unsqueeze(dim=-1)
                K = self.question_embedding_1(question[:, 0:t]).clone().detach()
                product_score = torch.bmm(K, Q).squeeze(dim=-1)
                _, indices = torch.topk(product_score, k=self.rank_k, dim=1)
                select_history = torch.cat(
                    tuple(
                        state_history[i][indices[i]].unsqueeze(dim=0)
                        for i in range(batch_size)
                    ),
                    dim=0,
                )
                current_history_state = torch.cat(
                    (current_state, select_history), dim=1
                )

            y_hat[:, t + 1] = self.predict(qs_concat, current_history_state).to(
                y_hat.dtype
            )
            state_history[:, t] = lstm_output.to(state_history.dtype)
        return y_hat

    def aggregate(self, emb_node_neighbor):
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                emb_node_neighbor[j] = self.sum_aggregate(
                    emb_node_neighbor[j], emb_node_neighbor[j + 1], j
                )
        return torch.relu(self.MLP_AGG_last(emb_node_neighbor[0]))

    def sum_aggregate(self, emb_self, emb_neighbor, hop):
        emb_sum_neighbor = torch.mean(emb_neighbor, dim=-2)
        emb_sum = emb_sum_neighbor + emb_self
        return torch.relu(self.dropout_gnn(self.mlps4agg[hop](emb_sum)))

    def aggregate_uq(self, emb_node_neighbor, id_node_neighbor):
        for i in range(self.agg_hops):
            for j in range(self.agg_hops - i):
                if j % 2 == 0:
                    emb_node_neighbor[j] = self.sum_aggregate_uq(
                        emb_node_neighbor[j],
                        emb_node_neighbor[j + 1],
                        j,
                        id_node_neighbor[j],
                        id_node_neighbor[j + 1],
                    )
                else:
                    emb_node_neighbor[j] = self.sum_aggregate(
                        emb_node_neighbor[j], emb_node_neighbor[j + 1], j
                    )
        return torch.relu(self.MLP_AGG_last(emb_node_neighbor[0]))

    def sum_aggregate_uq(self, emb_self, emb_neighbor, hop, self_ids, neighbor_ids):
        """聚合用户-问题图的邻居节点特征。

        Args:
            emb_self: 中心节点的嵌入，shape: [num_center_nodes, emb_dim]
            emb_neighbor: 邻居节点的嵌入，shape: [num_center_nodes, neighbor_size, emb_dim]
            hop: 当前的聚合层数
            self_ids: 中心节点的全局ID，shape: [num_center_nodes]
            neighbor_ids: 邻居节点的全局ID，shape: [num_center_nodes, neighbor_size]

        Returns:
            加权聚合后的嵌入
        """
        # 在 User-Question Graph 中，中心节点是 User，邻居是 Question
        user_ids = self_ids
        question_ids = neighbor_ids

        # 使用PyTorch高级索引，一次性、高效地获取所有需要的特征
        # uq_table[user_ids, :, :] -> [num_center_nodes, num_questions, 3]
        # .gather(...) -> 根据 question_ids 在第二维度上选取正确的特征
        # user_ids.unsqueeze(-1).expand(-1, question_ids.shape[1]) -> 构造用于gather的索引

        # 扩展 user_ids 以匹配 question_ids 的形状，用于索引
        expanded_user_ids = user_ids.unsqueeze(1).expand_as(question_ids)

        # 直接使用高级索引从 self.uq_table 中批量获取权重
        # self.uq_table 的形状: [全局用户数, 全局问题数, 3]
        # expanded_user_ids 的形状: [批次大小, 邻居数]
        # question_ids 的形状: [批次大小, 邻居数]
        # node_weights 的形状将是: [批次大小, 邻居数, 3]
        node_weights = self.uq_matrix[expanded_user_ids, question_ids, :]

        # 分别提取三个因子
        c_i = node_weights[..., 0].unsqueeze(-1)  # Shape: [批次大小, 邻居数, 1]
        g_p = node_weights[..., 1].unsqueeze(-1)  # Shape: [批次大小, 邻居数, 1]
        g_n = node_weights[..., 2].unsqueeze(-1)  # Shape: [批次大小, 邻居数, 1]

        # 计算融合权重 g_ij
        # self.w_c, self.w_p, self.w_n 是标量
        # fusion_weights 的形状: [批次大小, 邻居数, 1]
        fusion_weights = self.w_c * c_i + self.w_p * g_p + self.w_n * g_n

        # 将权重应用到邻居嵌入上
        # emb_neighbor 的形状: [批次大小, 邻居数, emb_dim]
        # fusion_weights 广播后与 emb_neighbor 相乘
        weighted_neighbor_embs = emb_neighbor * fusion_weights

        # 对加权后的邻居嵌入求平均
        weighted_emb_neighbor_sum = torch.mean(
            weighted_neighbor_embs, dim=1
        )  # Shape: [批次大小, emb_dim]

        # 与自身嵌入相加
        emb_sum = emb_self + weighted_emb_neighbor_sum

        # 应用MLP和激活函数
        return torch.relu(self.dropout_gnn(self.mlps4agg[hop](emb_sum)))

    def predict(self, qs_concat, current_history_state):
        output_g = torch.bmm(qs_concat, torch.transpose(current_history_state, 1, 2))
        num_qs, num_state = qs_concat.shape[1], current_history_state.shape[1]
        states = torch.unsqueeze(current_history_state, dim=1).repeat(1, num_qs, 1, 1)
        qs_concat2 = torch.unsqueeze(qs_concat, dim=2).repeat(1, 1, num_state, 1)
        K = torch.tanh(self.MLP_query(states))
        Q = torch.tanh(self.MLP_key(qs_concat2))
        tmp = self.MLP_W(torch.cat((Q, K), dim=-1))
        tmp = torch.squeeze(tmp, dim=-1)
        alpha = torch.softmax(tmp, dim=2)
        p = torch.sum(torch.sum(alpha * output_g, dim=1), dim=1)
        result = torch.squeeze(p, dim=-1)
        return result
