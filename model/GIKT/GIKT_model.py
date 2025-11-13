import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, Linear, GraphConv
from torch.nn import functional as F


class HistoryRecap(nn.Module):
    r"""基于余弦相似度的历史邻居采样模块

    - 对于时间步 t，计算"下一题(t+1)"与"历史所有题目(0..t)"的余弦相似度
    - 使用 top-k 选取最相似的 M 个历史位置
    - 从指定的表示向量（如 LSTM 输出）中 gather 这些位置的特征

    参数:
        hist_neighbor_num (int): 要采样的历史邻居数量 M
        att_bound (float): 相似度阈值，低于此值的设为 0

    输入:
        input_q_emb: [B, S, D] 当前题目的 embedding (用于计算相似度)
        next_q_emb: [B, S, D] 下一题的 embedding (用于计算相似度)
        qa_emb: [B, S, H] 要采样的表示向量
        user_mask: [B, S] 有效位置掩码

    输出:
        hist_neighbors: [B, S, M, H] 采样得到的历史邻居表示
    """

    def __init__(self, hist_neighbor_num: int, att_bound: float = 0.0):
        super().__init__()
        self.hist_neighbor_num = hist_neighbor_num
        self.att_bound = att_bound

    def forward(
        self,
        input_q_emb: torch.Tensor,  # [B, S, D]
        next_q_emb: torch.Tensor,  # [B, S, D]
        qa_emb: torch.Tensor,  # [B, S, H]
        user_mask: torch.Tensor,  # [B, S]
    ):
        B, S, D = input_q_emb.size()
        H = qa_emb.size(-1)
        device = input_q_emb.device

        # 归一化向量
        next_q_norm = F.normalize(next_q_emb, p=2, dim=-1)  # [B, S, D]
        input_q_norm = F.normalize(input_q_emb, p=2, dim=-1)  # [B, S, D]
        # 计算两两余弦相似度：[B, S, D] @ [B, D, S] -> [B, S, S]
        q_similarity = torch.bmm(next_q_norm, input_q_norm.transpose(1, 2))  # [B, S, S]

        # 创建下三角矩阵：tri[i, j] = True if j < i (j 是历史位置)
        tri_mask = torch.tril(
            torch.ones(S, S, device=device, dtype=torch.bool), diagonal=-1
        )  # [S, S]

        # 结合用户有效掩码
        # user_mask: [B, S] -> [B, S, 1] (next位置) 和 [B, 1, S] (input位置)
        valid_next = user_mask.unsqueeze(2)  # [B, S, 1]
        valid_input = user_mask.unsqueeze(1)  # [B, 1, S]

        # 综合掩码: [1, S, S] & [B, S, 1] & [B, 1, S] -> [B, S, S]
        valid_mask = tri_mask.unsqueeze(0) & valid_next & valid_input

        # 将未来位置和无效位置的相似度清零
        q_similarity = q_similarity.masked_fill(~valid_mask, 0.0)

        # 应用相似度阈值
        q_similarity = torch.where(
            q_similarity > self.att_bound, q_similarity, torch.zeros_like(q_similarity)
        )

        # 选择历史邻居位置
        # 注意：对于没有足够历史的位置，会选到相似度=0的位置
        hist_attention_value, temp_hist_index = torch.topk(
            q_similarity,
            k=self.hist_neighbor_num,
            dim=2,  # 在 input 维度（历史维度）上取 top-k
            largest=True,
            sorted=True,
        )  # [B, S, M], [B, S, M]

        # 将相似度为0的位置索引标记为 -1
        temp_hist_index = torch.where(
            hist_attention_value > 0,
            temp_hist_index,
            torch.full_like(temp_hist_index, -1),
        )  # [B, S, M]

        # 在 qa_emb 后添加零向量作为 padding
        zero_padding = torch.zeros(B, 1, H, device=device, dtype=qa_emb.dtype)
        qa_emb_padded = torch.cat([qa_emb, zero_padding], dim=1)  # [B, S+1, H]

        # 将 -1 索引映射到最后一个位置（零向量）
        temp_hist_index = torch.where(
            temp_hist_index >= 0,
            temp_hist_index,
            torch.full_like(temp_hist_index, S),  # S 是 padding 的位置
        )  # [B, S, M]

        # 扩展索引以匹配特征维度
        # temp_hist_index: [B, S, M] -> [B, S, M, H]
        hist_index_expanded = temp_hist_index.unsqueeze(-1).expand(-1, -1, -1, H)

        # 从 qa_emb_padded 中按索引取特征
        hist_neighbors = torch.gather(
            qa_emb_padded.unsqueeze(1).expand(-1, S, -1, -1),  # [B, S, S+1, H]
            2,  # 在第 3 维（历史位置维）上 gather
            hist_index_expanded,
        )  # [B, S, M, H]

        return hist_neighbors


class GeneralInteraction(nn.Module):
    r"""广义交互预测模块

    核心思想：
    - 计算学生相关状态 Nh 与知识相关状态 Nn 的两两内积得分
    - 使用可学习的因子分解注意力：score[i,j] = f1[i] + f2[j]
    - Softmax 归一化后加权求和得到最终预测分数

    参数:
        hidden_dim (int): 特征维度 H

    输入:
        hist_candidates: [B, S, M+1, H] 学生相关状态
        next_candidates: [B, S, N+1, H] 知识相关状态
        user_mask: [B, S] 有效位置掩码

    输出:
        logits: [B, S] 每个时间步的预测分数
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 学生相关状态权重
        self.linear1 = nn.Linear(hidden_dim, 1)
        # 知识相关状态权重
        self.linear2 = nn.Linear(hidden_dim, 1)

        # Xavier 初始化
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(
        self,
        hist_candidates: torch.Tensor,  # [B, S, M+1, H]
        next_candidates: torch.Tensor,  # [B, S, N+1, H]
        user_mask: torch.Tensor,  # [B, S]
    ):
        B, S, M_plus_1, H = hist_candidates.size()
        N_plus_1 = next_candidates.size(2)

        # 计算两两内积得分
        # hist_candidates: [B, S, M+1, H] -> [B, S, M+1, 1, H]
        # next_candidates: [B, S, N+1, H] -> [B, S, 1, N+1, H]
        # interaction: [B, S, M+1, N+1, H]
        interaction = hist_candidates.unsqueeze(3) * next_candidates.unsqueeze(2)
        # 在特征维度上求和得到内积
        logits_raw = torch.sum(interaction, dim=-1)  # [B, S, M+1, N+1]

        # 历史侧得分: [B, S, M+1, H] -> [B*S*(M+1), H] -> [B*S*(M+1), 1]
        f1 = self.linear1(hist_candidates.reshape(-1, H))  # [B*S*(M+1), 1]
        f1 = f1.reshape(B, S, M_plus_1, 1)  # [B, S, M+1, 1]

        # 未来侧得分: [B, S, N+1, H] -> [B*S*(N+1), H] -> [B*S*(N+1), 1]
        f2 = self.linear2(next_candidates.reshape(-1, H))  # [B*S*(N+1), 1]
        f2 = f2.reshape(B, S, 1, N_plus_1)  # [B, S, 1, N+1]

        # 广播相加: [B, S, M+1, 1] + [B, S, 1, N+1] -> [B, S, M+1, N+1]
        attention_scores = torch.tanh(f1 + f2)  # [B, S, M+1, N+1]

        # Step 3: Softmax 归一化
        # 将二维配对展平: [B, S, M+1, N+1] -> [B, S, (M+1)*(N+1)]
        attention_scores_flat = attention_scores.reshape(B, S, -1)  # [B, S, K]
        logits_raw_flat = logits_raw.reshape(B, S, -1)  # [B, S, K]

        # 对无效位置（user_mask=0）的注意力分数设为 -inf
        mask_expanded = user_mask.unsqueeze(-1)  # [B, S, 1]
        attention_scores_flat = torch.where(
            mask_expanded,
            attention_scores_flat,
            torch.full_like(attention_scores_flat, float("-1e9")),
        )
        # Softmax
        attention_weights = F.softmax(attention_scores_flat, dim=-1)  # [B, S, K]

        # 加权求和
        # logits_raw_flat: [B, S, K], attention_weights: [B, S, K]
        logits = torch.sum(logits_raw_flat * attention_weights, dim=-1)  # [B, S]

        # 将无效位置的 logits 清零
        logits = logits * user_mask.float()

        return logits


class GNN_QS(nn.Module):
    r"""问题-技能图聚合

    输入：
    - x: 节点权重
    - edge_index: 边索引
    """

    def __init__(self, embedding_dim, n_hop, dropout):
        super().__init__()
        self.n_hop = n_hop
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        for _ in range(n_hop):
            conv = HeteroConv(
                {
                    ("question", "has_skill", "skill"): GraphConv(
                        (embedding_dim, embedding_dim), embedding_dim, aggr="add"
                    ),
                    ("skill", "rev_has_skill", "question"): GraphConv(
                        (embedding_dim, embedding_dim), embedding_dim, aggr="add"
                    ),
                },
                aggr="sum",
            )
            self.convs.append(conv)
        self.gnn_conv = nn.ModuleList(self.convs)

    def forward(self, x, edge_index):
        for conv in self.gnn_conv:
            x: torch.Tensor = conv(x, edge_index)
            x = {key: x.relu() for key, x in x.items()}
            x = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in x.items()
            }
        return x


class GIKT(nn.Module):
    r"""GIKT主模型"""

    def __init__(self, args, graph, data_metadata, **kwargs):
        super().__init__(**kwargs)
        # 保存参数
        self.args = args
        # 图
        self.graph = graph
        # 元数据
        self.data_metadata = data_metadata

        # 模型参数
        self.embedding_dim = args.embedding_dim  # 嵌入维度
        self.hidden_dim = args.hidden_dim  # 隐藏层维度
        self.lstm_layers = args.lstm_layers  # LSTM层数
        self.dropout = args.dropout  # Dropout概率，所有层共享

        # Embedding
        self.question_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.embedding_dim,
        )
        self.skill_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_skills"],
            embedding_dim=self.embedding_dim,
        )
        self.answer_embedding = torch.nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.embedding_dim,
        )
        self.embedding_dropout = torch.nn.Dropout(p=self.dropout)

        # GNN层
        self.conv = GNN_QS(
            embedding_dim=self.embedding_dim,
            n_hop=args.n_hop,
            dropout=self.dropout,
        )

        # 全连接层，将图卷积后的特征映射到隐藏维度
        self.fc_feature = Linear(
            self.embedding_dim, self.hidden_dim, weight_initializer="uniform"
        )
        self.fc_next_feature = Linear(
            self.embedding_dim, self.hidden_dim, weight_initializer="uniform"
        )

        # LSTM层
        self.lstm = nn.LSTM(
            # 将question和answer拼接作为输入
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # 历史回顾模块
        self.history_recap = HistoryRecap(
            hist_neighbor_num=args.history_neighbour,
            att_bound=args.att_bound,
        )

        # 广义交互模块
        self.general_interaction = GeneralInteraction(hidden_dim=self.hidden_dim)

        # Sigmoid输出层
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        user_sequence: torch.Tensor,
        user_response: torch.Tensor,
        user_mask: torch.Tensor,
    ):
        # 批量大小
        B, _ = user_sequence.size()

        # 获取用户答题序列的回复嵌入
        # [B, S, embedding_dim]
        answers_emb: torch.Tensor = self.answer_embedding(user_response)

        # 图中节点特征初始化
        x = {
            "question": self.question_embedding.weight,
            "skill": self.skill_embedding.weight,
        }

        # 全图卷积
        # question_conv [B, S, embedding_dim]
        question_conv: torch.Tensor = self.conv(x, self.graph.edge_index_dict)[
            "question"
        ]

        # 按照用户序列索引获取对应的问题节点表示
        # 扩展 user_sequence 以匹配 question_conv 的维度
        # user_sequence: [B, S] -> [B, S, embedding_dim]
        user_sequence_expanded = user_sequence.unsqueeze(-1).expand(
            -1, -1, self.embedding_dim
        )
        # 从图卷积后的 question_conv 中按用户序列 ID 取出特征
        # question_emb: [B, S, embedding_dim]
        question_emb = torch.gather(
            question_conv.unsqueeze(0).expand(B, -1, -1),
            1,
            user_sequence_expanded,
        )

        # 变换问题与答案嵌入到隐藏维度后再拼接，作为 LSTM 输入
        q_trans = F.relu(self.fc_feature(question_emb))  # [B, S, H]
        a_trans = F.relu(self.fc_feature(answers_emb))  # [B, S, H]
        exercise_emb = torch.cat([q_trans, a_trans], dim=-1)  # [B, S, 2H]

        # lstm_output [B, S, H]
        lstm_output, _ = self.lstm(exercise_emb)

        # 提取下一题特征，最后一个时间步用零向量占位
        # 构造下一题 ID
        next_user_sequence = torch.zeros_like(user_sequence)
        if user_sequence.size(1) > 1:
            # 将 user_sequence 向左移动一位，最后一位用0填充
            next_user_sequence[:, :-1] = user_sequence[:, 1:]
            next_user_sequence[:, -1] = 0

        next_user_sequence_expanded = next_user_sequence.unsqueeze(-1).expand(
            -1, -1, self.embedding_dim
        )
        # 从图卷积后的 question_conv 中按下一题 ID 取出特征
        # next_q_emb: [B, S, E]
        next_q_emb = torch.gather(
            question_conv.unsqueeze(0).expand(B, -1, -1),
            1,
            next_user_sequence_expanded,
        )
        next_q_trans = F.relu(self.fc_next_feature(next_q_emb))  # [B, S, H]

        # 历史回顾模块：采样历史邻居
        # question_emb: [B, S, E], next_q_emb: [B, S, E], lstm_output: [B, S, H]
        history_question_neighbors = self.history_recap(
            question_emb, next_q_emb, question_emb, user_mask
        )  # history_question_neighbors: [B, S, M, H]

        # 构造学生相关状态集合：当前LSTM输出 + 历史邻居
        # lstm_output: [B, S, H] -> [B, S, 1, H]
        # history_question_neighbors: [B, S, M, H]
        # hist_candidates: [B, S, M+1, H]
        hist_candidates = torch.cat(
            [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
        )  # [B, S, M+1, H]

        # next_q_trans: [B, S, H] -> [B, S, 1, H]
        next_candidates = next_q_trans.unsqueeze(2)  # [B, S, 1, H]

        # 广义交互模块
        logits = self.general_interaction(
            hist_candidates, next_candidates, user_mask
        )  # [B, S]

        return self.sigmoid(logits)  # [B, S]
