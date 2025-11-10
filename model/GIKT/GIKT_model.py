import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, Linear, GATv2Conv
from torch.nn import functional as F


class History_Recap(nn.Module):
    r"""历史回顾模块
        按时间步进行历史回顾：对每个时间步 t 使用下一题表征 next_q_dmb[:, t]
        作为查询，对历史 LSTM 输出 lstm_output[:, :t] 做加权求和。

        输入:
        - lstm_output: [B, S, H]
        - next_q_dmb: [B, S, H]
        - user_mask:   [B, S]  (0/1)

        输出:
        - recap: [B, S, H]
    """

    def __init__(self, hidden_dim, topk: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topk = topk if (topk is not None and topk > 0) else None
        self.atn_w1 = Linear(hidden_dim, 1, weight_initializer="uniform")
        self.atn_w2 = Linear(hidden_dim, 1, weight_initializer="uniform")

    def forward(
        self,
        lstm_output: torch.Tensor,
        next_q_dmb: torch.Tensor,
        user_mask: torch.Tensor,
    ):
        B, S, H = lstm_output.size()

        # 从隐藏维度映射到一维
        score1 = self.atn_w1(lstm_output).squeeze(-1)  # [B, S]
        score2 = self.atn_w2(next_q_dmb).squeeze(-1)  # [B, S]
        # 拼接成 [B, S(history), S(current)] 的两两配对打分矩阵
        scores: torch.Tensor = score1.unsqueeze(1) + score2.unsqueeze(2)  # [B, S, S]
        scores = torch.tanh(scores)

        # 将分数矩阵转为下三角矩阵以掩盖未来的信息
        tri = torch.tril(
            torch.ones(S, S, device=lstm_output.device, dtype=torch.bool), diagonal=-1
        )  # [S, S]
        mask_hist = tri.unsqueeze(0)  # [1, S, S]
        user_mask = user_mask.to(torch.bool)
        # 使用掩码保留有效的位置
        mask_j = user_mask.unsqueeze(-1)  # [B, S, 1]
        mask_t = user_mask.unsqueeze(1)  # [B, 1, S]
        valid = mask_hist & mask_j & mask_t  # [B, S, S]
        # 将无效的位置填为无穷小
        scores = scores.masked_fill(~valid, float(-1e9))

        # top-k 选择
        if self.topk is not None:
            if self.topk < S:  # 仅当 k 小于历史长度时才有裁剪意义
                # 选取每个当前时间步 t 的 top-k 历史位置
                _, topk_idx = torch.topk(scores, k=self.topk, dim=1)
                # 生成掩码，仅保留 top-k 位置
                keep = torch.zeros_like(scores).scatter(1, topk_idx, 1.0)
                # 将非 top-k 位置的分数置为 -inf
                scores = scores.masked_fill(keep < 0.5, float(-1e9))

        attn = F.softmax(scores, dim=2)  # [B, S, S]
        # 处理无历史的时间步
        has_hist = valid.any(dim=1, keepdim=True)  # [B, 1, S]
        attn = attn * has_hist  # 广播到 [B, S, S]

        recap = torch.matmul(attn, lstm_output)  # [B, S, H]
        return recap


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
                    ("question", "has_skill", "skill"): GATv2Conv(
                        (embedding_dim, embedding_dim),
                        embedding_dim,
                        add_self_loops=False,
                    ),
                    ("skill", "rev_has_skill", "question"): GATv2Conv(
                        (embedding_dim, embedding_dim),
                        embedding_dim,
                        add_self_loops=False,
                    ),
                },
                aggr="mean",
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
        history_topk = args.top_k
        self.history_recap = History_Recap(hidden_dim=self.hidden_dim, topk=history_topk)

        # Sigmoid输出层
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        user_sequence: torch.Tensor,
        user_response: torch.Tensor,
        user_mask: torch.Tensor,
    ):
        # 批量大小
        batch_size = user_sequence.size(0)

        # 获取用户答题序列的回复嵌入
        # 在时刻t预测的是基于t-1时刻的回复，而不是当前t时刻的回复
        # 将response向右移动一位，第一位用0填充
        user_response_shifted = torch.zeros_like(user_response)
        if user_response.size(1) > 1:
            user_response_shifted[:, 1:] = user_response[:, :-1]
        # [B, S, embedding_dim]
        answers_emb: torch.Tensor = self.answer_embedding(user_response_shifted)

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
            question_conv.unsqueeze(0).expand(batch_size, -1, -1),
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
            question_conv.unsqueeze(0).expand(batch_size, -1, -1),
            1,
            next_user_sequence_expanded,
        )
        next_q_trans = F.relu(self.fc_next_feature(next_q_emb))  # [B, S, H]

        # 历史回顾模块
        recap_representation = self.history_recap(
            lstm_output, next_q_trans, user_mask
        )  # [B, S, H]

        # 预测输出
        logits = torch.sum(recap_representation * next_q_trans, dim=-1)  # [B, S]

        return self.sigmoid(logits) # [B, S]
