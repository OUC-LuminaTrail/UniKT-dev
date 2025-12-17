import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, Linear, TransformerConv
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
        super(HistoryRecap, self).__init__()
        self.hist_neighbor_num = hist_neighbor_num
        self.att_bound = att_bound

    def forward(
        self,
        input_q_emb: torch.Tensor,  # [B, S, D]
        next_q_emb: torch.Tensor,  # [B, S, D]
        qa_emb: torch.Tensor,  # [B, S, H]
        user_mask: torch.Tensor,  # [B, S]
    ):
        B, S, _ = input_q_emb.size()
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
    r"""广义交互模块

    参数：
    - student_dim: 学生状态集合的维度
    - knowledge_dim: 知识状态集合的维度
    - attention_dim: 注意力网络中间层维度

    输入：
    - hist_candidates: 学生相关状态集合 [B, S, M+1, student_dim]
    - next_candidates: 知识相关状态集合 [B, S, N+1, knowledge_dim]
    - user_mask: 用户有效位置掩码 [B, S]

    输出：
    - logits: 预测分数 [B, S]
    """

    def __init__(self, student_dim: int, knowledge_dim: int, attention_dim: int = 64):
        super(GeneralInteraction, self).__init__()
        self.student_dim = student_dim
        self.knowledge_dim = knowledge_dim
        self.attention_dim = attention_dim

        # 统一投影维度：取两者中较小的维度
        self.unified_dim = min(student_dim, knowledge_dim)

        # 线性投影层：将两个集合投影到相同维度
        self.student_proj = (
            nn.Linear(student_dim, self.unified_dim)
            if student_dim != self.unified_dim
            else nn.Identity()
        )
        self.knowledge_proj = (
            nn.Linear(knowledge_dim, self.unified_dim)
            if knowledge_dim != self.unified_dim
            else nn.Identity()
        )

        # 初始化投影层权重
        if isinstance(self.student_proj, nn.Linear):
            nn.init.xavier_uniform_(self.student_proj.weight)
        if isinstance(self.knowledge_proj, nn.Linear):
            nn.init.xavier_uniform_(self.knowledge_proj.weight)

        # 加性注意力：输入拼接向量，输出注意力分数
        self.attention_net = nn.Sequential(
            nn.Linear(2 * self.unified_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        # 初始化权重
        nn.init.xavier_uniform_(self.attention_net[0].weight)
        nn.init.xavier_uniform_(self.attention_net[2].weight)

    def forward(self, hist_candidates, next_candidates, user_mask):
        B, S, M_plus_1, H_student = hist_candidates.size()
        N_plus_1 = next_candidates.size(2)
        H_knowledge = next_candidates.size(3)

        # 投影到统一维度
        # hist_candidates: [B, S, M+1, student_dim] -> [B, S, M+1, unified_dim]
        hist_projected = self.student_proj(
            hist_candidates.reshape(-1, H_student)
        ).reshape(B, S, M_plus_1, self.unified_dim)
        # next_candidates: [B, S, N+1, knowledge_dim] -> [B, S, N+1, unified_dim]
        next_projected = self.knowledge_proj(
            next_candidates.reshape(-1, H_knowledge)
        ).reshape(B, S, N_plus_1, self.unified_dim)

        # 计算两两内积得分
        interaction = hist_projected.unsqueeze(3) * next_projected.unsqueeze(2)
        logits_raw = torch.sum(interaction, dim=-1)  # [B, S, M+1, N+1]

        # 扩展维度以便拼接
        hist_expanded = hist_projected.unsqueeze(3).expand(-1, -1, -1, N_plus_1, -1)
        next_expanded = next_projected.unsqueeze(2).expand(-1, -1, M_plus_1, -1, -1)

        # 拼接交互对向量
        interaction_pairs = torch.cat([hist_expanded, next_expanded], dim=-1)

        # 通过注意力网络计算分数
        attention_scores = self.attention_net(
            interaction_pairs.reshape(-1, 2 * self.unified_dim)
        ).reshape(B, S, M_plus_1, N_plus_1)  # [B, S, M+1, N+1]

        # 展平维度进行softmax
        attention_scores_flat = attention_scores.reshape(
            B, S, -1
        )  # [B, S, (M+1)*(N+1)]
        logits_raw_flat = logits_raw.reshape(B, S, -1)  # [B, S, (M+1)*(N+1)]

        # 应用掩码和softmax
        mask_expanded = user_mask.unsqueeze(-1)
        attention_scores_flat = torch.where(
            mask_expanded,
            attention_scores_flat,
            torch.full_like(attention_scores_flat, -1e9),
        )

        attention_weights = F.softmax(
            attention_scores_flat, dim=-1
        )  # [B, S, (M+1)*(N+1)]

        # 加权求和
        logits = torch.sum(logits_raw_flat * attention_weights, dim=-1)  # [B, S]
        logits = logits * user_mask.float()

        return logits


class GNN_QS(nn.Module):
    r"""问题-技能图聚合

    参数：
    - embedding_dim: 节点嵌入维度
    - n_hop: GNN 层数
    - heads: 注意力头数
    - dropout: Dropout 概率

    输入：
    - x: 节点权重
    - edge_index: 边索引

    输出：
    - x: 聚合后的节点表示
    """

    def __init__(self, embedding_dim, n_hop, heads, dropout):
        super(GNN_QS, self).__init__()
        self.n_hop = n_hop
        self.heads = heads
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()

        for _ in range(n_hop):
            conv = HeteroConv(
                {
                    ("question", "has", "skill"): TransformerConv(
                        (embedding_dim, embedding_dim),
                        embedding_dim,
                        aggr="add",
                        heads=heads,
                        concat=False,
                    ),
                    ("skill", "rev_has", "question"): TransformerConv(
                        (embedding_dim, embedding_dim),
                        embedding_dim,
                        aggr="add",
                        heads=heads,
                        concat=False,
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

    def __init__(self, args, data_metadata, **kwargs):
        super(GIKT, self).__init__(**kwargs)
        # 保存参数
        self.args = args
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
            heads=args.heads,
            dropout=self.dropout,
        )

        # 全连接层，将练习嵌入投影到隐藏维度
        self.fc_exercise = Linear(
            self.embedding_dim * 2, self.hidden_dim, weight_initializer="uniform"
        )

        # LSTM层
        self.lstm = nn.LSTM(
            # 将question和answer拼接作为输入
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # 历史回顾模块
        self.history_review = HistoryRecap(
            hist_neighbor_num=args.history_neighbour,
            att_bound=args.att_bound,
        )

        # 广义交互模块
        self.general_interaction = GeneralInteraction(
            student_dim=self.hidden_dim, knowledge_dim=self.embedding_dim
        )

    def forward(
        self,
        user_sequence: torch.Tensor,
        user_response: torch.Tensor,
        user_mask: torch.Tensor,
        graph,
        question_skill_matrix: torch.Tensor,
    ):
        # 批量大小
        B, _ = user_sequence.size()

        # 获取用户答题序列的回复嵌入
        # [B, S, embedding_dim]
        answers_embedding: torch.Tensor = self.answer_embedding(user_response)

        # 全图卷积
        conv = self.conv(
            {
                "question": self.question_embedding.weight,
                "skill": self.skill_embedding.weight,
            },
            graph.edge_index_dict,
        )

        # 图卷积得到的问题嵌入 [num_questions, embedding_dim]
        question_conv: torch.Tensor = conv["question"]
        # 图卷积得到的技能嵌入 [num_skills, embedding_dim]
        skill_conv: torch.Tensor = conv["skill"]

        # 按照用户序列索引获取对应的问题的嵌入表示
        # user_sequence: [B, S], question_conv: [num_questions, embedding_dim]
        # question_embedding_sequence: [B, S, embedding_dim]
        question_embedding_sequence = question_conv[user_sequence]

        # 组合问题和答案嵌入得到练习嵌入
        exercise_emb = torch.cat(
            [question_embedding_sequence, answers_embedding], dim=-1
        )  # [B, S, 2*E]

        # 将练习嵌入投影到隐藏维度
        exercise_emb = F.relu(self.fc_exercise(exercise_emb))  # [B, S, H]
        exercise_emb = self.embedding_dropout(exercise_emb)

        # lstm_output [B, S, H]
        lstm_output, _ = self.lstm(exercise_emb)

        # 获取下一题问题序列，最后一个时间步用零向量占位
        next_user_sequence = torch.zeros_like(user_sequence)  # [B, S]
        if user_sequence.size(1) > 1:
            # 将 user_sequence 向左移动一位，最后一位用0填充
            next_user_sequence[:, :-1] = user_sequence[:, 1:]
            next_user_sequence[:, -1] = 0

        # 获取下一题的问题序列嵌入
        # next_question_embedding: [B, S, embedding_dim]
        next_question_embedding: torch.Tensor = question_conv[next_user_sequence]

        # 获取下一题学生回复，最后一个时间步用0占位
        next_user_response = torch.zeros_like(user_response)  # [B, S]
        if user_response.size(1) > 1:
            next_user_response[:, :-1] = user_response[:, 1:]
            next_user_response[:, -1] = 0

        # 获取下一题的回复嵌入
        # next_answer_embedding: [B, S, embedding_dim]
        next_answer_embedding: torch.Tensor = self.answer_embedding(next_user_response)

        # 组合下一题问题和答案嵌入
        next_exercise_emb = torch.cat(
            [next_question_embedding, next_answer_embedding], dim=-1
        )  # [B, S, 2*E]

        # 将下一题练习嵌入投影到隐藏维度
        next_exercise_emb = F.relu(self.fc_exercise(next_exercise_emb))  # [B, S, H]
        next_exercise_emb = self.embedding_dropout(next_exercise_emb)

        # 历史回顾模块
        # question_embedding_sequence: [B, S, E]
        # next_question_embedding: [B, S, E]
        # lstm_output: [B, S, H]
        history_question_neighbors = self.history_review(
            question_embedding_sequence,
            next_question_embedding,
            exercise_emb,
            user_mask,
        )  # history_question_neighbors: [B, S, M, H]

        # 构造学生相关状态集合：LSTM 输出 + 历史邻居
        # lstm_output: [B, S, H] -> [B, S, 1, H]
        # history_question_neighbors: [B, S, M, H]
        # student_status: [B, S, M+1, H]
        student_status = torch.cat(
            [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
        )  # [B, S, M+1, H]

        # 构建知识相关状态集合：下一题特征 + 相关知识点特征
        # 获取每个问题关联的技能
        # 此处得到的每一个问题关联的0/1向量表示其关联的技能
        # next_user_sequence: [B, S] 每个元素是问题ID
        q_skill_vectors = question_skill_matrix[
            next_user_sequence
        ]  # [B, S, num_skills]

        # 通过对二值向量降序排序，将值为1的技能索引排在前面
        sorted_skill_indices = torch.argsort(
            q_skill_vectors, dim=-1, descending=True
        )  # [B, S, num_skills]

        # 计算每行最大关联技能数
        max_skills_per_question = int(q_skill_vectors.sum(dim=-1).max().item())
        # 计算每个位置实际关联的技能数量（1的个数）
        skill_counts = q_skill_vectors.sum(dim=-1).long()  # [B, S]

        # 选取每个位置前 K 个技能索引（K=max_skills_per_q）
        related_skill_ids = sorted_skill_indices[
            ..., :max_skills_per_question
        ].clone()  # [B, S, K]

        # 不足 K 个技能的位置使用 padding 索引填充
        device = next_user_sequence.device
        pos = torch.arange(max_skills_per_question, device=device).view(
            1, 1, -1
        )  # [1,1,K]
        valid_pos_mask = pos < skill_counts.unsqueeze(-1)  # [B, S, K]

        padding_index = skill_conv.size(0)  # 额外的零向量行索引
        padding_ids = torch.full_like(related_skill_ids, padding_index)
        related_skill_ids = torch.where(
            valid_pos_mask, related_skill_ids, padding_ids
        )  # [B, S, K]

        # 为 padding 索引追加一行零向量
        skill_conv_padded = torch.cat(
            [
                skill_conv,
                torch.zeros(
                    1, self.embedding_dim, device=device, dtype=skill_conv.dtype
                ),
            ],
            dim=0,
        )  # [num_skills+1, embedding_dim]

        # 获取相关知识点嵌入
        # 从 skill_conv_padded 中取出对应的技能嵌入
        # related_skill_ids: [B, S, max_skills_per_question]
        # related_skill_embs: [B, S, max_skills_per_question, embedding_dim]
        related_skill_embs = skill_conv_padded[related_skill_ids]

        # 拼接得到知识相关状态集合
        # next_question_embedding: [B, S, embedding_dim] -> [B, S, 1, embedding_dim]
        # related_skill_embs: [B, S, max_skills_per_question, embedding_dim]
        knowledge_status = torch.cat(
            [next_question_embedding.unsqueeze(2), related_skill_embs],
            dim=2,
        )  # [B, S, max_skills_per_question+1, embedding_dim]

        # 广义交互模块
        logits = self.general_interaction(
            student_status, knowledge_status, user_mask
        )  # [B, S]

        return logits  # [B, S]
