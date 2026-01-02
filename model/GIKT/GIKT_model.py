import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, Linear, TransformerConv
from torch.nn import functional as F
from utils.core import MODELS
from ..layers import GeneralInteraction, HistoryRecap


class GNN_QS(nn.Module):
    """问题-技能图聚合。

    Args:
        embedding_dim: 节点嵌入维度
        n_hop: GNN 层数
        heads: 注意力头数
        dropout: Dropout 概率

    输入：
        x: 节点权重
        edge_index: 边索引

    Returns:
        x: 聚合后的节点表示
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


@MODELS.register("GIKT")
class GIKT(nn.Module):
    """GIKT主模型。"""

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
        self.general_interaction = GeneralInteraction(hidden_dim=self.hidden_dim)

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
            lstm_output,  # qa_emb
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
                torch.zeros(1, self.hidden_dim, device=device, dtype=skill_conv.dtype),
            ],
            dim=0,
        )  # [num_skills+1, H]

        # 获取相关知识点嵌入
        # 从 skill_conv_padded 中取出对应的技能嵌入
        # related_skill_ids: [B, S, max_skills_per_question]
        # related_skill_embs: [B, S, max_skills_per_question, H]
        related_skill_embs = skill_conv_padded[related_skill_ids]

        # 拼接得到知识相关状态集合
        # next_q_trans: [B, S, H] -> [B, S, 1, H]
        # related_skill_trans: [B, S, max_skills_per_question, H]
        knowledge_status = torch.cat(
            [next_question_embedding.unsqueeze(2), related_skill_embs],
            dim=2,
        )  # [B, S, max_skills_per_question+1, H]

        # 广义交互模块
        logits = self.general_interaction(
            student_status, knowledge_status, user_mask
        )  # [B, S]

        return logits  # [B, S]
