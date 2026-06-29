import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, TransformerConv


class HistoryRecap(nn.Module):
    r"""基于余弦相似度的历史邻居采样模块

    - 对于时间步 t，计算"下一题(t+1)"与"历史所有题目(0..t)"的余弦相似度
    - 使用 top-k 选取最相似的 M 个历史位置
    - 从指定的表示向量（如 LSTM 输出）中 gather 这些位置的特征
    - 当相似度低于阈值时，使用预计算的 hist_neighbor_index 作为备用索引

    参数:
        hist_neighbor_num (int): 要采样的历史邻居数量 M
        att_bound (float): 相似度阈值，低于此值的设为 0

    输入:
        input_q_emb: [B, S, D] 当前题目的 embedding (用于计算相似度)
        next_q_emb: [B, S, D] 下一题的 embedding (用于计算相似度)
        qa_emb: [B, S, H] 要采样的表示向量
        user_mask: [B, S] 有效位置掩码
        hist_neighbor_index: [B, S, M] 预计算的备用索引（可选）

    输出:
        hist_neighbors: [B, S, M, H] 采样得到的历史邻居表示
    """

    def __init__(self, hist_neighbor_num: int, att_bound: float = 0.0):
        super().__init__()
        self.hist_neighbor_num = hist_neighbor_num
        self.att_bound = att_bound
        self.register_buffer("tri_mask", None, persistent=False)

    def forward(
        self,
        input_q_emb: torch.Tensor,  # [B, S, D]
        next_q_emb: torch.Tensor,  # [B, S, D]
        qa_emb: torch.Tensor,  # [B, S, H]
        user_mask: torch.Tensor,  # [B, S]
        hist_neighbor_index: torch.Tensor | None = None,  # [B, S, M] 可选的备用索引
    ) -> torch.Tensor:
        B, S, _ = input_q_emb.size()
        device = input_q_emb.device

        # 归一化向量
        next_q_norm = F.normalize(next_q_emb, p=2, dim=-1)  # [B, S, D]
        input_q_norm = F.normalize(input_q_emb, p=2, dim=-1)  # [B, S, D]
        # 计算两两余弦相似度：[B, S, D] @ [B, D, S] -> [B, S, S]
        q_similarity = torch.bmm(next_q_norm, input_q_norm.transpose(1, 2))  # [B, S, S]

        # 创建下三角矩阵：tri[i, j] = True if j < i (j 是历史位置)
        if (
            self.tri_mask is None
            or self.tri_mask.size(-1) != S
            or self.tri_mask.device != device
        ):
            self.tri_mask = torch.tril(
                torch.ones(S, S, device=device, dtype=torch.bool), diagonal=-1
            )
        tri_mask = self.tri_mask  # [S, S]

        # 结合用户有效掩码
        # user_mask: [B, S] -> [B, S, 1] (next位置) 和 [B, 1, S] (input位置)
        valid_next = user_mask.unsqueeze(2)  # [B, S, 1]
        valid_input = user_mask.unsqueeze(1)  # [B, 1, S]

        # 综合掩码: [1, S, S] & [B, S, 1] & [B, 1, S] -> [B, S, S]
        valid_mask = tri_mask.unsqueeze(0) & valid_next & valid_input

        # 将未来/无效位置及低于阈值的相似度一并清零
        keep = valid_mask & (q_similarity > self.att_bound)
        q_similarity = q_similarity.masked_fill(~keep, 0.0)

        hist_attention_value, temp_hist_index = torch.topk(
            q_similarity,
            k=self.hist_neighbor_num,
            dim=2,
            largest=True,
            sorted=True,
        )

        fallback = (
            hist_neighbor_index
            if hist_neighbor_index is not None
            else torch.full_like(temp_hist_index, -1)
        )
        temp_hist_index = torch.where(
            hist_attention_value > 0, temp_hist_index, fallback
        )

        qa_emb_padded = F.pad(qa_emb, (0, 0, 0, 1))
        temp_hist_index = temp_hist_index.masked_fill(temp_hist_index < 0, S)

        batch_idx = torch.arange(B, device=device).view(B, 1, 1)
        hist_neighbors = qa_emb_padded[batch_idx, temp_hist_index]

        return hist_neighbors


class GeneralInteraction(nn.Module):
    r"""广义交互模块

    参数：
    - hidden_dim: 隐藏层维度

    输入：
    - hist_candidates: 学生相关状态集合 [B, S, M+1, H]
    - next_candidates: 知识相关状态集合 [B, S, N+1, H]
    - user_mask: 用户有效位置掩码 [B, S]

    输出：
    - logits: 预测分数 [B, S]
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 加性注意力的两个权重向量和偏置
        self.w1 = nn.Parameter(torch.empty(hidden_dim, 1))
        self.w2 = nn.Parameter(torch.empty(hidden_dim, 1))
        self.b1 = nn.Parameter(torch.empty(1))
        self.b2 = nn.Parameter(torch.empty(1))

        # Xavier初始化
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def forward(
        self,
        hist_candidates: torch.Tensor,  # [B, S, M+1, H]
        next_candidates: torch.Tensor,  # [B, S, N+1, H]
        user_mask: torch.Tensor,  # [B, S]
    ) -> torch.Tensor:
        B, S, M_plus_1, H = hist_candidates.size()
        N_plus_1 = next_candidates.size(2)

        # 1. 计算两两内积得分
        logits_raw = torch.einsum("bsmh,bsnh->bsmn", hist_candidates, next_candidates)
        # [B, S, M+1, N+1]

        # 2. 计算加性注意力分数
        # 分别对hist和next应用线性变换
        # hist_candidates: [B, S, M+1, H] -> [B, S, M+1, 1]
        f1 = (
            torch.matmul(
                hist_candidates.reshape(-1, H),  # [B*S*(M+1), H]
                self.w1,  # [H, 1]
            ).reshape(B, S, M_plus_1, 1)
            + self.b1
        )  # [B, S, M+1, 1]

        # next_candidates: [B, S, N+1, H] -> [B, S, 1, N+1]
        f2 = (
            torch.matmul(
                next_candidates.reshape(-1, H),  # [B*S*(N+1), H]
                self.w2,  # [H, 1]
            ).reshape(B, S, 1, N_plus_1)
            + self.b2
        )  # [B, S, 1, N+1]

        # 3. 广播相加并应用tanh激活
        # f1: [B, S, M+1, 1]
        # f2: [B, S, 1, N+1]
        # f: [B, S, M+1, N+1]
        attention_scores = torch.tanh(f1 + f2)  # [B, S, M+1, N+1]

        # 4. 展平并进行softmax
        attention_scores_flat = attention_scores.reshape(
            B, S, -1
        )  # [B, S, (M+1)*(N+1)]
        logits_raw_flat = logits_raw.reshape(B, S, -1)  # [B, S, (M+1)*(N+1)]

        # 5. 应用用户掩码
        mask_expanded = user_mask.unsqueeze(-1)  # [B, S, 1]
        attention_scores_flat = torch.where(
            mask_expanded,
            attention_scores_flat,
            torch.full_like(attention_scores_flat, -1e9),
        )

        # 6. Softmax归一化
        attention_weights = F.softmax(
            attention_scores_flat,
            dim=-1,
        )  # [B, S, (M+1)*(N+1)]

        # 7. 加权求和
        logits = torch.sum(logits_raw_flat * attention_weights, dim=-1)  # [B, S]

        # 8. 应用掩码（无效位置置零）
        logits = logits * user_mask.float()

        return logits


class GNN_QS(nn.Module):
    """问题-技能图聚合模块。

    使用 TransformerConv 进行多层异构图神经网络聚合。

    Args:
        embedding_dim: 节点嵌入维度
        n_hop: GNN 层数
        heads: 注意力头数
        dropout: Dropout 概率

    Example:
        >>> gnn = GNN_QS(embedding_dim=128, n_hop=2, heads=4, dropout=0.2)
        >>> x = {"question": q_emb, "skill": s_emb}
        >>> edge_index = {("question", "has", "skill"): edge1, ("skill", "rev_has", "question"): edge2}
        >>> output = gnn(x, edge_index)
    """

    def __init__(
        self,
        embedding_dim: int,
        n_hop: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
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

    def forward(
        self,
        x: dict[str, torch.Tensor],
        edge_index: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x: 节点特征字典
            edge_index: 边索引字典

        Returns:
            聚合后的节点表示字典
        """
        for conv in self.gnn_conv:
            x: torch.Tensor = conv(x, edge_index)
            x = {key: x.relu() for key, x in x.items()}
            x = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in x.items()
            }
        return x
