import torch
import torch.nn.functional as F
from torch import nn


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

    def forward(
        self,
        input_q_emb: torch.Tensor,  # [B, S, D]
        next_q_emb: torch.Tensor,  # [B, S, D]
        qa_emb: torch.Tensor,  # [B, S, H]
        user_mask: torch.Tensor,  # [B, S]
        hist_neighbor_index: torch.Tensor | None = None,  # [B, S, M] 可选的备用索引
    ) -> torch.Tensor:
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

        # 当相似度 <= 0 时，使用预计算的备用索引
        if hist_neighbor_index is not None:
            # 使用预计算的 hist_neighbor_index 作为备用
            temp_hist_index = torch.where(
                hist_attention_value > 0,
                temp_hist_index,
                hist_neighbor_index,  # 使用预计算的索引而非 -1
            )
        else:
            # 如果没有提供 hist_neighbor_index，使用 -1 作为无效标记
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
