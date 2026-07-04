import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter


def _cone_map(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """锥几何的对数映射 ξ(·)。

    将向量的最后一维作为"半径"分量，其余维度作为"方向"分量。

    Args:
        x: 输入张量 [..., d_k]

    Returns:
        (direction_scaled, radius): direction_scaled 形状 [..., d_k-1]，
        radius 形状 [...]（= exp(x_last / d_k)）。
    """
    dim = x.shape[-1]
    direction = x[..., :-1]
    radius = torch.exp(x[..., -1] / dim)
    return direction * radius.unsqueeze(-1), radius


def cone_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    sinh_r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """锥注意力打分（负距离）。

    打分 = -gamma * max(max(q_r, k_r), (cdist(q_dir, k_dir) / sinh(r) + q_r + k_r) / 2)

    Args:
        query: [bs, heads, seq_len, d_k]
        key:   [bs, heads, seq_len, d_k]
        sinh_r: 预计算的 sinh(r) 标量张量
        gamma: 温度缩放

    Returns:
        scores: [bs, heads, seq_len, seq_len]
    """
    query_dir, query_radius = _cone_map(query)
    key_dir, key_radius = _cone_map(key)

    # [bs, heads, seq_len, 1] 与 [bs, heads, 1, seq_len] 广播取最大
    query_radius_pair = query_radius.unsqueeze(-1)  # 行方向（query 位置）
    key_radius_pair = key_radius.unsqueeze(-2)  # 列方向（key 位置）
    radius_max = torch.maximum(query_radius_pair, key_radius_pair)

    # 方向分量间的欧式距离
    direction_dist = torch.cdist(query_dir, key_dir)
    mixed = (direction_dist / sinh_r + query_radius_pair + key_radius_pair) / 2.0

    scores = torch.maximum(radius_max, mixed)
    return -gamma * scores


class KerpleLogBias(nn.Module):
    """KerpleLog 相对位置核偏置。

    bias[h, i, j] = -p_h * log(1 + a_h * max(i - j, 0))
    其中 p_h、a_h 为每个注意力头可学习的非负参数。
    """

    def __init__(self, num_heads: int, eps: float = 1e-2):
        super().__init__()
        self.num_heads = num_heads
        self.eps = eps

        scale = 2.0
        self.bias_p = Parameter(
            torch.rand(num_heads, dtype=torch.float32)[:, None, None] * scale
        )
        self.bias_a = Parameter(
            torch.rand(num_heads, dtype=torch.float32)[:, None, None] * 1.0
        )

        # 相对位置差矩阵缓存
        self.cached_diff: torch.Tensor | None = None
        self.cached_seq_len: int | None = None

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        seq_len_k = scores.shape[-1]

        if self.cached_seq_len != seq_len_k:
            row_idx = torch.arange(seq_len_k, device=scores.device).view(seq_len_k, 1)
            col_idx = torch.arange(0, -seq_len_k, -1, device=scores.device)
            diff = torch.tril(row_idx + col_idx).to(scores.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_diff = diff
        else:
            diff = self.cached_diff

        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)

        bias = -self.bias_p * torch.log(1.0 + self.bias_a * diff)  # [heads, seq, seq]
        return scores + bias


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_dim: int,
    mask_out: torch.Tensor,
    zero_row_mask: torch.Tensor | None,
    dropout: nn.Dropout,
    sinh_r: torch.Tensor,
    gamma: float,
    kernel_bias: KerpleLogBias,
) -> torch.Tensor:
    """锥注意力 + KerpleLog 偏置 + 因果掩码。

    Args:
        query/key/value: [bs, heads, seq_len, head_dim]
        head_dim: 每个头的维度 d_k
        mask_out: bool 掩码 [1, 1, seq_len, seq_len]，True 表示需要屏蔽
        zero_row_mask: 行掩码 [1, 1, seq_len, 1]，第 0 行为 0 其余为 1；
            非 None 时将第 0 行注意力权重置零（首题无历史交互）。None 表示不置零。
        dropout: 注意力权重 dropout
        sinh_r: 预计算的 sinh(r) 标量张量
        gamma: cone attention 温度参数
        kernel_bias: KerpleLog 相对位置偏置模块

    Returns:
        output: [bs, heads, seq_len, head_dim]
    """
    # 锥注意力打分并叠加相对位置核偏置
    scores = cone_attention(query, key, sinh_r, gamma)
    scores = kernel_bias(scores)

    # 因果掩码：屏蔽未来位置
    scores = scores.masked_fill(mask_out, -1e32)
    scores = F.softmax(scores, dim=-1)

    # 第 0 行置零：首位置无历史，输出为 0 向量。
    if zero_row_mask is not None:
        scores = scores * zero_row_mask

    weights = dropout(scores)
    output = torch.matmul(weights, value)
    return output


class MultiHeadAttention(nn.Module):
    """多头锥注意力。"""

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        num_heads: int,
        dropout: float,
        kq_same: bool,
        r: float,
        gamma: float,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.kq_same = kq_same

        self.value_linear = nn.Linear(d_model, d_model, bias=bias)
        self.key_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.query_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.r = r
        self.gamma = gamma
        self.sinh_r = torch.sinh(torch.tensor(r))
        self.kernel_bias = KerpleLogBias(num_heads)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.key_linear.weight)
        xavier_uniform_(self.value_linear.weight)
        if not self.kq_same:
            xavier_uniform_(self.query_linear.weight)
        constant_(self.key_linear.bias, 0.0)
        constant_(self.value_linear.bias, 0.0)
        if not self.kq_same:
            constant_(self.query_linear.bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_out: torch.Tensor,
        zero_row_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bs = query.size(0)

        # 线性投影并按头拆分：[bs, seq, heads, head_dim] -> [bs, heads, seq, head_dim]
        key = self.key_linear(key).view(bs, -1, self.num_heads, self.head_dim)
        if self.kq_same:
            query = self.key_linear(query).view(bs, -1, self.num_heads, self.head_dim)
        else:
            query = self.query_linear(query).view(bs, -1, self.num_heads, self.head_dim)
        value = self.value_linear(value).view(bs, -1, self.num_heads, self.head_dim)

        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # 锥注意力计算
        scores = attention(
            query,
            key,
            value,
            self.head_dim,
            mask_out,
            zero_row_mask,
            self.dropout,
            self.sinh_r,
            self.gamma,
            self.kernel_bias,
        )

        # 合并多头
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output


class TransformerLayer(nn.Module):
    """单层 Transformer 块：锥多头注意力 + 前馈网络。"""

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
        kq_same: bool,
        r: float,
        gamma: float,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model=d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            dropout=dropout,
            kq_same=kq_same,
            r=r,
            gamma=gamma,
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        values: torch.Tensor,
        mask_out: torch.Tensor,
        zero_row_mask: torch.Tensor | None,
        apply_pos: bool = True,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            query/key/values: [bs, seq, d_model]
            mask_out: 因果掩码 [1, 1, seq, seq]，True 表示需要屏蔽
            zero_row_mask: 行掩码 [1, 1, seq, 1]，第 0 行为 0（置零首个位置）
            apply_pos: 是否应用前馈子层

        Returns:
            query: [bs, seq, d_model]
        """
        # 注意力子层
        attn_out = self.attention(query, key, values, mask_out, zero_row_mask)

        query = query + self.dropout1(attn_out)
        query = self.layer_norm1(query)

        if apply_pos:
            ffn_out = self.ffn_linear2(
                self.ffn_dropout(self.activation(self.ffn_linear1(query)))
            )
            query = query + self.dropout2(ffn_out)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """CSKT 单流架构。

    维护 Transformer 流：query=key=技能嵌入 x，value=交互嵌入 y（恒定）。
    每层都使用因果注意力（mask=0）+ 前馈子层。
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
        kq_same: bool,
        r: float,
        gamma: float,
    ):
        super().__init__()
        head_dim = d_model // num_heads
        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    head_dim=head_dim,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    dropout=dropout,
                    kq_same=kq_same,
                    r=r,
                    gamma=gamma,
                )
                for _ in range(num_blocks)
            ]
        )
        # 因果掩码与首行置零掩码缓存
        self._mask_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_masks(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构建因果掩码与首行置零掩码。

        - mask_out: bool [1, 1, seq, seq]，True 表示屏蔽（上三角含对角，即 j >= i）
        - zero_row_mask: dtype [1, 1, seq, 1]，第 0 行为 0、其余为 1
        """
        key = (seq_len, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        # 上三角含对角线
        mask_out = (
            torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=0)
            .bool()
            .view(1, 1, seq_len, seq_len)
        )

        zero_row_mask = torch.ones(1, 1, seq_len, 1, device=device, dtype=dtype)
        zero_row_mask[0, 0, 0, 0] = 0.0

        self._mask_cache[key] = (mask_out, zero_row_mask)
        return mask_out, zero_row_mask

    def forward(
        self,
        skill_embed: torch.Tensor,
        interaction_embed: torch.Tensor,
    ) -> torch.Tensor:
        """单流前向。

        Args:
            skill_embed: 技能嵌入 [bs, seq, d_model]，作为 query/key
            interaction_embed: 交互嵌入 [bs, seq, d_model]，作为 value（全程不变）

        Returns:
            知识状态表示 [bs, seq, d_model]
        """
        seq_len = skill_embed.size(1)
        mask_out, zero_row_mask = self._get_masks(
            seq_len, skill_embed.device, skill_embed.dtype
        )

        query = skill_embed
        values = interaction_embed
        for block in self.blocks:
            query = block(
                query=query,
                key=query,
                values=values,
                mask_out=mask_out,
                zero_row_mask=zero_row_mask,
                apply_pos=True,
            )
        return query


class CSKT(nn.Module):
    """CSKT模型。

    Args:
        num_c: 技能（概念）数量
        n_pid: 题目数量，0 表示不使用 Rasch 题目难度
        d_model: 模型隐藏维度
        num_blocks: Transformer 块数量
        dropout: Dropout 概率
        d_ff: 前馈网络维度
        num_attn_heads: 注意力头数量
        r: cone attention 半径参数
        gamma: cone attention 温度参数
        kq_same: Key 和 Query 是否共享线性变换（1=是）
        final_fc_dim: 输出层第一层全连接维度
        final_fc_dim2: 输出层第二层全连接维度
        separate_qa: 是否使用独立的交互嵌入
        emb_type: 嵌入类型（"qid" 默认；含 "scalar" 则题目难度为标量）
    """

    def __init__(
        self,
        num_c: int,
        n_pid: int = 0,
        d_model: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
        d_ff: int = 256,
        num_attn_heads: int = 4,
        r: float = 0.6,
        gamma: float = 1.0,
        kq_same: int = 1,
        final_fc_dim: int = 512,
        final_fc_dim2: int = 256,
        separate_qa: bool = False,
        emb_type: str = "qid",
    ):
        super().__init__()
        self.num_c = num_c
        self.n_pid = n_pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_dim = d_model

        self.r = r
        self.gamma = gamma

        # Rasch 题目难度相关嵌入
        if self.n_pid > 0:
            if "scalar" in emb_type:
                # 标量难度
                self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            else:
                # 向量难度（默认）
                self.difficult_param = nn.Embedding(self.n_pid + 1, embed_dim)
            self.skill_embed_diff = nn.Embedding(self.num_c + 1, embed_dim)
            self.interaction_embed_diff = nn.Embedding(2 * self.num_c + 1, embed_dim)

        # 技能嵌入层
        self.skill_embed = nn.Embedding(self.num_c, embed_dim)
        if self.separate_qa:
            self.interaction_embed = nn.Embedding(2 * self.num_c + 1, embed_dim)
        else:
            self.interaction_embed = nn.Embedding(2, embed_dim)

        # 单流 Transformer 架构
        self.encoder = Architecture(
            num_blocks=num_blocks,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_attn_heads,
            dropout=dropout,
            kq_same=self.kq_same,
            r=r,
            gamma=gamma,
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model + embed_dim, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )

        self.reset()

    def reset(self):
        """将 Rasch 题目难度参数初始化为 0。"""
        if self.n_pid > 0:
            for param in self.parameters():
                if param.size(0) == self.n_pid + 1:
                    torch.nn.init.constant_(param, 0.0)

    def base_emb(
        self, skill_seq: torch.Tensor, response_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算基础技能嵌入与交互嵌入。

        Args:
            skill_seq: 技能ID序列 [bs, seq]
            response_seq: 响应序列 [bs, seq]

        Returns:
            skill_embed_data: 技能嵌入
            interaction_embed_data: 交互嵌入
        """
        skill_embed_data = self.skill_embed(skill_seq)  # c_ct
        if self.separate_qa:
            interaction_seq = skill_seq + self.num_c * response_seq
            interaction_embed_data = self.interaction_embed(interaction_seq)
        else:
            # e_(ct,rt) = g_rt + c_ct
            interaction_embed_data = (
                self.interaction_embed(response_seq) + skill_embed_data
            )
        return skill_embed_data, interaction_embed_data

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        pid_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            sequence: 技能ID序列 [bs, seq]
            response: 响应序列 [bs, seq]
            pid_data: 题目ID序列 [bs, seq]（Rasch 模型用）

        Returns:
            preds: 预测概率 [bs, seq]

        说明：
            输出语义为同位置（same_position）：preds[:, t] 基于 skill[0:t+1] 与
            interaction[0:t]（因果掩码屏蔽当前及未来响应）预测 response[t]。
        """
        emb_type = self.emb_type

        # 基础嵌入
        skill_embed_data, interaction_embed_data = self.base_emb(sequence, response)

        # Rasch 题目难度增强
        if self.n_pid > 0 and "norasch" not in emb_type:
            skill_diff = self.skill_embed_diff(sequence)  # d_ct
            pid_embed = self.difficult_param(pid_data)  # uq（题目难度）

            # 技能编码：uq * d_ct + c_ct
            skill_embed_data = skill_embed_data + pid_embed * skill_diff

            if "aktrasch" in emb_type:
                interaction_diff = self.interaction_embed_diff(response)  # h_rt
                # uq * (h_rt + d_ct) + e_(ct,rt)
                interaction_embed_data = interaction_embed_data + pid_embed * (
                    interaction_diff + skill_diff
                )

        # 单流 Transformer
        knowledge_state = self.encoder(skill_embed_data, interaction_embed_data)

        # 拼接知识状态与技能嵌入后过输出层
        concat = torch.cat([knowledge_state, skill_embed_data], dim=-1)
        logits = self.output_layer(concat).squeeze(-1)
        preds = torch.sigmoid(logits)
        return preds
