from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.config_mamba import MambaConfig

from .mamba_modules import MambaEncoder


class TimeEncoder(nn.Module):
    """Cosine 时间编码器"""

    def __init__(
        self,
        time_dim: int,
        parameter_requires_grad: bool = True,
        same_timestamp_set: bool = False,
        pos_encoder_set: bool = False,
        add_pos_encoding: bool = False,
        beta: float = 9,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(0, beta, time_dim, dtype=np.float32)
            ).reshape(time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

        self.same_timestamp_set = same_timestamp_set
        self.pos_encoder_set = pos_encoder_set
        self.add_pos_encoding = add_pos_encoding

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        timestamps: Tensor, shape (batch_size, seq_len)
        Returns: Tensor, shape (batch_size, seq_len, time_dim)
        """
        bt_sz, seq_len = timestamps.shape
        if self.pos_encoder_set:
            timestamps = torch.arange(seq_len).repeat(bt_sz, 1).to(timestamps)
        if self.same_timestamp_set:
            timestamps = timestamps.new_ones(timestamps.shape)

        timestamps = timestamps.unsqueeze(dim=2)
        output = torch.cos(self.w(timestamps))

        if self.add_pos_encoding:
            positions = torch.arange(seq_len).repeat(bt_sz, 1).to(timestamps)
            positions = positions.unsqueeze(dim=2)
            output += torch.cos(self.w(positions))

        return output


class DualViewTimeEncoder(nn.Module):
    """双流时间编码器：将时间差分解为短期（相邻间隔）和长期（距参考点间隔）
    两个视图，通过可学习的个性化门控自适应融合。
    """

    def __init__(
        self,
        time_dim: int,
        context_dim: int,
        parameter_requires_grad: bool = True,
        beta_short: float = 2.5,
        beta_long: float = 2.0,
    ):
        super().__init__()
        self.time_dim = time_dim

        self.short_encoder = TimeEncoder(
            time_dim=time_dim,
            parameter_requires_grad=parameter_requires_grad,
            beta=beta_short,
        )
        self.long_encoder = TimeEncoder(
            time_dim=time_dim,
            parameter_requires_grad=parameter_requires_grad,
            beta=beta_long,
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        padded_nodes_neighbor_times: torch.Tensor,
        node_interact_times: torch.Tensor,
        padded_nodes_neighbor_ids: torch.Tensor,
        user_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            padded_nodes_neighbor_times: [B, S] 邻居序列时间戳（或时间差）
            node_interact_times: [B] 当前交互时间戳（或累计时间参考点）
            padded_nodes_neighbor_ids: [B, S] 邻居节点 ID（用于 padding mask）
            user_context: [B, context_dim] 用于生成个性化门控

        Returns:
            [B, S, time_dim] 双流融合后的时间特征
        """
        # 短期：序列内部相邻时间差
        delta_short = torch.diff(
            padded_nodes_neighbor_times,
            dim=1,
            prepend=padded_nodes_neighbor_times[:, :1],
        )
        delta_short = torch.clamp(torch.log1p(delta_short.abs()), max=15.0)

        # 长期：距当前交互时间点的绝对时间差
        delta_long = node_interact_times.unsqueeze(1) - padded_nodes_neighbor_times
        delta_long = torch.clamp(torch.log1p(delta_long.abs()), max=15.0)

        short_feat = self.short_encoder(delta_short)
        long_feat = self.long_encoder(delta_long)

        # 个性化门控
        gamma = torch.sigmoid(self.gate_mlp(user_context))  # [B, 1]

        # 自适应融合
        time_feat = (
            gamma.unsqueeze(1) * short_feat + (1 - gamma.unsqueeze(1)) * long_feat
        )

        # padding 位置置零
        time_feat[padded_nodes_neighbor_ids == 0] = 0.0

        return time_feat


class SkillPairCrossEffect(nn.Module):
    """技能对感知的 Hawkes 风格交叉效应模块。

    对齐 HawkesKT 的两个核心参数:
    - α (互激励): 历史 (skill, response) 对 target skill 的即时交叉效应强度
    - β (自适应衰减): 不同 (history_event, target_skill) 对的不同遗忘速率

    使用双线性分解: α(history, target) ≈ P(hist) · Q(target)
    等价于 HawkesKT 的矩阵分解重参数化 (P_A · Q_A^T)，
    参数量 O(2|S|·D) 而非 O(|S|²)。

    正确性调制: 将 correctness 拼接到 skill 特征上，
    使 (skill_A, 正确) 和 (skill_A, 错误) 产生不同的 α/β。
    """

    def __init__(self, skill_feat_dim: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # α: 互激励 (即时交叉效应)
        self.P_alpha = nn.Linear(skill_feat_dim + 1, embed_dim, bias=False)
        self.Q_alpha = nn.Linear(skill_feat_dim, embed_dim, bias=False)
        self.alpha_out = nn.Linear(embed_dim, 1, bias=True)

        # β: 自适应衰减 (遗忘速率)
        self.P_beta = nn.Linear(skill_feat_dim + 1, embed_dim, bias=False)
        self.Q_beta = nn.Linear(skill_feat_dim, embed_dim, bias=False)
        self.beta_out = nn.Linear(embed_dim, 1, bias=True)

        # α 初始偏向 1 (sigmoid(1) ≈ 0.73): 大多数历史交互有正效应
        nn.init.zeros_(self.alpha_out.weight)
        nn.init.constant_(self.alpha_out.bias, 1.0)
        # β 初始偏向 0 (sigmoid(0) = 0.5): 中等默认衰减
        nn.init.zeros_(self.beta_out.weight)
        nn.init.constant_(self.beta_out.bias, 0.0)

    def forward(
        self,
        history_skill_feats: torch.Tensor,
        target_skill_feat: torch.Tensor,
        correctness: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history_skill_feats: [B, S, D_skill] 历史交互的技能特征
            target_skill_feat:   [B, D_skill]    目标题目的技能特征
            correctness:         [B, S]           历史交互的正确性 (0/1)
            padding_mask:        [B, S, 1]        1=有效位置, 0=padding
        Returns:
            alpha_weights: [B, S, 1] 交叉效应权重 (用于加权 time channel)
            beta_rates:    [B, S, 1] 自适应衰减率 (用于调制 dt)
        """
        # 拼接技能特征 + 正确性
        hist_input = torch.cat(
            [history_skill_feats, correctness.unsqueeze(-1)], dim=-1
        )  # [B, S, D_skill+1]

        # α: 双线性分解 P(history) · Q(target)
        h_alpha = self.P_alpha(hist_input)           # [B, S, D_emb]
        t_alpha = self.Q_alpha(target_skill_feat)    # [B, D_emb]
        alpha_hidden = h_alpha * t_alpha.unsqueeze(1)  # [B, S, D_emb]
        alpha_weights = torch.sigmoid(self.alpha_out(alpha_hidden))  # [B, S, 1]

        # β: 双线性分解 (独立参数)
        h_beta = self.P_beta(hist_input)             # [B, S, D_emb]
        t_beta = self.Q_beta(target_skill_feat)      # [B, D_emb]
        beta_hidden = h_beta * t_beta.unsqueeze(1)   # [B, S, D_emb]
        beta_rates = torch.sigmoid(self.beta_out(beta_hidden))  # [B, S, 1]

        # padding 位置: α=1 (不改变), β=0 (不额外衰减)
        alpha_weights = alpha_weights * padding_mask + (1.0 - padding_mask)
        beta_rates = beta_rates * padding_mask

        return alpha_weights, beta_rates

class MLPPredictor(nn.Module):
    """MLP 预测头。

    拼接 src_emb, dst_emb 后过 MLP。
    """

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, src_emb, dst_emb, question_ids):
        x = torch.cat([src_emb, dst_emb], dim=-1)
        return self.net(x)


class NeighborCooccurrenceEncoder(nn.Module):
    """邻居共现特征编码器。

    统计 src 和 dst 邻居序列中节点的共现次数，通过 MLP 编码为特征向量。
    """

    def __init__(
        self,
        neighbor_co_occurrence_feat_dim: int,
    ):
        super().__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(
                in_features=1,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.neighbor_co_occurrence_feat_dim,
                out_features=self.neighbor_co_occurrence_feat_dim,
            ),
        )

    def count_nodes_appearances(
        self,
        src_padded_nodes_neighbor_ids: torch.Tensor,
        dst_padded_nodes_neighbor_ids: torch.Tensor,
    ):
        """统计 src 和 dst 邻居序列中节点的出现次数"""
        src_ids = src_padded_nodes_neighbor_ids.long()
        dst_ids = dst_padded_nodes_neighbor_ids.long()

        # 自身计数：src[i] 在 src 序列中出现几次
        src_self = (src_ids.unsqueeze(2) == src_ids.unsqueeze(1)).sum(dim=2).float()
        dst_self = (dst_ids.unsqueeze(2) == dst_ids.unsqueeze(1)).sum(dim=2).float()

        # 交叉计数：src[i] 在 dst 序列中出现几次，反之亦然
        src_cross = (src_ids.unsqueeze(2) == dst_ids.unsqueeze(1)).sum(dim=2).float()
        dst_cross = (dst_ids.unsqueeze(2) == src_ids.unsqueeze(1)).sum(dim=2).float()

        # padding 位置（id == 0）置零
        src_mask = (src_ids != 0).float()
        dst_mask = (dst_ids != 0).float()
        src_self = src_self * src_mask
        src_cross = src_cross * src_mask
        dst_self = dst_self * dst_mask
        dst_cross = dst_cross * dst_mask

        src_appearances = torch.stack([src_self, src_cross], dim=2)
        dst_appearances = torch.stack([dst_self, dst_cross], dim=2)

        return src_appearances, dst_appearances

    def forward(
        self,
        src_padded_nodes_neighbor_ids: torch.Tensor,
        dst_padded_nodes_neighbor_ids: torch.Tensor,
    ):
        src_padded_nodes_appearances, dst_padded_nodes_appearances = (
            self.count_nodes_appearances(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            )
        )

        src_padded_nodes_neighbor_co_occurrence_features = (
            self.neighbor_co_occurrence_encode_layer(
                src_padded_nodes_appearances.unsqueeze(dim=-1)
            ).sum(dim=2)
        )
        dst_padded_nodes_neighbor_co_occurrence_features = (
            self.neighbor_co_occurrence_encode_layer(
                dst_padded_nodes_appearances.unsqueeze(dim=-1)
            ).sum(dim=2)
        )

        return (
            src_padded_nodes_neighbor_co_occurrence_features,
            dst_padded_nodes_neighbor_co_occurrence_features,
        )


class DyGMamba(nn.Module):
    """DyGMamba 知识追踪模型"""

    def __init__(
        self,
        data_metadata: dict[str, Any],
        *,
        device: Any,
        time_feat_dim: int,
        channel_embedding_dim: int,
        num_layers: int,
        dropout: float,
        max_input_sequence_length: int,
        remove_time_channel: bool = False,
        dual_view_time: bool = True,
        hawkes_cross_dim: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        time_mamba: bool = False,
        no_selective: bool = False,
        plain_mamba: bool = False,
    ):
        super().__init__()

        # 模型超参数
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.remove_time_channel = remove_time_channel
        self.num_channels = 4 if self.remove_time_channel else 5

        # 元数据
        num_questions = data_metadata["num_questions"]
        num_users = data_metadata["num_users"]
        self.question_id_offset = data_metadata["question_id_offset"]
        self.user_id_offset = data_metadata["user_id_offset"]
        self.num_nodes = self.user_id_offset + num_users

        # 构建 node_raw_features
        question_features = data_metadata["question_features"]
        question_features = np.asarray(question_features, dtype=np.float32)

        node_feature_dim = question_features.shape[1]
        node_raw_features = np.zeros(
            (self.num_nodes, node_feature_dim), dtype=np.float32
        )
        node_raw_features[
            self.question_id_offset : self.question_id_offset + num_questions,
            :,
        ] = question_features

        self.register_buffer(
            "node_raw_features",
            torch.from_numpy(node_raw_features),
            persistent=False,
        )

        self.node_feat_dim = node_feature_dim
        self.edge_feat_dim = 1

        # 时间编码器
        self.dual_view_time = dual_view_time
        if self.dual_view_time:
            self.time_encoder = DualViewTimeEncoder(
                time_dim=self.time_feat_dim,
                context_dim=self.node_feat_dim,
            )
            self.dt_time_encoder = DualViewTimeEncoder(
                time_dim=self.time_feat_dim,
                context_dim=self.node_feat_dim,
                beta_short=3.0,
                beta_long=2.0,
            )
        else:
            self.time_encoder = TimeEncoder(time_dim=self.time_feat_dim)
            self.dt_time_encoder = TimeEncoder(time_dim=self.time_feat_dim, beta=3.0)
        self.projection_dt = nn.Linear(
            in_features=self.time_feat_dim,
            out_features=self.num_channels * self.channel_embedding_dim,
            bias=True,
        )

        # Hawkes 风格技能对交叉效应
        self.hawkes_cross_dim = hawkes_cross_dim
        if self.hawkes_cross_dim > 0:
            self.hawkes_cross = SkillPairCrossEffect(
                skill_feat_dim=node_feature_dim,
                embed_dim=self.hawkes_cross_dim,
            )

        # 通道编码
        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(
            neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim,
        )

        self.projection_layer = nn.ModuleDict(
            {
                "node": nn.Linear(
                    in_features=self.node_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "edge": nn.Linear(
                    in_features=self.edge_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "time": nn.Linear(
                    in_features=self.time_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "neighbor_co_occurrence": nn.Linear(
                    in_features=self.neighbor_co_occurrence_feat_dim,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
                "hint_count": nn.Linear(
                    in_features=1,
                    out_features=self.channel_embedding_dim,
                    bias=True,
                ),
            }
        )

        # Mamba 编码器
        config = MambaConfig()
        config.d_model = self.num_channels * self.channel_embedding_dim
        config.n_layer = self.num_layers
        config.ssm_cfg = {
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "time_mamba": time_mamba,
            "no_selective": no_selective,
            "plain_mamba": plain_mamba,
        }
        config.rms_norm = False
        config.fused_add_norm = False
        config.residual_in_fp32 = False

        factory_kwargs = {"device": device, "dtype": None}
        self.encoder = MambaEncoder(config, **factory_kwargs)

        self.src_gate = nn.Linear(
            self.num_channels * self.channel_embedding_dim, 1, bias=False
        )
        self.dst_gate = nn.Linear(
            self.num_channels * self.channel_embedding_dim, 1, bias=False
        )

        self.dropout_layer = nn.Dropout(self.dropout)

        self.link_predictor = MLPPredictor(
            embed_dim=self.num_channels * self.channel_embedding_dim,
            dropout=self.dropout,
        )

    def _build_padded_sequences(self, batch: dict[str, torch.Tensor]):
        """构建 src/dst 的 padded 序列，返回各通道特征和 dt 特征。"""
        device = batch["user"].device
        batch_size = batch["user"].shape[0]

        src_node_ids = batch["user"]
        dst_node_ids = batch["question"]
        node_interact_times = batch["time"]

        src_neighbor_ids = batch["src_neighbor_node_ids"]
        src_neighbor_times = batch["src_neighbor_times"]
        src_neighbor_edges = batch["src_neighbor_edge_feats"]
        dst_neighbor_ids = batch["dst_neighbor_node_ids"]
        dst_neighbor_times = batch["dst_neighbor_times"]
        dst_neighbor_edges = batch["dst_neighbor_edge_feats"]
        src_neighbor_hint_counts = batch["src_neighbor_hint_count"]
        dst_neighbor_hint_counts = batch["dst_neighbor_hint_count"]

        src_S = min(src_neighbor_ids.shape[1], self.max_input_sequence_length - 1)
        dst_S = min(dst_neighbor_ids.shape[1], self.max_input_sequence_length - 1)
        src_seq_len = src_S + 2
        dst_seq_len = dst_S + 2

        src_padded_nodes_neighbor_ids = torch.zeros(
            (batch_size, src_seq_len), dtype=torch.long, device=device
        )
        src_padded_nodes_edge_ids = torch.zeros(
            (batch_size, src_seq_len), dtype=torch.float32, device=device
        )
        src_padded_nodes_neighbor_times = torch.zeros(
            (batch_size, src_seq_len), dtype=torch.float32, device=device
        )
        src_padded_nodes_neighbor_ids[:, 0] = src_node_ids
        src_padded_nodes_neighbor_ids[:, 1 : src_S + 1] = src_neighbor_ids[:, :src_S]
        src_padded_nodes_edge_ids[:, 1 : src_S + 1] = src_neighbor_edges[:, :src_S]
        src_padded_nodes_neighbor_times[:, 0] = node_interact_times
        src_padded_nodes_neighbor_times[:, 1 : src_S + 1] = src_neighbor_times[
            :, :src_S
        ]

        src_padded_nodes_hint_count = torch.zeros(
            batch_size, src_seq_len, device=device
        )
        src_padded_nodes_hint_count[:, 1 : src_S + 1] = src_neighbor_hint_counts[
            :, :src_S
        ]

        dst_padded_nodes_neighbor_ids = torch.zeros(
            (batch_size, dst_seq_len), dtype=torch.long, device=device
        )
        dst_padded_nodes_edge_ids = torch.zeros(
            (batch_size, dst_seq_len), dtype=torch.float32, device=device
        )
        dst_padded_nodes_neighbor_times = torch.zeros(
            (batch_size, dst_seq_len), dtype=torch.float32, device=device
        )
        dst_padded_nodes_neighbor_ids[:, 0] = dst_node_ids
        dst_padded_nodes_neighbor_ids[:, 1 : dst_S + 1] = dst_neighbor_ids[:, :dst_S]
        dst_padded_nodes_edge_ids[:, 1 : dst_S + 1] = dst_neighbor_edges[:, :dst_S]
        dst_padded_nodes_neighbor_times[:, 0] = node_interact_times
        dst_padded_nodes_neighbor_times[:, 1 : dst_S + 1] = dst_neighbor_times[
            :, :dst_S
        ]

        dst_padded_nodes_hint_count = torch.zeros(
            batch_size, dst_seq_len, device=device
        )
        dst_padded_nodes_hint_count[:, 1 : dst_S + 1] = dst_neighbor_hint_counts[
            :, :dst_S
        ]

        return {
            "batch_size": batch_size,
            "src_seq_len": src_seq_len,
            "dst_seq_len": dst_seq_len,
            "src_padded_nodes_neighbor_ids": src_padded_nodes_neighbor_ids,
            "src_padded_nodes_edge_ids": src_padded_nodes_edge_ids,
            "src_padded_nodes_neighbor_times": src_padded_nodes_neighbor_times,
            "src_padded_nodes_hint_count": src_padded_nodes_hint_count,
            "dst_padded_nodes_neighbor_ids": dst_padded_nodes_neighbor_ids,
            "dst_padded_nodes_edge_ids": dst_padded_nodes_edge_ids,
            "dst_padded_nodes_neighbor_times": dst_padded_nodes_neighbor_times,
            "dst_padded_nodes_hint_count": dst_padded_nodes_hint_count,
            "node_interact_times": node_interact_times,
        }

    def compute_src_dst_node_temporal_embeddings(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        """计算 source (用户) 和 destination (题目) 的时间嵌入。"""
        padded = self._build_padded_sequences(batch)
        batch_size = padded["batch_size"]
        src_padded_nodes_neighbor_ids = padded["src_padded_nodes_neighbor_ids"]
        src_padded_nodes_edge_ids = padded["src_padded_nodes_edge_ids"]
        src_padded_nodes_neighbor_times = padded["src_padded_nodes_neighbor_times"]
        src_padded_nodes_hint_count = padded["src_padded_nodes_hint_count"]
        dst_padded_nodes_neighbor_ids = padded["dst_padded_nodes_neighbor_ids"]
        dst_padded_nodes_edge_ids = padded["dst_padded_nodes_edge_ids"]
        dst_padded_nodes_neighbor_times = padded["dst_padded_nodes_neighbor_times"]
        dst_padded_nodes_hint_count = padded["dst_padded_nodes_hint_count"]
        node_interact_times = padded["node_interact_times"]

        # 门控上下文
        # 用户侧：聚合该用户历史答题的题目 skill 特征的均值作为用户画像
        # 题目侧：使用题目自身的特征
        if self.dual_view_time:
            src_history_feats = self.node_raw_features[
                src_padded_nodes_neighbor_ids[:, 1:]
            ]
            src_history_mask = (
                (src_padded_nodes_neighbor_ids[:, 1:] != 0).float().unsqueeze(-1)
            )
            src_context = (src_history_feats * src_history_mask).sum(
                dim=1
            ) / src_history_mask.sum(dim=1).clamp(min=1.0)
            dst_context = self.node_raw_features[dst_padded_nodes_neighbor_ids[:, 0]]
        else:
            src_context = None
            dst_context = None

        # 邻居共现特征
        (
            src_padded_nodes_neighbor_co_occurrence_features,
            dst_padded_nodes_neighbor_co_occurrence_features,
        ) = self.neighbor_co_occurrence_encoder(
            src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
        )

        # 获取特征
        (
            src_padded_nodes_neighbor_node_raw_features,
            src_padded_nodes_edge_raw_features,
            src_padded_nodes_neighbor_time_features,
        ) = self.get_features(
            node_interact_times=node_interact_times,
            padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            padded_nodes_edge_ids=src_padded_nodes_edge_ids,
            padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
            time_encoder=self.time_encoder,
            user_context=src_context,
        )

        # ─── Hawkes 风格技能对交叉效应 (α: 互激励) ───
        # 仅作用于 src 序列：历史题目 skill → 目标题目 skill
        if self.hawkes_cross_dim > 0:
            # 目标题目的 skill 特征
            target_skill_feat = self.node_raw_features[
                dst_padded_nodes_neighbor_ids[:, 0].long()
            ]  # [B, D_skill]

            # 历史位置 pos 1..K 的 skill 特征和正确性
            src_hist_skill = src_padded_nodes_neighbor_node_raw_features[:, 1:]  # [B, S, D_skill]
            src_hist_correct = src_padded_nodes_edge_raw_features[:, 1:].squeeze(-1)  # [B, S]

            # padding mask: 有效位置=1, padding=0
            src_hist_mask = (
                (src_padded_nodes_neighbor_ids[:, 1:] != 0).unsqueeze(-1).float()
            )  # [B, S, 1]

            src_alpha, src_beta = self.hawkes_cross(
                src_hist_skill, target_skill_feat, src_hist_correct, src_hist_mask
            )

            # α 乘到 time channel: 交叉效应强的技能对保留更多时间信息
            src_padded_nodes_neighbor_time_features = torch.cat([
                src_padded_nodes_neighbor_time_features[:, :1],
                src_padded_nodes_neighbor_time_features[:, 1:] * src_alpha,
            ], dim=1)

        (
            dst_padded_nodes_neighbor_node_raw_features,
            dst_padded_nodes_edge_raw_features,
            dst_padded_nodes_neighbor_time_features,
        ) = self.get_features(
            node_interact_times=node_interact_times,
            padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            padded_nodes_edge_ids=dst_padded_nodes_edge_ids,
            padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
            time_encoder=self.time_encoder,
            user_context=dst_context,
        )

        # Delta-time 特征
        src_padded_dt_features = self.get_dt_features(
            padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
            padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
            time_encoder=self.dt_time_encoder,
            user_context=src_context if self.dual_view_time else None,
        )

        # ─── Hawkes 风格技能对交叉效应 (β: 自适应衰减) ───
        # β 大 → dt 被放大 → Mamba Δ 增大 → 该位置遗忘加快
        if self.hawkes_cross_dim > 0:
            src_padded_dt_features = torch.cat([
                src_padded_dt_features[:, :1],
                src_padded_dt_features[:, 1:] * (1.0 + src_beta),
            ], dim=1)
        dst_padded_dt_features = self.get_dt_features(
            padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
            padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
            time_encoder=self.dt_time_encoder,
            user_context=dst_context if self.dual_view_time else None,
        )
        src_padded_dt_features = self.projection_dt(src_padded_dt_features)
        dst_padded_dt_features = self.projection_dt(dst_padded_dt_features)

        # 投影各通道
        src_padded_nodes_neighbor_node_raw_features = self.projection_layer["node"](
            src_padded_nodes_neighbor_node_raw_features
        )
        src_padded_nodes_edge_raw_features = self.projection_layer["edge"](
            src_padded_nodes_edge_raw_features
        )
        src_padded_nodes_neighbor_time_features = self.projection_layer["time"](
            src_padded_nodes_neighbor_time_features
        )
        src_padded_nodes_neighbor_co_occurrence_features = self.projection_layer[
            "neighbor_co_occurrence"
        ](src_padded_nodes_neighbor_co_occurrence_features)
        src_padded_nodes_hint_count_features = self.projection_layer["hint_count"](
            src_padded_nodes_hint_count.unsqueeze(-1)
        )

        dst_padded_nodes_neighbor_node_raw_features = self.projection_layer["node"](
            dst_padded_nodes_neighbor_node_raw_features
        )
        dst_padded_nodes_edge_raw_features = self.projection_layer["edge"](
            dst_padded_nodes_edge_raw_features
        )
        dst_padded_nodes_neighbor_time_features = self.projection_layer["time"](
            dst_padded_nodes_neighbor_time_features
        )
        dst_padded_nodes_neighbor_co_occurrence_features = self.projection_layer[
            "neighbor_co_occurrence"
        ](dst_padded_nodes_neighbor_co_occurrence_features)
        dst_padded_nodes_hint_count_features = self.projection_layer["hint_count"](
            dst_padded_nodes_hint_count.unsqueeze(-1)
        )

        src_seq_len = src_padded_nodes_neighbor_node_raw_features.shape[1]
        dst_seq_len = dst_padded_nodes_neighbor_node_raw_features.shape[1]

        # Stack 通道特征
        src_padded_data = [
            src_padded_nodes_neighbor_node_raw_features,
            src_padded_nodes_edge_raw_features,
            src_padded_nodes_neighbor_co_occurrence_features,
            src_padded_nodes_hint_count_features,
        ]
        dst_padded_data = [
            dst_padded_nodes_neighbor_node_raw_features,
            dst_padded_nodes_edge_raw_features,
            dst_padded_nodes_neighbor_co_occurrence_features,
            dst_padded_nodes_hint_count_features,
        ]
        if not self.remove_time_channel:
            # 插入到第 3 个位置（index 2），保持原有顺序
            src_padded_data.insert(2, src_padded_nodes_neighbor_time_features)
            dst_padded_data.insert(2, dst_padded_nodes_neighbor_time_features)
        src_padded_data = torch.stack(src_padded_data, dim=2)
        dst_padded_data = torch.stack(dst_padded_data, dim=2)
        src_padded_data = src_padded_data.reshape(
            batch_size, src_seq_len, self.num_channels * self.channel_embedding_dim
        )
        dst_padded_data = dst_padded_data.reshape(
            batch_size, dst_seq_len, self.num_channels * self.channel_embedding_dim
        )

        # Mamba 编码
        src_padded_data = self.encoder(src_padded_data, dts=src_padded_dt_features)
        dst_padded_data = self.encoder(dst_padded_data, dts=dst_padded_dt_features)

        # Soft Attention 池化
        src_routing_weights = F.softmax(self.src_gate(src_padded_data), dim=1)
        dst_routing_weights = F.softmax(self.dst_gate(dst_padded_data), dim=1)

        src_node_embeddings = (src_padded_data * src_routing_weights).sum(dim=1)
        dst_node_embeddings = (dst_padded_data * dst_routing_weights).sum(dim=1)

        return src_node_embeddings, dst_node_embeddings

    def get_features(
        self,
        node_interact_times: torch.Tensor,
        padded_nodes_neighbor_ids: torch.Tensor,
        padded_nodes_edge_ids: torch.Tensor,
        padded_nodes_neighbor_times: torch.Tensor,
        time_encoder: nn.Module,
        user_context: torch.Tensor | None = None,
    ):
        """获取 node、edge、time 特征"""
        # node features
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[
            padded_nodes_neighbor_ids
        ]

        # edge features
        padded_nodes_edge_raw_features = padded_nodes_edge_ids.unsqueeze(-1)

        # time features
        if isinstance(time_encoder, DualViewTimeEncoder):
            padded_nodes_neighbor_time_features = time_encoder(
                padded_nodes_neighbor_times=padded_nodes_neighbor_times,
                node_interact_times=node_interact_times,
                padded_nodes_neighbor_ids=padded_nodes_neighbor_ids,
                user_context=user_context,
            )
        else:
            padded_nodes_neighbor_time_features = time_encoder(
                timestamps=node_interact_times.unsqueeze(1)
                - padded_nodes_neighbor_times
            )
            padded_nodes_neighbor_time_features[padded_nodes_neighbor_ids == 0] = 0.0

        return (
            padded_nodes_neighbor_node_raw_features,
            padded_nodes_edge_raw_features,
            padded_nodes_neighbor_time_features,
        )

    def get_dt_features(
        self,
        padded_nodes_neighbor_times: torch.Tensor,
        padded_nodes_neighbor_ids: torch.Tensor,
        time_encoder: nn.Module,
        user_context: torch.Tensor | None = None,
    ):
        dt = torch.diff(padded_nodes_neighbor_times, dim=1)
        dt = F.pad(dt, (1, 0), value=1.0)
        dt = dt.abs()

        if isinstance(time_encoder, DualViewTimeEncoder) and user_context is not None:
            # 短期：二阶时间差（交互节奏的加速度）
            # 长期：一阶 dt 本身的绝对幅度（原始间隔大小）
            padded_nodes_neighbor_time_features = time_encoder(
                padded_nodes_neighbor_times=dt,
                node_interact_times=torch.zeros(
                    padded_nodes_neighbor_times.shape[0],
                    device=dt.device,
                ),
                padded_nodes_neighbor_ids=padded_nodes_neighbor_ids,
                user_context=user_context,
            )
        else:
            padded_nodes_neighbor_time_features = time_encoder(timestamps=dt)
            padded_nodes_neighbor_time_features[padded_nodes_neighbor_ids == 0] = 0.0

        return padded_nodes_neighbor_time_features

    def forward(self, batch):
        src_emb, dst_emb = self.compute_src_dst_node_temporal_embeddings(batch)
        src_emb = self.dropout_layer(src_emb)
        dst_emb = self.dropout_layer(dst_emb)
        logits = (
            self.link_predictor(src_emb, dst_emb, batch["question"].long())
            .squeeze(dim=-1)
            .float()
        )
        return {
            "logits": logits,
            "src_embeddings": src_emb,
            "dst_embeddings": dst_emb,
        }
