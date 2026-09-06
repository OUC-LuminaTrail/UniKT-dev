import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv, Linear

from ..layers import GeneralInteraction, HGNNConv, HistoryRecap, Hypergraph


class HeteroGNN(nn.Module):
    """基于 HGT 的异质图神经网络模块。

    使用 Heterogeneous Graph Transformer 进行多层异构图聚合。

    Args:
        embedding_dim: 节点嵌入维度
        n_hop: GNN 层数
        heads: 注意力头数
        dropout: Dropout 概率
        metadata: 异构图元数据

    Example:
        >>> gnn = HeteroGNN(embedding_dim=128, n_hop=2, heads=4, dropout=0.2, metadata=metadata)
        >>> output = gnn(x_dict, edge_index_dict)
    """

    def __init__(
        self,
        embedding_dim: int,
        n_hop: int,
        heads: int,
        dropout: float,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        output_node_types: tuple[str, ...] | None = ("question", "skill"),
    ) -> None:
        super().__init__()
        self.n_hop = n_hop
        self.heads = heads
        self.dropout = dropout
        self.output_node_types = output_node_types
        self.convs = torch.nn.ModuleList()

        for _ in range(n_hop):
            conv = HGTConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                metadata=metadata,
                heads=heads,
            )
            self.convs.append(conv)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            x_dict: 节点特征字典
            edge_index_dict: 边索引字典

        Returns:
            聚合后的节点表示字典
        """
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            is_last = i == self.n_hop - 1
            new_x_dict = {}
            for node_type, x in x_dict.items():
                if x is not None:
                    needs_post = (
                        self.output_node_types is None
                        or node_type in self.output_node_types
                        or not is_last
                    )
                    if needs_post:
                        x = F.gelu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                new_x_dict[node_type] = x
            x_dict = new_x_dict
        return x_dict


class HyperGNN(nn.Module):
    """双层 HGNN 超图神经网络

    使用 ``model.layers.hypergraph.HGNNConv`` 实现双层超图卷积，支持加权超图。

    数学公式：
        X' = σ(D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ)

    其中：
        - X 是输入顶点特征矩阵
        - H 是超图关联矩阵
        - W_e 是超边权重对角矩阵（可自定义或默认为单位矩阵）
        - D_v 是顶点度数对角矩阵
        - D_e 是超边度数对角矩阵
        - Θ 是可学习参数
    """

    def __init__(
        self,
        in_ch: int,
        n_hid: int,
        n_class: int,
        dropout: float = 0.0,
        use_edge_weights: bool = True,
    ) -> None:
        super().__init__()
        self.use_edge_weights = use_edge_weights

        # First conv layer: aggregates direct-neighbor question features
        # is_last=False enables the built-in activation and dropout
        self.hgc1 = HGNNConv(
            in_ch, n_hid, bias=True, use_bn=False, drop_rate=dropout, is_last=False
        )
        # Second conv layer: aggregates indirect-neighbor question features
        # is_last=True skips the built-in activation/dropout; ReLU is applied manually in forward
        self.hgc2 = HGNNConv(
            n_hid, n_class, bias=True, use_bn=False, drop_rate=dropout, is_last=True
        )

    def forward(self, x: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入特征矩阵 [num_vertices, in_ch]
            hg: 超图结构

        Returns:
            输出特征矩阵 [num_vertices, n_class]
        """
        x1 = self.hgc1(x, hg)
        x2 = F.relu(self.hgc2(x1, hg))

        return x2


class VariationalBottleneck(nn.Module):
    """Diagonal-Gaussian stochastic encoder used by the graph bottlenecks."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_stats = nn.Linear(dim, dim * 2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.to_stats(x).chunk(2, dim=-1)
        # Keep the stochastic code numerically stable and bounded away from an
        # effectively deterministic continuous channel.
        logvar = logvar.clamp(min=-8.0, max=8.0)
        if self.training:
            code = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            code = mean
        return code, mean, logvar

    @staticmethod
    def rate_per_dimension(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Return KL(q(z|x) || N(0, I)) in nats per latent dimension."""
        return 0.5 * (mean.square() + logvar.exp() - logvar - 1.0).mean(dim=-1)


class GaussianConditional(nn.Module):
    """Variational conditional density used by vCLUB."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2),
        )

    def statistics(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.net(condition).chunk(2, dim=-1)
        return mean, logvar.clamp(min=-8.0, max=8.0)

    @staticmethod
    def log_prob_per_dimension(
        value: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        return -0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (value - mean).square() * torch.exp(-logvar)
        ).mean(dim=-1)


class SparseRelationDecoder(nn.Module):
    """Negative-sampled Bernoulli decoder for a fixed bipartite relation.

    All positive targets of a selected source node are evaluated exactly.
    Uniform samples from its true non-neighbours give an unbiased estimate of
    the remaining negative sum. The result is the estimated mean BCE over all
    possible targets, returned separately for every selected source node.
    """

    def __init__(self, dim: int, num_targets: int) -> None:
        super().__init__()
        if num_targets <= 0:
            raise ValueError("num_targets must be positive")
        self.num_targets = int(num_targets)
        self.source_projection = nn.Linear(dim, dim, bias=False)
        self.target_embedding = nn.Embedding(num_targets, dim)
        self.target_bias = nn.Embedding(num_targets, 1)
        nn.init.zeros_(self.target_bias.weight)

    def _score(self, source: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        target = self.target_embedding(target_ids)
        projected = self.source_projection(source)
        return (projected * target).sum(dim=-1) / math.sqrt(source.size(-1)) + (
            self.target_bias(target_ids).squeeze(-1)
        )

    @staticmethod
    def _membership(
        keys: torch.Tensor, sorted_positive_keys: torch.Tensor
    ) -> torch.Tensor:
        if sorted_positive_keys.numel() == 0:
            return torch.zeros_like(keys, dtype=torch.bool)
        locations = torch.searchsorted(sorted_positive_keys, keys)
        safe_locations = locations.clamp(max=sorted_positive_keys.numel() - 1)
        return (locations < sorted_positive_keys.numel()) & (
            sorted_positive_keys[safe_locations] == keys
        )

    def forward(
        self,
        codes: torch.Tensor,
        edge_index: torch.Tensor,
        source_ids: torch.Tensor,
        *,
        negative_samples: int,
    ) -> torch.Tensor:
        if source_ids.numel() == 0:
            return codes.new_zeros((0,))
        if negative_samples <= 0:
            raise ValueError("negative_samples must be positive")

        source_ids = source_ids.unique(sorted=True)
        num_sources = source_ids.numel()
        edge_sources, edge_targets = edge_index

        selected_mask = torch.isin(edge_sources, source_ids)
        positive_sources = edge_sources[selected_mask]
        positive_targets = edge_targets[selected_mask]
        local_positive_sources = torch.searchsorted(source_ids, positive_sources)

        positive_sums = codes.new_zeros(num_sources)
        degrees = torch.zeros(num_sources, dtype=torch.long, device=codes.device)
        if positive_sources.numel() > 0:
            positive_scores = self._score(codes[positive_sources], positive_targets)
            positive_losses = F.softplus(-positive_scores)
            positive_sums.scatter_add_(0, local_positive_sources, positive_losses)
            degrees.scatter_add_(
                0,
                local_positive_sources,
                torch.ones_like(local_positive_sources),
            )

        non_neighbour_counts = self.num_targets - degrees
        valid_local = torch.nonzero(non_neighbour_counts > 0, as_tuple=False).squeeze(
            -1
        )
        negative_sums = codes.new_zeros(num_sources)
        sampled_counts = torch.zeros(num_sources, dtype=torch.long, device=codes.device)

        if valid_local.numel() > 0:
            negative_local_sources = valid_local.repeat_interleave(negative_samples)
            negative_sources = source_ids[negative_local_sources]
            negative_targets = torch.randint(
                self.num_targets,
                (negative_sources.numel(),),
                device=codes.device,
            )

            all_positive_keys = (edge_sources * self.num_targets + edge_targets).unique(
                sorted=True
            )
            candidate_keys = negative_sources * self.num_targets + negative_targets
            collisions = self._membership(candidate_keys, all_positive_keys)
            # The relations are sparse in normal HDHKT data. The cap only
            # guards malformed or nearly-complete synthetic relations.
            for _ in range(32):
                if not collisions.any():
                    break
                negative_targets[collisions] = torch.randint(
                    self.num_targets,
                    (int(collisions.sum().item()),),
                    device=codes.device,
                )
                candidate_keys = negative_sources * self.num_targets + negative_targets
                collisions = self._membership(candidate_keys, all_positive_keys)

            # Exact fallback for a nearly-complete relation. Sampling from the
            # explicit complement preserves the intended uniform distribution
            # instead of silently dropping unresolved collisions.
            if collisions.any():
                for sample_index in torch.nonzero(collisions, as_tuple=False).flatten():
                    source_id = negative_sources[sample_index]
                    positive_for_source = edge_targets[edge_sources == source_id]
                    available = torch.ones(
                        self.num_targets, dtype=torch.bool, device=codes.device
                    )
                    available[positive_for_source] = False
                    candidates = torch.nonzero(available, as_tuple=False).flatten()
                    chosen = torch.randint(candidates.numel(), (), device=codes.device)
                    negative_targets[sample_index] = candidates[chosen]

            if negative_sources.numel() > 0:
                negative_scores = self._score(codes[negative_sources], negative_targets)
                negative_losses = F.softplus(negative_scores)
                negative_sums.scatter_add_(0, negative_local_sources, negative_losses)
                sampled_counts.scatter_add_(
                    0,
                    negative_local_sources,
                    torch.ones_like(negative_local_sources),
                )

        negative_estimate = torch.where(
            sampled_counts > 0,
            negative_sums
            * non_neighbour_counts.to(codes.dtype)
            / sampled_counts.clamp_min(1).to(codes.dtype),
            torch.zeros_like(negative_sums),
        )
        return (positive_sums + negative_estimate) / self.num_targets


class MoEFusion(nn.Module):
    """混合专家融合 (Mixture-of-Experts Fusion)。

    将两个视图的特征处理视为不同的"专家"。
    引入一个共享专家(Shared Expert)捕获共性。
    使用门控网络(Router)动态分配权重。
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim

        # Expert networks:
        # Expert 1: processes View 1
        # Expert 2: processes View 2
        # Expert 3: processes View 1 + View 2 (shared)
        self.expert1 = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.expert2 = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.expert_shared = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout)
        )

        # Router: takes both views and outputs per-expert weights
        self.router = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        # Flatten batch dimensions for Linear layers
        B_shape = view1.shape[:-1]
        v1_flat = view1.reshape(-1, self.dim)
        v2_flat = view2.reshape(-1, self.dim)

        e1 = self.expert1(v1_flat)
        e2 = self.expert2(v2_flat)
        combined = torch.cat([v1_flat, v2_flat], dim=-1)
        e_shared = self.expert_shared(combined)

        # Stack expert outputs: [N, 3, D]
        experts = torch.stack([e1, e2, e_shared], dim=1)

        weights = self.router(combined)  # [N, 3]

        # weights: [N, 3] -> [N, 3, 1]
        # experts: [N, 3, D]
        fused = torch.sum(experts * weights.unsqueeze(-1), dim=1)

        return self.norm(fused).reshape(*B_shape, self.dim)


class InformationBottleneckMoE(nn.Module):
    """Cross-channel information bottleneck deeply coupled to the MoE.

    HGT and HGNN outputs are decomposed into private and common stochastic
    codes. Neither the experts nor the router receives a raw GNN output, so the
    mixture cannot bypass the bottleneck.
    """

    def __init__(
        self,
        dim: int,
        num_skills: int,
        num_hyperedges: int,
        *,
        dropout: float,
        negative_samples: int,
        route_temperature: float,
    ) -> None:
        super().__init__()
        if negative_samples <= 0:
            raise ValueError("negative_samples must be positive")
        if route_temperature <= 0:
            raise ValueError("route_temperature must be positive")
        self.negative_samples = negative_samples
        self.route_temperature = route_temperature

        self.private1 = VariationalBottleneck(dim)
        self.private2 = VariationalBottleneck(dim)
        self.common1 = VariationalBottleneck(dim)
        self.common2 = VariationalBottleneck(dim)

        # Private codes reconstruct their own fixed relation; common codes
        # reconstruct the paired opposite relation.
        self.private1_decoder = SparseRelationDecoder(dim, num_skills)
        self.private2_decoder = SparseRelationDecoder(dim, num_hyperedges)
        self.common1_cross_decoder = SparseRelationDecoder(dim, num_hyperedges)
        self.common2_cross_decoder = SparseRelationDecoder(dim, num_skills)

        self.private1_from_view2 = GaussianConditional(dim)
        self.private2_from_view1 = GaussianConditional(dim)

        self.expert1 = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.expert2 = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.shared_expert = nn.Sequential(
            nn.Linear(dim * 4, dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.router = nn.Sequential(
            nn.Linear(dim * 4 + 4, dim),
            nn.Tanh(),
            nn.Linear(dim, 3),
        )
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _symmetric_gaussian_kl(
        mean1: torch.Tensor,
        logvar1: torch.Tensor,
        mean2: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> torch.Tensor:
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        squared_mean_difference = (mean1 - mean2).square()
        kl12 = 0.5 * (
            logvar2 - logvar1 + (var1 + squared_mean_difference) / var2 - 1.0
        ).mean(dim=-1)
        kl21 = 0.5 * (
            logvar1 - logvar2 + (var2 + squared_mean_difference) / var1 - 1.0
        ).mean(dim=-1)
        return 0.5 * (kl12 + kl21)

    @staticmethod
    def _deranged_indices(size: int, device: torch.device) -> torch.Tensor:
        if size < 2:
            return torch.arange(size, device=device)
        shift = int(torch.randint(1, size, (), device=device).item())
        return (torch.arange(size, device=device) + shift) % size

    def _club_terms(
        self,
        condition: torch.Tensor,
        value: torch.Tensor,
        estimator: GaussianConditional,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-node vCLUB penalty and detached-target fit NLL."""
        fit_mean, fit_logvar = estimator.statistics(condition.detach())
        fit_log_prob = estimator.log_prob_per_dimension(
            value.detach(), fit_mean, fit_logvar
        )
        fit_loss = -fit_log_prob.mean()

        # The CLUB penalty updates the private encoder only. The fit loss above
        # is the conditional estimator's sole gradient source, preventing the
        # estimator and representation from colluding in one backward pass.
        frozen_mean = fit_mean.detach()
        frozen_logvar = fit_logvar.detach()
        positive = estimator.log_prob_per_dimension(value, frozen_mean, frozen_logvar)
        if value.size(0) < 2:
            return torch.zeros_like(positive), fit_loss
        negative_value = value[self._deranged_indices(value.size(0), value.device)]
        negative = estimator.log_prob_per_dimension(
            negative_value, frozen_mean, frozen_logvar
        )
        return positive - negative, fit_loss

    def _fuse(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        logvars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.expert1(p1)
        e2 = self.expert2(p2)
        shared_input = torch.cat([c1, c2, c1 * c2, (c1 - c2).abs()], dim=-1)
        ec = self.shared_expert(shared_input)

        uncertainty = torch.cat(
            [logvar.exp().mean(dim=-1, keepdim=True) for logvar in logvars],
            dim=-1,
        )
        router_input = torch.cat([p1, p2, c1, c2, uncertainty], dim=-1)
        weights = F.softmax(self.router(router_input), dim=-1)
        experts = torch.stack([e1, e2, ec], dim=1)
        fused = torch.sum(experts * weights.unsqueeze(-1), dim=1)
        return self.norm(fused), weights

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        *,
        question_skill_edges: torch.Tensor | None = None,
        question_hyperedge_edges: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p1, p1_mean, p1_logvar = self.private1(view1)
        p2, p2_mean, p2_logvar = self.private2(view2)
        c1, c1_mean, c1_logvar = self.common1(view1)
        c2, c2_mean, c2_logvar = self.common2(view2)

        fused, route_weights = self._fuse(
            p1,
            p2,
            c1,
            c2,
            (p1_logvar, p2_logvar, c1_logvar, c2_logvar),
        )
        if (
            not self.training
            or source_ids is None
            or source_ids.numel() == 0
            or question_skill_edges is None
            or question_hyperedge_edges is None
        ):
            return fused, {}

        source_ids = source_ids.unique(sorted=True)
        selected_p1 = p1[source_ids]
        selected_p2 = p2[source_ids]

        p1_reconstruction = self.private1_decoder(
            p1,
            question_skill_edges,
            source_ids,
            negative_samples=self.negative_samples,
        )
        p2_reconstruction = self.private2_decoder(
            p2,
            question_hyperedge_edges,
            source_ids,
            negative_samples=self.negative_samples,
        )
        c1_cross_reconstruction = self.common1_cross_decoder(
            c1,
            question_hyperedge_edges,
            source_ids,
            negative_samples=self.negative_samples,
        )
        c2_cross_reconstruction = self.common2_cross_decoder(
            c2,
            question_skill_edges,
            source_ids,
            negative_samples=self.negative_samples,
        )

        club21, club21_fit = self._club_terms(
            view2[source_ids], selected_p1, self.private1_from_view2
        )
        club12, club12_fit = self._club_terms(
            view1[source_ids], selected_p2, self.private2_from_view1
        )

        rate_p1 = VariationalBottleneck.rate_per_dimension(
            p1_mean[source_ids], p1_logvar[source_ids]
        )
        rate_p2 = VariationalBottleneck.rate_per_dimension(
            p2_mean[source_ids], p2_logvar[source_ids]
        )
        rate_c1 = VariationalBottleneck.rate_per_dimension(
            c1_mean[source_ids], c1_logvar[source_ids]
        )
        rate_c2 = VariationalBottleneck.rate_per_dimension(
            c2_mean[source_ids], c2_logvar[source_ids]
        )
        align = self._symmetric_gaussian_kl(
            c1_mean[source_ids],
            c1_logvar[source_ids],
            c2_mean[source_ids],
            c2_logvar[source_ids],
        )

        private1_utility = -p1_reconstruction - club21 - rate_p1
        private2_utility = -p2_reconstruction - club12 - rate_p2
        common_utility = -0.5 * (
            c1_cross_reconstruction + c2_cross_reconstruction + rate_c1 + rate_c2
        )
        common_utility = common_utility - align
        utilities = torch.stack(
            [private1_utility, private2_utility, common_utility], dim=-1
        )
        information_route = F.softmax(
            utilities.detach() / self.route_temperature, dim=-1
        )
        selected_route = route_weights[source_ids].clamp_min(1e-8)
        route_loss = F.kl_div(
            selected_route.log(), information_route, reduction="batchmean"
        )

        auxiliary = {
            "private_loss": p1_reconstruction.mean()
            + p2_reconstruction.mean()
            + club21.mean().clamp_min(0.0)
            + club12.mean().clamp_min(0.0),
            "common_loss": (c1_cross_reconstruction + c2_cross_reconstruction).mean(),
            "align_loss": align.mean(),
            "route_loss": route_loss,
            "club_fit_loss": club21_fit + club12_fit,
            "rate_p1": rate_p1.mean(),
            "rate_p2": rate_p2.mean(),
            "rate_c1": rate_c1.mean(),
            "rate_c2": rate_c2.mean(),
        }
        return fused, auxiliary


class HDHKT(nn.Module):
    """HDHKT 主模型。

    层次化图知识追踪模型，融合异构图和超图进行预测。

    Args:
        data_metadata: 数据集元数据
        hetero_metadata: 异构图元数据
        hidden_dim: 隐藏层维度
        n_hop: GNN 层数
        heads: 注意力头数（架构常量，默认 1）
        lstm_layers: LSTM 层数（架构常量，默认 1）
        dropout: Dropout 概率（所有层共享）
        history_neighbour: 历史邻居数量（架构常量，默认 5）
        att_bound: 注意力边界（架构常量，默认 0.1）
        num_hyperedges: 难度加权超图中的超边数量
        use_information_bottleneck: 是否启用跨通道信息瓶颈
        **kwargs: 额外的关键字参数

    Example:
        >>> model = HDHKT(data_metadata, hetero_metadata, hidden_dim=250, n_hop=4,
        ...               dropout=0.25, num_hyperedges=100)
        >>> logits = model(user_sequence, user_response, user_mask, hetero_graph, hypergraph, question_skill_matrix)
    """

    def __init__(
        self,
        data_metadata: dict[str, Any],
        hetero_metadata: tuple[list[str], list[tuple[str, str, str]]],
        *,
        hidden_dim: int,
        n_hop: int,
        heads: int = 1,
        lstm_layers: int = 1,
        dropout: float,
        history_neighbour: int = 5,
        att_bound: float = 0.1,
        num_hyperedges: int,
        use_information_bottleneck: bool = True,
        ib_negative_samples: int = 8,
        ib_route_temperature: float = 0.5,
        ib_max_questions: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_metadata = data_metadata

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.use_information_bottleneck = use_information_bottleneck
        if ib_max_questions <= 0:
            raise ValueError("ib_max_questions must be positive")
        self.ib_max_questions = ib_max_questions

        self.question_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.question_embedding_hyper = torch.nn.Embedding(
            num_embeddings=data_metadata["num_questions"],
            embedding_dim=self.hidden_dim,
        )
        self.skill_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_skills"],
            embedding_dim=self.hidden_dim,
        )
        self.assignment_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_assignments"],
            embedding_dim=self.hidden_dim,
        )
        self.template_embedding = torch.nn.Embedding(
            num_embeddings=data_metadata["num_templates"],
            embedding_dim=self.hidden_dim,
        )
        self.answer_embedding = torch.nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.hidden_dim,
        )
        self.embedding_dropout = torch.nn.Dropout(p=self.dropout)

        self.hetero_conv = HeteroGNN(
            embedding_dim=self.hidden_dim,
            n_hop=n_hop,
            heads=heads,
            dropout=self.dropout,
            metadata=hetero_metadata,
        )

        self.hgnn_conv = HyperGNN(
            in_ch=self.hidden_dim,
            n_hid=self.hidden_dim,
            n_class=self.hidden_dim,
            dropout=self.dropout,
        )

        if self.use_information_bottleneck:
            self.fuse = InformationBottleneckMoE(
                dim=self.hidden_dim,
                num_skills=data_metadata["num_skills"],
                num_hyperedges=num_hyperedges,
                dropout=self.dropout,
                negative_samples=ib_negative_samples,
                route_temperature=ib_route_temperature,
            )
        else:
            self.fuse = MoEFusion(dim=self.hidden_dim, dropout=self.dropout)

        self.fc_exercise = Linear(
            self.hidden_dim * 2, self.hidden_dim, weight_initializer="uniform"
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        self.history_review = HistoryRecap(
            hist_neighbor_num=history_neighbour,
            att_bound=att_bound,
        )

        self.general_interaction = GeneralInteraction(hidden_dim=self.hidden_dim)

        # Eval-mode graph-backbone cache.
        self._graph_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    def train(self, mode: bool = True) -> "HDHKT":
        """Enter train/eval mode, invalidating the eval GNN cache on switch."""
        prev = self.training
        out = super().train(mode)
        if mode != prev:
            self._graph_cache.clear()
        return out

    def _compute_graph_outputs(
        self,
        hetero_graph: Any,
        hypergraph: Hypergraph,
        skill_ids_per_question: torch.Tensor | None = None,
        ib_question_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run the graph backbone.

        Returns:
            Fused questions, heterogeneous-graph skills, and training-only
            information-bottleneck auxiliary losses.
        """
        question_hyper_conv: torch.Tensor = self.hgnn_conv(
            self.question_embedding_hyper.weight, hypergraph
        )
        conv = self.hetero_conv(
            {
                "question": self.question_embedding.weight,
                "skill": self.skill_embedding.weight,
                "assignment": self.assignment_embedding.weight,
                "template": self.template_embedding.weight,
            },
            hetero_graph.edge_index_dict,
        )
        question_hetero_conv: torch.Tensor = conv["question"]
        skill_hetero_conv: torch.Tensor = conv["skill"]
        auxiliary: dict[str, torch.Tensor] = {}
        if self.use_information_bottleneck:
            question_skill_edges = None
            if skill_ids_per_question is not None:
                num_skills = self.data_metadata["num_skills"]
                valid = skill_ids_per_question != num_skills
                question_ids = torch.arange(
                    skill_ids_per_question.size(0),
                    device=skill_ids_per_question.device,
                ).unsqueeze(1)
                question_ids = question_ids.expand_as(skill_ids_per_question)
                question_skill_edges = torch.stack(
                    [question_ids[valid], skill_ids_per_question[valid]], dim=0
                )
            question_conv_fused, auxiliary = self.fuse(
                question_hetero_conv,
                question_hyper_conv,
                question_skill_edges=question_skill_edges,
                question_hyperedge_edges=hypergraph.hyperedge_index,
                source_ids=ib_question_ids,
            )
        else:
            question_conv_fused = self.fuse(question_hetero_conv, question_hyper_conv)
        return question_conv_fused, skill_hetero_conv, auxiliary

    def _cached_graph_outputs(
        self,
        hetero_graph: Any,
        hypergraph: Hypergraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached eval-mode graph outputs, computing once per graph pair."""
        key = (id(hetero_graph), id(hypergraph))
        cached = self._graph_cache.get(key)
        if cached is None:
            with torch.no_grad():
                question_conv_fused, skill_hetero_conv, _ = self._compute_graph_outputs(
                    hetero_graph, hypergraph
                )
            cached = {
                "question_conv_fused": question_conv_fused,
                "skill_hetero_conv": skill_hetero_conv,
            }
            self._graph_cache[key] = cached
        return cached["question_conv_fused"], cached["skill_hetero_conv"]

    def forward(
        self,
        user_sequence: torch.Tensor,  # [B, S]
        user_response: torch.Tensor,  # [B, S]
        user_mask: torch.Tensor,  # [B, S]
        hetero_graph: Any,  # HeteroData
        hypergraph: Hypergraph,
        skill_ids_per_question: torch.Tensor,  # [Q, K_max] (padding_index = num_skills)
        return_states: bool = False,
        return_auxiliary: bool = False,
    ) -> torch.Tensor:  # [B, S]
        """前向传播。

        Args:
            user_sequence: 用户问题序列 [B, S]
            user_response: 用户回答序列 [B, S]
            user_mask: 有效位置掩码 [B, S]
            hetero_graph: 异构图数据
            hypergraph: 超图数据
            skill_ids_per_question: 预计算的每题关联技能 id 表 [Q, K_max]，
                未用槽位填 ``num_skills``（前向时追加零向量行）
            return_states: 是否返回内部状态（用于知识状态计算）
            return_auxiliary: 是否返回训练期信息瓶颈辅助损失

        Returns:
            预测 logits [B, S]，或 (logits, skill_hetero_conv, lstm_output) 当 return_states=True
        """
        B, _ = user_sequence.size()

        # [B, S, embedding_dim]
        answers_embedding: torch.Tensor = self.answer_embedding(user_response)

        auxiliary: dict[str, torch.Tensor] = {}
        if self.training:
            ib_question_ids = torch.unique(user_sequence[user_mask], sorted=True)
            if ib_question_ids.numel() > self.ib_max_questions:
                permutation = torch.randperm(
                    ib_question_ids.numel(), device=ib_question_ids.device
                )[: self.ib_max_questions]
                ib_question_ids = ib_question_ids[permutation].sort().values
            question_conv_fused, skill_hetero_conv, auxiliary = (
                self._compute_graph_outputs(
                    hetero_graph,
                    hypergraph,
                    skill_ids_per_question=skill_ids_per_question,
                    ib_question_ids=ib_question_ids,
                )
            )
        else:
            question_conv_fused, skill_hetero_conv = self._cached_graph_outputs(
                hetero_graph, hypergraph
            )  # [num_questions, H] / [num_skills, H]

        question_embedding_sequence = question_conv_fused[user_sequence]

        exercise_emb = torch.cat(
            [question_embedding_sequence, answers_embedding], dim=-1
        )  # [B, S, 2*E]

        exercise_emb = F.relu(self.fc_exercise(exercise_emb))  # [B, S, H]
        exercise_emb = self.embedding_dropout(exercise_emb)

        # [B, S, H]
        lstm_output, _ = self.lstm(exercise_emb)

        # Shift to next-question sequence; last timestep is zero-padded
        next_user_sequence = torch.cat(
            [user_sequence[:, 1:], user_sequence.new_zeros(B, 1)], dim=1
        )  # [B, S]

        # [B, S, embedding_dim]
        next_question_embedding: torch.Tensor = question_conv_fused[next_user_sequence]

        history_question_neighbors = self.history_review(
            question_embedding_sequence,
            next_question_embedding,
            exercise_emb,
            user_mask,
        )  # [B, S, M, H]

        # Student status = LSTM output + history neighbors
        student_status = torch.cat(
            [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
        )  # [B, S, M+1, H]

        # Knowledge status = next-question features + related skill features
        num_skills = skill_hetero_conv.size(0)
        k_max = skill_ids_per_question.size(-1)
        skill_ids_full = skill_ids_per_question[next_user_sequence]  # [B, S, K_max]
        skill_counts = (skill_ids_full != num_skills).sum(dim=-1)  # [B, S]
        batch_k = skill_counts.max()  # 0-d tensor, kept on-device
        k_slot_mask = (
            torch.arange(k_max, device=skill_ids_full.device) < batch_k
        )  # [K_max]

        # Append a zero-vector row so the padding index resolves to zeros
        skill_conv_padded = F.pad(skill_hetero_conv, (0, 0, 0, 1))  # [num_skills+1, H]

        # [B, S, K_max, embedding_dim]
        related_skill_embs = skill_conv_padded[skill_ids_full]

        knowledge_status = torch.cat(
            [next_question_embedding.unsqueeze(2), related_skill_embs],
            dim=2,
        )  # [B, S, K_max+1, embedding_dim]

        logits = self.general_interaction(
            student_status, knowledge_status, user_mask, skill_slot_mask=k_slot_mask
        )  # [B, S]

        if return_states and return_auxiliary:
            return logits, skill_hetero_conv, lstm_output, auxiliary
        if return_states:
            return logits, skill_hetero_conv, lstm_output
        if return_auxiliary:
            return logits, auxiliary
        return logits  # [B, S]
