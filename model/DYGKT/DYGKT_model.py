"""DYGKT model migrated from the original DyGKT implementation.

This version keeps the original graph-style embedding update logic while adapting
inputs to kt-exp-graph batch dictionaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from utils.core import register_model


class TimeEncoder(nn.Module):
    """Original cosine time encoder."""

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)).reshape(time_dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        timestamps = timestamps.unsqueeze(dim=2)
        return torch.cos(self.w(timestamps))


class TimeDualDecayEncoder(nn.Module):
    """Original dual-decay time encoder."""

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True) -> None:
        super().__init__()
        self.time_dim = time_dim

        self.w_short = nn.Linear(1, time_dim)
        self.w_long = nn.Linear(1, time_dim)
        self.w_short.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)).reshape(time_dim, -1)
        )
        self.w_short.bias = nn.Parameter(torch.zeros(time_dim))
        self.w_long.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)).reshape(time_dim, -1)
        )
        self.w_long.bias = nn.Parameter(torch.zeros(time_dim))

        self.w_o = nn.Linear(time_dim, time_dim)
        self.w_o.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim * time_dim, dtype=np.float32)).reshape(
                time_dim, -1
            )
        )
        self.w_o.bias = nn.Parameter(torch.zeros(time_dim))

        self.f = nn.ReLU()

        if not parameter_requires_grad:
            self.w_short.weight.requires_grad = False
            self.w_short.bias.requires_grad = False
            self.w_long.weight.requires_grad = False
            self.w_long.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        timestamps = timestamps.unsqueeze(dim=2)

        timestamps_right = timestamps.clone()
        timestamps_right = torch.cat([timestamps_right[:, 1:, :], timestamps_right[:, -1, :].unsqueeze(1)], dim=1)
        timestamps_diff = timestamps_right - timestamps

        timestamps_mask = (timestamps_diff > 3600 * 24).float()

        timestamps_short = self.f(self.w_short(timestamps_diff * timestamps_mask))
        timestamps_long = self.f(self.w_long(timestamps_diff * (1 - timestamps_mask)))
        output = self.w_o(timestamps_short + timestamps_long)

        return output


class MergeLayer(nn.Module):
    """Original link predictor used in DyGKT training."""

    def __init__(self, input_dim1: int, input_dim2: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_1, input_2], dim=1)
        return self.fc2(self.act(self.fc1(x)))


class DyKTSeq(nn.Module):
    """GRU updater block from original DyGKT."""

    def __init__(self, edge_dim: int, node_dim: int) -> None:
        super().__init__()
        self.patch_enc_layer = nn.Linear(edge_dim, node_dim)
        self.hid_node_updater = nn.GRU(input_size=edge_dim, hidden_size=node_dim, batch_first=True)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.hid_node_updater(x)
        return torch.squeeze(hidden, dim=0)


@register_model("DYGKT")
class DYGKT(nn.Module):
    """DYGKT migrated model.

    The model consumes per-interaction neighborhood tensors produced by
    ``model/DYGKT/DYGKT_data.py`` and follows the original DyGKT embedding update
    equations.
    """

    def __init__(self, args: Any, data_metadata: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.args = args
        self.data_metadata = data_metadata

        self.num_neighbors = int(getattr(args, "num_neighbor", data_metadata.get("num_neighbor", 50)))
        self.ablation = str(getattr(args, "ablation", "-1"))
        self.time_dim = int(getattr(args, "dim_time", 16))

        self.edge_dim = 64
        self.node_dim = 64

        num_questions = int(data_metadata["num_questions"])
        num_users = int(data_metadata["num_users"])
        self.num_nodes = num_questions + num_users

        question_features = data_metadata.get("question_features")
        question_skill_ids = data_metadata.get("question_skill_ids")
        if question_skill_ids is None:
            question_skill_ids = np.zeros(num_questions, dtype=np.int64)
        question_skill_ids = np.asarray(question_skill_ids, dtype=np.int64)
        if len(question_skill_ids) != num_questions:
            raise ValueError(
                f"question_skill_ids length mismatch: expected {num_questions}, got {len(question_skill_ids)}"
            )

        if question_features is None:
            question_features = question_skill_ids.astype(np.float32)[:, np.newaxis]
        question_features = np.asarray(question_features, dtype=np.float32)
        if question_features.shape[0] != num_questions:
            raise ValueError(
                "question_features row mismatch: "
                f"expected {num_questions}, got {question_features.shape[0]}"
            )

        node_feature_dim = int(question_features.shape[1])
        node_raw_features = np.zeros((self.num_nodes, node_feature_dim), dtype=np.float32)
        node_raw_features[:num_questions, :] = question_features

        node_skill_ids = np.zeros(self.num_nodes, dtype=np.int64)
        node_skill_ids[:num_questions] = question_skill_ids

        self.num_skills = int(question_skill_ids.max()) + 1 if question_skill_ids.size > 0 else 1

        self.register_buffer("node_raw_features", torch.from_numpy(node_raw_features), persistent=False)
        self.register_buffer("node_skill_ids", torch.from_numpy(node_skill_ids), persistent=False)

        self.projection_layer = nn.ModuleDict(
            {
                "feature_Linear": nn.Linear(in_features=node_feature_dim, out_features=self.node_dim, bias=True),
                "feature_Embed": nn.Embedding(self.num_skills, self.node_dim),
                "node": nn.Embedding(self.num_nodes, self.node_dim),
                "edge": nn.Linear(in_features=1, out_features=self.node_dim, bias=True),
                "time": nn.Linear(in_features=self.time_dim, out_features=self.node_dim, bias=True),
                "struct": nn.Linear(in_features=1, out_features=self.node_dim, bias=True),
            }
        )

        self.output_layer = nn.Linear(in_features=self.node_dim, out_features=self.node_dim, bias=True)
        self.dropout_layer = nn.Dropout(float(getattr(args, "dropout", 0.3)))

        self.src_node_updater = DyKTSeq(edge_dim=self.edge_dim, node_dim=self.node_dim)
        self.dst_node_updater = DyKTSeq(edge_dim=self.edge_dim, node_dim=self.node_dim)

        if self.ablation == "dual":
            self.time_encoder = TimeEncoder(time_dim=self.time_dim)
        else:
            self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)

        self.link_predictor = MergeLayer(input_dim1=64, input_dim2=64, hidden_dim=64, output_dim=1)

    def set_neighbor_sampler(self, neighbor_sampler: Any) -> None:
        # Kept for compatibility with the original API. The migrated version
        # uses precomputed neighborhoods from the dataset.
        self.neighbor_sampler = neighbor_sampler

    def get_features(
        self,
        nodes_neighbor_ids: torch.Tensor,
        nodes_edge_features: torch.Tensor,
        nodes_neighbor_times: torch.Tensor,
        node_interact_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.ablation in ["embed", "q_kid"]:
            skill_ids = self.node_skill_ids[nodes_neighbor_ids].long()
            nodes_neighbor_node_raw_features = self.projection_layer["feature_Embed"](skill_ids)
        elif self.ablation == "q_qid":
            nodes_neighbor_node_raw_features = self.projection_layer["node"](nodes_neighbor_ids)
        else:
            raw = self.node_raw_features[nodes_neighbor_ids].float()
            nodes_neighbor_node_raw_features = self.projection_layer["feature_Linear"](raw)

        if self.ablation == "dual":
            rel_time = node_interact_times.unsqueeze(1) - nodes_neighbor_times
            nodes_neighbor_time_features = self.time_encoder(rel_time)
        else:
            nodes_neighbor_time_features = self.time_encoder(nodes_neighbor_times)

        nodes_neighbor_time_features = self.projection_layer["time"](nodes_neighbor_time_features)
        nodes_edge_raw_features = self.projection_layer["edge"](nodes_edge_features)

        if self.ablation == "time":
            nodes_neighbor_time_features = nodes_neighbor_time_features * 0
        elif self.ablation == "skill":
            nodes_neighbor_node_raw_features = nodes_neighbor_node_raw_features * 0

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features

    def compute_src_dst_node_temporal_embeddings(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        src_node_ids = batch["user"].long()
        dst_node_ids = batch["question"].long()
        node_interact_times = batch["time"].float()

        src_neighbor_node_ids = batch["src_neighbor_node_ids"].long()
        dst_neighbor_node_ids = batch["dst_neighbor_node_ids"].long()

        src_neighbor_times = batch["src_neighbor_times"].float()
        dst_neighbor_times = batch["dst_neighbor_times"].float()

        src_neighbor_edge_feats = batch["src_neighbor_edge_feats"].float().unsqueeze(-1)
        dst_neighbor_edge_feats = batch["dst_neighbor_edge_feats"].float().unsqueeze(-1)

        batch_size = src_node_ids.shape[0]
        device = src_node_ids.device

        # Append current node/time as the (num_neighbors + 1)-th element.
        src_neighbor_node_ids = torch.cat([src_neighbor_node_ids, src_node_ids.unsqueeze(1)], dim=1)
        dst_neighbor_node_ids = torch.cat([dst_neighbor_node_ids, dst_node_ids.unsqueeze(1)], dim=1)

        src_neighbor_times = torch.cat([src_neighbor_times, node_interact_times.unsqueeze(1)], dim=1)
        dst_neighbor_times = torch.cat([dst_neighbor_times, node_interact_times.unsqueeze(1)], dim=1)

        zero_edge = torch.zeros(batch_size, 1, 1, device=device)
        src_neighbor_edge_feats = torch.cat([src_neighbor_edge_feats, zero_edge], dim=1)
        dst_neighbor_edge_feats = torch.cat([dst_neighbor_edge_feats, zero_edge], dim=1)

        src_nodes_neighbor_co_occurrence_features = (
            batch["src_neighbor_node_ids"].long() == dst_node_ids.unsqueeze(1).repeat(1, self.num_neighbors)
        ).unsqueeze(-1).float()
        dst_nodes_neighbor_co_occurrence_features = (
            batch["dst_neighbor_node_ids"].long() == src_node_ids.unsqueeze(1).repeat(1, self.num_neighbors)
        ).unsqueeze(-1).float()

        src_node_skill = self.node_skill_ids[src_neighbor_node_ids][:, :-1].long()
        dst_node_skill = (
            self.node_skill_ids[dst_neighbor_node_ids][:, -1].long().unsqueeze(1).repeat(1, self.num_neighbors)
        )
        src_nodes_neighbor_skill_features = (src_node_skill == dst_node_skill).unsqueeze(-1).float()

        a = 0.0 if self.ablation == "counter" else 1.0

        src_nodes_neighbor_struct_features = self.projection_layer["struct"](a * src_nodes_neighbor_co_occurrence_features)
        dst_nodes_neighbor_struct_features = self.projection_layer["struct"](a * dst_nodes_neighbor_co_occurrence_features)
        src_nodes_neighbor_skill_struct_features = self.projection_layer["struct"](a * src_nodes_neighbor_skill_features)

        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = self.get_features(
            nodes_neighbor_ids=src_neighbor_node_ids,
            nodes_edge_features=src_neighbor_edge_feats,
            nodes_neighbor_times=src_neighbor_times,
            node_interact_times=node_interact_times,
        )
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = self.get_features(
            nodes_neighbor_ids=dst_neighbor_node_ids,
            nodes_edge_features=dst_neighbor_edge_feats,
            nodes_neighbor_times=dst_neighbor_times,
            node_interact_times=node_interact_times,
        )

        src_nodes_features = (
            src_nodes_neighbor_node_raw_features + src_nodes_edge_raw_features + src_nodes_neighbor_time_features
        )
        dst_nodes_features = (
            dst_nodes_neighbor_node_raw_features + dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features
        )

        src_node_embeddings = self.src_node_updater.update(
            src_nodes_features[:, :-1, :] + src_nodes_neighbor_skill_struct_features + src_nodes_neighbor_struct_features
        ) + (src_nodes_edge_raw_features + src_nodes_neighbor_time_features)[:, -1, :]

        if self.ablation in ["q_qid", "q_kid"]:
            dst_node_embeddings = dst_nodes_neighbor_node_raw_features[:, -1, :]
        else:
            dst_node_embeddings = self.dst_node_updater.update(
                (dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features)[:, :-1, :]
                + dst_nodes_neighbor_struct_features
            ) + dst_nodes_features[:, -1, :]

        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)

        return src_node_embeddings, dst_node_embeddings

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        src_node_embeddings, dst_node_embeddings = self.compute_src_dst_node_temporal_embeddings(batch)

        src_node_embeddings = self.dropout_layer(src_node_embeddings)
        dst_node_embeddings = self.dropout_layer(dst_node_embeddings)

        logits = self.link_predictor(src_node_embeddings, dst_node_embeddings).squeeze(dim=-1)
        return logits
