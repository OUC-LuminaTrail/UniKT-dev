"""DyGKT model trainer."""

from typing import Literal

import torch
import torch.nn.functional as F

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["DyGKTTrainer", "DyGKTConfig"]


@register_model_config("DyGKT")
class DyGKTConfig(ModelConfig):
    """DyGKT model configuration.

    Args:
        edge_dim: Edge feature dimension.
        node_dim: Node embedding dimension.
        dim_time: Time encoding dimension.
        ablation: Ablation mode from original DyGKT (-1, counter, dual, q_qid, q_kid, embed, skill, time).
        num_neighbor: Number of neighbors for history.
        neighbor_sampling_strategy: Neighbor sampling strategy in DyGKT data layer: recent truncation or time-decay weighted sampling.
        time_decay_factor: Time decay factor for time_decay neighbor sampling (weight=exp(-factor*delta_t)).
        neighbor_candidate_pool: Candidate pool size before sampling neighbors; <=0 means full history.
        neighbor_sampling_seed: Random seed for time-decay neighbor sampling.
        graph_neg_sampling: Enable graph-style in-batch negative sampling auxiliary loss.
        graph_neg_num_samples: Number of in-batch negative samples per interaction for auxiliary contrastive loss.
        graph_neg_temperature: Temperature for graph negative sampling contrastive logits.
        graph_neg_loss_weight: Weight of graph negative sampling auxiliary loss.
        dropout: Dropout rate.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    edge_dim: int = 64
    node_dim: int = 64
    dim_time: int = 16
    ablation: str = "-1"
    num_neighbor: int = 50
    neighbor_sampling_strategy: Literal["recent", "time_decay"] = "time_decay"
    time_decay_factor: float = 1e-5
    neighbor_candidate_pool: int = 200
    neighbor_sampling_seed: int = 2020
    graph_neg_sampling: bool = True
    graph_neg_num_samples: int = 2
    graph_neg_temperature: float = 0.2
    graph_neg_loss_weight: float = 0.05
    dropout: float = 0.1
    epochs: int = 100
    learning_rate: float = 5e-4
    lr_decay: float | None = None
    weight_decay: float = 1e-4
    batch_size: int = 2000


@register_trainer("DyGKT")
class DyGKTTrainer(BaseTrainer):
    """DyGKT 模型训练器"""

    def build_components(self, rc, data_src):
        from model.DyGKT.DyGKT_data import DyGKTModelData
        from model.DyGKT.DyGKT_model import DyGKT

        m = rc.model
        self.graph_neg_sampling = bool(m.graph_neg_sampling)
        self.graph_neg_num_samples = max(1, int(m.graph_neg_num_samples))
        self.graph_neg_temperature = max(1e-6, float(m.graph_neg_temperature))
        self.graph_neg_loss_weight = max(0.0, float(m.graph_neg_loss_weight))

        model_data = DyGKTModelData(data_src)
        train_dataset, val_dataset, test_dataset, model_metadata = (
            model_data.prepare_data(rc)
        )

        logger.info("Initializing DyGKT model...")
        model = DyGKT(
            model_metadata,
            num_neighbor=m.num_neighbor,
            ablation=m.ablation,
            dim_time=m.dim_time,
            edge_dim=m.edge_dim,
            node_dim=m.node_dim,
            dropout=m.dropout,
        )

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            max_clip_grad_norm=10.0,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            collate_fn=train_dataset.get_batch,
            val_collate_fn=val_dataset.get_batch,
            test_collate_fn=test_dataset.get_batch,
        )

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """Compute total loss = BCE + optional graph negative-sampling auxiliary loss."""
        base_loss = super()._compute_loss(outputs)
        if not self.graph_neg_sampling or self.graph_neg_loss_weight <= 0.0:
            return base_loss

        neg_loss = self._compute_graph_negative_loss(outputs)
        return base_loss + self.graph_neg_loss_weight * neg_loss

    def _compute_graph_negative_loss(
        self, outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """In-batch graph negative sampling via contrastive objective on node embeddings."""
        src_embeddings = outputs.get("src_embeddings")
        dst_embeddings = outputs.get("dst_embeddings")
        dst_node_ids = outputs.get("dst_node_ids")

        if src_embeddings is None or dst_embeddings is None or dst_node_ids is None:
            return outputs["y_hat"].new_zeros(())

        batch_size = src_embeddings.shape[0]
        if batch_size < 2:
            return outputs["y_hat"].new_zeros(())

        pos_logits = (src_embeddings * dst_embeddings).sum(
            dim=-1
        ) / self.graph_neg_temperature
        neg_logits_list: list[torch.Tensor] = []

        base_index = torch.arange(batch_size, device=src_embeddings.device)
        for _ in range(self.graph_neg_num_samples):
            perm = torch.randperm(batch_size, device=src_embeddings.device)
            if torch.all(perm == base_index):
                perm = torch.roll(perm, shifts=1)

            neg_dst_embeddings = dst_embeddings[perm]
            neg_dst_ids = dst_node_ids[perm]
            neg_logits = (src_embeddings * neg_dst_embeddings).sum(
                dim=-1
            ) / self.graph_neg_temperature
            same_target_mask = neg_dst_ids == dst_node_ids
            neg_logits = neg_logits.masked_fill(same_target_mask, -1e9)
            neg_logits_list.append(neg_logits)

        logits = torch.stack([pos_logits] + neg_logits_list, dim=1)
        valid_rows = torch.isfinite(logits[:, 1:]).any(dim=1)
        if not bool(valid_rows.any()):
            return outputs["y_hat"].new_zeros(())

        labels = torch.zeros(
            int(valid_rows.sum().item()), dtype=torch.long, device=logits.device
        )
        return F.cross_entropy(logits[valid_rows], labels)

    def forward_pass(self, batch_data: dict) -> dict[str, torch.Tensor]:
        """DyGKT 前向传播。

        Args:
            batch_data: 字典，包含所有交互信息和历史邻居

        Returns:
            包含 y_hat, y_label, y_predict 等的字典
        """
        batch = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch[key] = self._move_tensor_to_device(value)
            else:
                batch[key] = value

        src_embeddings, dst_embeddings = (
            self.model.compute_src_dst_node_temporal_embeddings(batch)
        )
        src_embeddings = self.model.dropout_layer(src_embeddings)
        dst_embeddings = self.model.dropout_layer(dst_embeddings)

        y_hat = (
            self.model.link_predictor(src_embeddings, dst_embeddings)
            .squeeze(dim=-1)
            .float()
        )  # [B]

        y_label = batch["correctness"].float()

        y_prob = torch.sigmoid(y_hat)
        y_predict = self._generate_binary_predictions(y_prob, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_prob,
            "y_prob": y_prob,
            "src_embeddings": src_embeddings,
            "dst_embeddings": dst_embeddings,
            "dst_node_ids": batch["question"].long(),
        }
