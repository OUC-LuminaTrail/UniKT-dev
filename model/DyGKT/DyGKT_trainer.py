"""DyGKT model trainer."""

from typing import Any

import torch
import torch.nn.functional as F

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["DyGKTTrainer", "DyGKTModelParams"]


@register_model_params("DyGKT")
class DyGKTModelParams(BaseParamConfig):
    """DyGKT model parameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "DyGKT Parameters"
        params = {
            "edge_dim": {
                "type": int,
                "default": 64,
                "help": "Edge feature dimension (default: 64)",
            },
            "node_dim": {
                "type": int,
                "default": 64,
                "short": "nd",
                "help": "Node embedding dimension (default: 64)",
            },
            "dim_time": {
                "type": int,
                "default": 16,
                "short": "dt",
                "help": "Time encoding dimension (default: 16)",
            },
            "ablation": {
                "type": str,
                "default": "-1",
                "help": "Ablation mode from original DyGKT (-1, counter, dual, q_qid, q_kid, embed, skill, time)",
            },
            "num_neighbor": {
                "type": int,
                "default": 50,
                "short": "nn",
                "help": "Number of neighbors for history (default: 50)",
            },
            "neighbor_sampling_strategy": {
                "type": str,
                "default": "time_decay",
                "choices": ["recent", "time_decay"],
                "help": "Neighbor sampling strategy in DyGKT data layer: recent truncation or time-decay weighted sampling (default: time_decay).",
            },
            "time_decay_factor": {
                "type": float,
                "default": 1e-5,
                "help": "Time decay factor for time_decay neighbor sampling (weight=exp(-factor*delta_t), default: 1e-5).",
            },
            "neighbor_candidate_pool": {
                "type": int,
                "default": 200,
                "help": "Candidate pool size before sampling neighbors; <=0 means full history (default: 200).",
            },
            "neighbor_sampling_seed": {
                "type": int,
                "default": 2020,
                "help": "Random seed for time-decay neighbor sampling (default: 2020).",
            },
            "graph_neg_sampling": {
                "type": bool,
                "default": True,
                "help": "Enable graph-style in-batch negative sampling auxiliary loss (default: True).",
            },
            "graph_neg_num_samples": {
                "type": int,
                "default": 2,
                "help": "Number of in-batch negative samples per interaction for auxiliary contrastive loss (default: 2).",
            },
            "graph_neg_temperature": {
                "type": float,
                "default": 0.2,
                "help": "Temperature for graph negative sampling contrastive logits (default: 0.2).",
            },
            "graph_neg_loss_weight": {
                "type": float,
                "default": 0.05,
                "help": "Weight of graph negative sampling auxiliary loss (default: 0.05).",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "short": "dp",
                "help": "Dropout rate (default: 0.1)",
            },
            "epochs": {
                "type": int,
                "default": 100,
                "short": "ep",
                "help": "Number of training epochs (default: 100)",
            },
            "learning_rate": {
                "type": float,
                "default": 0.0005,
                "short": "lr",
                "help": "Learning rate for optimizer (default: 0.0005)",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch (default: None)",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-4,
                "short": "wd",
                "help": "Weight decay (L2 regularization) for optimizer (default: 1e-4)",
            },
            "batch_size": {
                "type": int,
                "default": 2000,
                "short": "bs",
                "help": "Batch size for training (default: 2000)",
            },
        }

        return group_name, params


@register_trainer("DyGKT")
class DyGKTTrainer(BaseTrainer):
    """DyGKT 模型训练器"""

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        self.graph_neg_sampling = bool(getattr(args, "graph_neg_sampling", True))
        self.graph_neg_num_samples = max(
            1, int(getattr(args, "graph_neg_num_samples", 2))
        )
        self.graph_neg_temperature = max(
            1e-6, float(getattr(args, "graph_neg_temperature", 0.2))
        )
        self.graph_neg_loss_weight = max(
            0.0, float(getattr(args, "graph_neg_loss_weight", 0.05))
        )

        # 1. 准备数据
        from model.DyGKT.DyGKT_data import DyGKTModelData

        model_data = DyGKTModelData(data_src)
        train_dataset, val_dataset, test_dataset, model_metadata = (
            model_data.prepare_data(args)
        )

        # 2. 初始化模型
        from model.DyGKT.DyGKT_model import DyGKT

        logger.info("Initializing DyGKT model...")
        model = DyGKT(args, model_metadata)

        # 3. 创建优化器和损失函数
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # 4. 创建学习率调度器
        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        # 5. 初始化基类训练器
        super().__init__(model)

        # 6. 构建早停配置
        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 0.0),
            )

        # 7. 配置训练器
        self.with_training(
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
        ).with_data(
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            batch_size=args.batch_size,
            collate_fn=train_dataset.get_batch,
            val_collate_fn=val_dataset.get_batch,
            test_collate_fn=test_dataset.get_batch,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_clip_grad_norm=10.0,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="DyGKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

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
        # 移动所有张量到设备
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

        # 模型前向传播，返回 logits
        y_hat = (
            self.model.link_predictor(src_embeddings, dst_embeddings)
            .squeeze(dim=-1)
            .float()
        )  # [B]

        # 标签是 correctness
        y_label = batch["correctness"].float()

        # 生成概率和二分类预测
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
