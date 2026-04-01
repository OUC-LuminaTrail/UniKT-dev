"""DYGKT model trainer."""

import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.config import (
    BaseParamConfig,
    DataLoaderConfig,
    EarlyStoppingConfig,
    create_optimized_dataloader,
    register_model_params,
)
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["DYGKTTrainer", "DYGKTModelParams"]


class _IndexDataset(Dataset):
    """Dataset that only returns sample indices for vectorized collate."""

    def __init__(self, size: int) -> None:
        self._size = int(size)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> int:
        return int(index)


@register_model_params("DYGKT")
class DYGKTModelParams(BaseParamConfig):
    """DYGKT model parameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "DYGKT Parameters"
        params = {
            "hidden_dim": {
                "type": int,
                "default": 128,
                "short": "hd",
                "help": "Hidden layer dimension (default: 128)",
            },
            "num_predict_layer": {
                "type": int,
                "default": 2,
                "help": "Number of predictor layers (default: 2)",
            },
            "activate_type": {
                "type": str,
                "default": "relu",
                "help": "Activation type for predictor (default: relu)",
            },
            "embedding_dim": {
                "type": int,
                "default": 128,
                "short": "ed",
                "help": "Embedding dimension (default: 128)",
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
                "help": "Neighbor sampling strategy in DYGKT data layer: recent truncation or time-decay weighted sampling (default: time_decay).",
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
            "dygkt_split_protocol": {
                "type": str,
                "default": "kfold",
                "choices": ["kfold", "time_quantile"],
                "help": "Data split protocol for DYGKT: kfold (project default) or time_quantile (original DyGKT-like).",
            },
            "dygkt_val_ratio": {
                "type": float,
                "default": 0.1,
                "help": "Validation ratio when dygkt_split_protocol=time_quantile (default: 0.1).",
            },
            "dygkt_test_ratio": {
                "type": float,
                "default": 0.1,
                "help": "Test ratio when dygkt_split_protocol=time_quantile (default: 0.1).",
            },
            "max_similarity_matrix_questions": {
                "type": int,
                "default": 12000,
                "help": "Max question count to build full question-question similarity matrix; larger datasets use local on-the-fly similarity (default: 12000)",
            },
            "compat_fields": {
                "type": bool,
                "default": False,
                "help": "Whether to generate legacy compatibility fields in DYGKT dataset (default: False, faster)",
            },
            "no_cache": {
                "type": bool,
                "default": False,
                "help": "Disable DYGKT dataset cache and force rebuilding preprocessing artifacts",
            },
            "cache_dir": {
                "type": str,
                "default": None,
                "help": "Directory for DYGKT dataset cache (default: ./cache/dygkt)",
            },
            "cache_version": {
                "type": int,
                "default": 2,
                "help": "Manual cache version to invalidate stale DYGKT cache entries (default: 2)",
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
            "profile_batches": {
                "type": int,
                "default": 0,
                "help": "Profile first N train batches and log data-wait vs compute time (0 disables)",
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
            "max_grad_norm": {
                "type": float,
                "default": 10.0,
                "help": "Maximum norm for gradient clipping (default: 10.0)",
            },
            "batch_size": {
                "type": int,
                "default": 2000,
                "short": "bs",
                "help": "Batch size for training (default: 2000)",
            },
            "loader_num_workers": {
                "type": int,
                "default": -1,
                "help": "DataLoader worker count (-1 means auto)",
            },
            "loader_prefetch_factor": {
                "type": int,
                "default": 2,
                "help": "DataLoader prefetch factor when num_workers > 0 (default: 2)",
            },
            "loader_persistent_workers": {
                "type": bool,
                "default": True,
                "help": "Enable persistent DataLoader workers when num_workers > 0",
            },
            "eval_batch_size": {
                "type": int,
                "default": 0,
                "help": "Validation/test batch size (0 means auto=2*train batch size)",
            },
            "eval_loader_num_workers": {
                "type": int,
                "default": -1,
                "help": "Validation/test DataLoader worker count (-1 means use loader_num_workers)",
            },
        }

        return group_name, params


@TRAINERS.register("DYGKT")
class DYGKTTrainer(BaseTrainer):
    """DYGKT 模型训练器"""

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        # 自动检测 GPU
        if not hasattr(args, "device") or args.device is None or args.device == "auto":
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {args.device}")

        self.profile_batches = max(0, int(getattr(args, "profile_batches", 0)))
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
        self._profile_batch_count = 0
        self._profile_last_batch_end: float | None = None
        self._profile_sums: dict[str, float] = {
            "wait_data": 0.0,
            "zero_grad": 0.0,
            "forward": 0.0,
            "loss": 0.0,
            "backward": 0.0,
            "clip": 0.0,
            "step": 0.0,
            "total_compute": 0.0,
        }
        self._profile_logged = False
        if self.profile_batches > 0:
            logger.info(
                "DYGKT profiling enabled for first %s train batches",
                self.profile_batches,
            )

        # 1. 准备数据
        from model.DYGKT import DYGKTModelData

        model_data = DYGKTModelData(data_src)
        train_dataset, val_dataset, test_dataset, model_metadata = (
            model_data.prepare_data(args)
        )

        # 2. 初始化模型
        from model.DYGKT.DYGKT_model import DYGKT

        logger.info("Initializing DYGKT model...")
        model = DYGKT(args, model_metadata)

        # Keep train batches chronological to match the original DyGKT setup.
        loader_device = (
            args.device
            if isinstance(args.device, torch.device)
            else torch.device(args.device)
        )
        loader_num_workers_arg = int(getattr(args, "loader_num_workers", -1))
        loader_num_workers: int | str = (
            "auto" if loader_num_workers_arg < 0 else loader_num_workers_arg
        )
        loader_prefetch_factor = max(1, int(getattr(args, "loader_prefetch_factor", 2)))
        loader_persistent_workers = bool(
            getattr(args, "loader_persistent_workers", True)
        )
        eval_batch_size_arg = int(getattr(args, "eval_batch_size", 0))
        eval_batch_size = (
            eval_batch_size_arg
            if eval_batch_size_arg > 0
            else max(1, int(args.batch_size) * 2)
        )
        eval_loader_num_workers_arg = int(getattr(args, "eval_loader_num_workers", -1))
        if eval_loader_num_workers_arg < 0:
            eval_loader_num_workers = loader_num_workers
        else:
            eval_loader_num_workers = eval_loader_num_workers_arg

        loader_config = DataLoaderConfig(
            num_workers=loader_num_workers,
            pin_memory=True,
            prefetch_factor=loader_prefetch_factor,
            persistent_workers=loader_persistent_workers,
        )
        eval_loader_config = DataLoaderConfig(
            num_workers=eval_loader_num_workers,
            pin_memory=True,
            prefetch_factor=loader_prefetch_factor,
            persistent_workers=loader_persistent_workers,
        )

        train_index_dataset = _IndexDataset(len(train_dataset))
        val_index_dataset = _IndexDataset(len(val_dataset))
        test_index_dataset = _IndexDataset(len(test_dataset))

        train_loader = create_optimized_dataloader(
            train_index_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            device=loader_device,
            config=loader_config,
            collate_fn=train_dataset.collate_indices,
        )

        val_loader = create_optimized_dataloader(
            val_index_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            device=loader_device,
            config=eval_loader_config,
            collate_fn=val_dataset.collate_indices,
        )

        test_loader = create_optimized_dataloader(
            test_index_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            device=loader_device,
            config=eval_loader_config,
            collate_fn=test_dataset.collate_indices,
        )

        # 3. 创建优化器和损失函数
        # DYGKT 模型现在返回 logits，对应 BCEWithLogitsLoss。
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # 保存 max_grad_norm 以备后用
        self.max_grad_norm = getattr(args, "max_grad_norm", 10.0)

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
            train_data=train_loader,
            val_data=val_loader,
            test_data=test_loader,
            batch_size=args.batch_size,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="DYGKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def _run_train_batch(self, batch_data: tuple[Any, ...]) -> float:
        """Train one batch with gradient clipping."""
        profile_this_batch = self.profile_batches > 0 and not self._profile_logged
        batch_start = time.perf_counter()

        if profile_this_batch and self._profile_last_batch_end is not None:
            self._profile_sums["wait_data"] += (
                batch_start - self._profile_last_batch_end
            )

        t0 = batch_start
        self.opt.zero_grad(set_to_none=True)
        t1 = time.perf_counter()
        output = self.forward_pass(batch_data)
        t2 = time.perf_counter()
        loss = self._compute_loss(output)
        t3 = time.perf_counter()
        loss.backward()
        t4 = time.perf_counter()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.max_grad_norm
        )
        t5 = time.perf_counter()

        self.opt.step()
        t6 = time.perf_counter()

        # 累积预测
        self.metrics_accumulator.update("train", output)

        if profile_this_batch:
            self._profile_batch_count += 1
            self._profile_sums["zero_grad"] += t1 - t0
            self._profile_sums["forward"] += t2 - t1
            self._profile_sums["loss"] += t3 - t2
            self._profile_sums["backward"] += t4 - t3
            self._profile_sums["clip"] += t5 - t4
            self._profile_sums["step"] += t6 - t5
            self._profile_sums["total_compute"] += t6 - t0
            self._profile_last_batch_end = t6

            if self._profile_batch_count >= self.profile_batches:
                avg_wait = self._profile_sums["wait_data"] / max(
                    self.profile_batches - 1, 1
                )
                avg_compute = self._profile_sums["total_compute"] / self.profile_batches
                ratio = avg_wait / max(avg_compute, 1e-12)

                logger.info(
                    "DYGKT profile summary (%s batches): avg_wait_data=%.4fs, avg_compute=%.4fs, wait/compute=%.2f",
                    self.profile_batches,
                    avg_wait,
                    avg_compute,
                    ratio,
                )
                logger.info(
                    "DYGKT profile breakdown per batch (s): zero_grad=%.4f, forward=%.4f, loss=%.4f, backward=%.4f, clip=%.4f, step=%.4f",
                    self._profile_sums["zero_grad"] / self.profile_batches,
                    self._profile_sums["forward"] / self.profile_batches,
                    self._profile_sums["loss"] / self.profile_batches,
                    self._profile_sums["backward"] / self.profile_batches,
                    self._profile_sums["clip"] / self.profile_batches,
                    self._profile_sums["step"] / self.profile_batches,
                )

                if ratio > 0.6:
                    logger.info(
                        "Profile hint: data pipeline is likely the bottleneck (consider larger workers/prefetch/cache)."
                    )
                else:
                    logger.info(
                        "Profile hint: model compute is likely the bottleneck (consider AMP/torch.compile/larger batch)."
                    )
                self._profile_logged = True
        else:
            self._profile_last_batch_end = t6

        return loss.item()

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

    def _move_tensor_to_device(
        self, tensor: torch.Tensor, dtype: torch.dtype = None
    ) -> torch.Tensor:
        """Train epoch with non-blocking GPU transfer."""
        non_blocking = (
            self.device_ is not None
            and self.device_.type == "cuda"
            and tensor.device.type == "cpu"
        )
        result = tensor.to(self.device_, non_blocking=non_blocking)
        if dtype is not None:
            result = result.to(dtype)
        return result

    @torch.inference_mode()
    def _run_eval_batch(self, batch_data: tuple[Any, ...]) -> float:
        """Validate with inference mode."""
        output = self.forward_pass(batch_data)
        loss = self._compute_loss(output)
        self.metrics_accumulator.update("val", output)
        return loss.item()

    @torch.inference_mode()
    def _run_test_batch(self, batch_data: tuple[Any, ...]) -> float:
        """Test with inference mode."""
        output = self.test_forward_pass(batch_data)
        loss = self._compute_loss(output)
        self.metrics_accumulator.update("test", output)
        return loss.item()

    def forward_pass(self, batch_data: dict) -> dict[str, torch.Tensor]:
        """DYGKT 前向传播（接受 batch 字典）。

        Args:
            batch_data: 字典，包含所有交互信息和历史邻居

        Returns:
            包含 y_hat, y_label, y_predict 等的字典
        """
        # batch_data 已经是字典格式（由 DYGKTDataset.__getitem__ 返回）
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
