"""TCKT 模型训练器。

负责：
    1. 数据准备（委托 :class:`TCKTModelData`）与模型构建；
    2. ``forward_pass``：调用模型并按 same-position 约定提取有效预测；
    3. **在线全局字典生成**：通过回调在每个 epoch 开始时，对当前模型在训练集上产出的
       交互嵌入做 MiniBatchKMeans 聚类，得到 N 个中心，更新模型的全局字典缓冲区
       （论文 4.3 节“dictionary is updated during the end-to-end training”）。

原作者代码缺失训练基础设施（``KTM`` 类）与全局聚类生成；前者由 ``BaseTrainer`` 替代，
后者由此处的 :class:`GlobalDictRefreshCallback` + :meth:`refresh_global_dict` 补全。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from utils.config import (
    BaseParamConfig,
    EarlyStoppingConfig,
    register_model_params,
)
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer
from utils.training.callbacks import Callback

logger = get_logger(__name__)


@register_model_params("TCKT")
class TCKTModelParams(BaseParamConfig):
    """TCKT 模型超参数。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "TCKT Parameters"
        params = {
            "d_k": {"type": int, "default": 128, "help": "Hidden dimension d_k"},
            "d_a": {
                "type": int,
                "default": 64,
                "help": "Expansion dim for answer / difficulty (paper d_a)",
            },
            "d_e": {"type": int, "default": 128, "help": "Exercise embedding dim"},
            "num_heads": {"type": int, "default": 8, "help": "Attention heads"},
            "dropout": {"type": float, "default": 0.2, "help": "Dropout rate"},
            "q_gamma": {
                "type": float,
                "default": 0.03,
                "help": "Q-matrix smoothing factor",
            },
            "global_dict_size": {
                "type": int,
                "default": 400,
                "help": "Global dictionary size N (K-Means clusters)",
            },
            "max_rt_seconds": {
                "type": int,
                "default": 300,
                "help": "Response time cap in seconds (n_at = this + 1)",
            },
            "max_it_minutes": {
                "type": int,
                "default": 120,
                "help": "Interval time cap in minutes (n_it = this + 1)",
            },
            "cluster_sample_size": {
                "type": int,
                "default": 300000,
                "help": "Max interactions sampled for K-Means each epoch",
            },
            "learning_rate": {
                "type": float,
                "default": 0.002,
                "short": "lr",
                "help": "Learning rate",
            },
            "adam_beta1": {
                "type": float,
                "default": 0.1,
                "help": "Adam beta1 (reference code uses 0.1)",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-6,
                "short": "wd",
                "help": "Weight decay",
            },
            "max_grad_norm": {
                "type": float,
                "default": 1.0,
                "short": "mgn",
                "help": "Max gradient norm for clipping, 0 disables",
            },
            "lr_decay_step": {
                "type": int,
                "default": 10,
                "help": "StepLR step size",
            },
            "lr_decay_rate": {
                "type": float,
                "default": 0.5,
                "help": "StepLR gamma",
            },
            "epochs": {
                "type": int,
                "default": 100,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "batch_size": {
                "type": int,
                "default": 32,
                "short": "bs",
                "help": "Batch size",
            },
        }
        return group_name, params


class GlobalDictRefreshCallback(Callback):
    """每个 epoch 开始时刷新全局字典。

    - epoch 0：用随机初始化的模型产出初始字典（冷启动）。
    - 后续 epoch：用上一轮更新后的模型产出新字典。
    这样同一 epoch 的训练与验证使用同一份字典。
    """

    def __init__(self, trainer: TCKTTrainer):
        self.trainer = trainer

    def on_epoch_begin(self, epoch: int, **kwargs):
        self.trainer.refresh_global_dict()


@register_trainer("TCKT")
class TCKTTrainer(BaseTrainer):
    """TCKT 训练器。"""

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.TCKT.TCKT_data import TCKTModelData
        from model.TCKT.TCKT_model import TCKTNet

        # Force the math SDP backend for nn.MultiheadAttention. PyTorch >=2.x
        # defaults to flash / memory-efficient attention, whose backward kernel
        # returns NaN gradients once this model's attention becomes peaked (which
        # happens as weights grow on large datasets such as assistments12). The NaN
        # gradient silently corrupts the embedding weights, and the next forward
        # feeds NaN predictions into BCELoss -> device-side assert. The math backend
        # computes the same attention with a numerically robust backward.
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        # 数据
        model_data = TCKTModelData(data_src)
        train_dataset, val_dataset, test_dataset, info = model_data.prepare_data(args)

        # 模型
        model = TCKTNet(
            num_questions=info["num_questions"],
            num_skills=info["num_skills"],
            n_at=info["n_at"],
            n_it=info["n_it"],
            d_k=args.d_k,
            d_a=args.d_a,
            d_e=args.d_e,
            num_heads=args.num_heads,
            seq_len=info["max_seq_len"],
            global_dict_size=args.global_dict_size,
            dropout=args.dropout,
        )
        model.set_q_matrix(info["q_matrix"], args.q_gamma)
        model.set_difficulty(info["difficulty"])

        # 记录聚类相关配置，供回调使用。
        self.global_dict_size = args.global_dict_size
        self.cluster_sample_size = args.cluster_sample_size

        # 优化器 / 损失 / 调度器
        loss_fn = nn.BCELoss(reduction="none")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            eps=1e-8,
            betas=(args.adam_beta1, 0.999),
            weight_decay=args.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate
        )

        super().__init__(model)

        early_stopping_cfg = EarlyStoppingConfig(
            monitor=getattr(args, "es_monitor", "auc"),
            mode=getattr(args, "es_mode", "max"),
            patience=getattr(args, "es_patience", 10),
            min_delta=getattr(args, "es_min_delta", 0.0),
        )

        self.with_callbacks(callbacks=[GlobalDictRefreshCallback(self)]).with_training(
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
        ).with_data(
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            batch_size=args.batch_size,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            max_clip_grad_norm=args.max_grad_norm or None,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="TCKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """TCKT 前向传播。

        模型输出 ``pred[:, t]`` 预测 ``a[:, t]``（same-position 约定）。
        """
        e, at, a, it, c, mask = batch_data
        e = self._move_tensor_to_device(e)
        at = self._move_tensor_to_device(at)
        a = self._move_tensor_to_device(a)
        it = self._move_tensor_to_device(it)
        c = self._move_tensor_to_device(c)
        mask = self._move_tensor_to_device(mask)

        pred = self.model(e, at, a, it, c)  # [B, S]

        y_hat, y_label, _ = self._extract_valid_predictions(
            pred, a, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """BCE 逐元素后按 batch 内所有有效位置求和。"""
        y_hat = outputs["y_hat"].clamp(1e-7, 1.0 - 1e-7)
        return self.loss(y_hat, outputs["y_label"]).sum()

    @torch.inference_mode()
    def refresh_global_dict(self) -> None:
        """对当前模型在训练集上的交互嵌入做 K-Means，更新全局字典。"""
        from sklearn.cluster import MiniBatchKMeans

        device = self.device_
        rng = np.random.RandomState(self.seed)
        cap = self.cluster_sample_size

        chunks: list[np.ndarray] = []
        collected = 0
        for batch in self.train_data:
            e, _at, a, _it, c, mask = batch
            e = e.to(device)
            c = c.to(device)
            a = a.to(device)
            mask = mask.to(device)
            x = self.model.compute_interaction_embeddings(e, c, a)  # [B, S, d_k]
            emb = x[mask].detach().cpu().numpy().astype(np.float32)
            chunks.append(emb)
            collected += emb.shape[0]
            # 控制峰值内存：累积到 2 倍上限时先抽样合并一次。
            if collected >= cap * 2:
                merged = np.concatenate(chunks, axis=0)
                idx = rng.choice(merged.shape[0], cap, replace=False)
                chunks = [merged[idx]]
                collected = cap

        embeds = np.concatenate(chunks, axis=0)
        if embeds.shape[0] > cap:
            idx = rng.choice(embeds.shape[0], cap, replace=False)
            embeds = embeds[idx]

        # 样本数不足簇数时，用重复采样补齐（保证 K-Means 能产出 N 个中心）。
        n_samples = embeds.shape[0]
        if n_samples < self.global_dict_size:
            idx = rng.choice(n_samples, self.global_dict_size, replace=True)
            embeds = embeds[idx]

        km = MiniBatchKMeans(
            n_clusters=self.global_dict_size,
            random_state=int(self.seed),
            n_init=3,
            batch_size=4096,
        )
        km.fit(embeds)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        self.model.update_global_dict(centers)
        logger.debug(
            f"Global dict refreshed: clustered {embeds.shape[0]} embeddings "
            f"into {self.global_dict_size} centers"
        )
