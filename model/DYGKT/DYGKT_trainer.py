"""DYGKT 模型训练器。

定义 DYGKT 模型特定的训练逻辑。
"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["DYGKTTrainer", "DYGKTModelParams"]


@register_model_params("DYGKT")
class DYGKTModelParams(BaseParamConfig):
    """DYGKT 模型参数配置。"""

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
                "default": 1e-5,
                "short": "wd",
                "help": "Weight decay (L2 regularization) for optimizer (default: 1e-5)",
            },
            "max_grad_norm": {
                "type": float,
                "default": 10.0,
                "help": "Maximum norm for gradient clipping (default: 10.0)",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size for training (default: 64)",
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
        if not hasattr(args, 'device') or args.device is None or args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Auto-detected device: {args.device}")
        
        # 1. 准备数据
        from model.DYGKT import DYGKTModelData

        model_data = DYGKTModelData(data_src)
        train_dataset, val_dataset, test_dataset, model_metadata = model_data.prepare_data(args)

        # 2. 初始化模型
        from model.DYGKT.DYGKT_model import DYGKT

        logger.info("Initializing DYGKT model...")
        model = DYGKT(args, model_metadata)

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
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
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
        """执行一个训练批次，包含梯度裁剪。"""
        self.opt.zero_grad()
        output = self.forward_pass(batch_data)
        loss = self._compute_loss(output)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        
        self.opt.step()

        # 累积预测
        self.metrics_accumulator.update("train", output)

        return loss.item()

    def forward_pass(
        self, batch_data: dict
    ) -> dict[str, torch.Tensor]:
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
        
        # 模型前向传播，返回 logits
        y_hat = self.model(batch).float()  # [B]
        
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
        }
