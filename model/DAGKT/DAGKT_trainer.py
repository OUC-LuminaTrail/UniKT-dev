"""DAGKT 模型训练器。

定义 DAGKT 模型特定的训练逻辑，包括组合损失函数。
"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["DAGKTTrainer", "DAGKTModelParams"]


@register_model_params("DAGKT")
class DAGKTModelParams(BaseParamConfig):
    """DAGKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "DAGKT Parameters"
        params = {
            # 基础架构参数（同 GIKT）
            "hidden_dim": {
                "type": int,
                "default": 100,
                "short": "hd",
                "help": "Hidden layer dimension (default: 100)",
            },
            "embedding_dim": {
                "type": int,
                "default": 100,
                "short": "ed",
                "help": "Embedding dimension (default: 100)",
            },
            "lstm_layers": {
                "type": int,
                "default": 2,
                "short": "ll",
                "help": "Number of LSTM layers (default: 2)",
            },
            "n_hop": {
                "type": int,
                "default": 3,
                "short": "nh",
                "help": "Number of GNN hops (default: 3)",
            },
            "heads": {
                "type": int,
                "default": 2,
                "short": "hs",
                "help": "Number of attention heads (default: 2)",
            },
            "history_neighbour": {
                "type": int,
                "default": 5,
                "short": "hn",
                "help": "History neighbor count (default: 5)",
            },
            "att_bound": {
                "type": float,
                "default": 0.2,
                "short": "ab",
                "help": "Attention bound (default: 0.2)",
            },
            "dropout": {
                "type": float,
                "default": 0.4,
                "short": "dp",
                "help": "Dropout rate (default: 0.4)",
            },
            # DAGKT 特有参数
            "ae_hidden_dim": {
                "type": int,
                "default": 50,
                "short": "aeh",
                "help": "Autoencoder hidden layer dimension (default: 50)",
            },
            "loss_diff_weight": {
                "type": float,
                "default": 1.0,
                "short": "ldw",
                "help": "Weight for difficulty autoencoder reconstruction loss (default: 1.0)",
            },
            "loss_attempt_weight": {
                "type": float,
                "default": 1.0,
                "short": "law",
                "help": "Weight for attempt autoencoder reconstruction loss (default: 1.0)",
            },
            # 训练参数（同 GIKT）
            "epochs": {
                "type": int,
                "default": 150,
                "short": "ep",
                "help": "Number of training epochs (default: 150)",
            },
            "learning_rate": {
                "type": float,
                "default": 0.001,
                "short": "lr",
                "help": "Learning rate for optimizer (default: 0.001)",
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
                "help": "Weight decay (L2 regularization) for optimizer (default: 0.0001)",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size for training (default: 128)",
            },
        }

        return group_name, params


@register_trainer("DAGKT")
class DAGKTTrainer(BaseTrainer):
    """DAGKT 模型训练器。

    在 GIKT 训练基础上增加辅助损失（难度和尝试次数的自编码器重建损失）。
    """

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        # 1. 准备数据
        from model.DAGKT.DAGKT_data import DAGKTModelData

        model_data = DAGKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.graph,
            self.question_skill_matrix,
            question_difficulty,
        ) = model_data.prepare_data(args)

        # 2. 初始化模型（传入题目难度）
        from model.DAGKT.DAGKT_model import DAGKT

        logger.info("Initializing DAGKT model...")
        model = DAGKT(args, data_src.get_metadata(), question_difficulty)

        # 3. 创建优化器和损失函数
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
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

        # 6. 保存辅助损失权重
        self.loss_diff_weight = getattr(args, "loss_diff_weight", 1.0)
        self.loss_attempt_weight = getattr(args, "loss_attempt_weight", 1.0)

        # 7. 构建早停配置
        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 0.0),
            )

        # 8. 配置训练器
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
            model_name="DAGKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

        # 9. 将静态数据移动到设备中
        self.graph = self.graph.to(self.device_)
        self.question_skill_matrix = self.question_skill_matrix.to(self.device_)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """DAGKT 前向传播。

        Args:
            batch_data: 包含 (sequence, response, mask, attempt_counts) 的四元组

        Returns:
            包含 y_hat, y_label, y_predict, _ae_loss_diff, _ae_loss_attempt 的字典
        """
        # 解包数据并移动到设备
        sequence, response, mask, attempt_counts = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        attempt_counts = self._move_tensor_to_device(attempt_counts, dtype=torch.float)

        # 模型前向传播
        y_hat_full, loss_diff, loss_attempt = self.model(
            sequence,
            response,
            mask,
            self.graph,
            self.question_skill_matrix,
            attempt_counts,
        )  # y_hat_full: [B, S]

        # 提取有效位置的预测和标签
        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

        # 处理空批次
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # 生成二分类预测
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
            "_ae_loss_diff": loss_diff,
            "_ae_loss_attempt": loss_attempt,
        }

    def _compute_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """计算组合损失: BCE + difficulty_ae_loss + attempt_ae_loss。"""
        base_loss = super()._compute_loss(outputs)

        loss_diff = outputs.get("_ae_loss_diff", torch.tensor(0.0))
        loss_attempt = outputs.get("_ae_loss_attempt", torch.tensor(0.0))

        total_loss = (
            base_loss
            + self.loss_diff_weight * loss_diff
            + self.loss_attempt_weight * loss_attempt
        )

        return total_loss
