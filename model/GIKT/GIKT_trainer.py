"""GIKT 模型训练器。

定义 GIKT 模型特定的训练逻辑。
"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["GIKTTrainer", "GIKTModelParams"]


@register_model_params("GIKT")
class GIKTModelParams(BaseParamConfig):
    """GIKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "GIKT Parameters"
        params = {
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


@TRAINERS.register("GIKT")
class GIKTTrainer(BaseTrainer):
    """GIKT 模型训练器"""

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        # 1. 初始化模型
        from model.GIKT.GIKT_model import GIKT

        logger.info("Initializing GIKT model...")
        model = GIKT(args, data_src.get_metadata())

        # 3. 准备数据
        from model.GIKT import GIKTModelData

        model_data = GIKTModelData(data_src)
        train_dataset, val_dataset, test_dataset, self.graph, self.question_skill_matrix = (
            model_data.prepare_data(args)
        )

        # 4. 创建优化器和损失函数
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # 5. 创建学习率调度器
        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        # 6. 初始化基类训练器
        super().__init__(model)

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
            model_name="GIKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

        # 8. 将静态数据移动到设备中
        self.graph = self.graph.to(self.device_)
        self.question_skill_matrix = self.question_skill_matrix.to(self.device_)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """GIKT 前向传播，使用基类辅助方法统一处理数据移动和预测生成。

        Args:
            batch_data: 包含 (sequence, response, mask) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        # 解包数据并移动到设备
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # 模型前向传播
        # 模型在时刻 t 的输出预测的是 t+1 的标签
        y_hat_full = self.model(
            sequence, response, mask, self.graph, self.question_skill_matrix
        )  # [B, S]

        # 提取有效位置的预测和标签
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=True
        )

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
        }
