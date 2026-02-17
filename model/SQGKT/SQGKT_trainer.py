"""SQGKT 模型训练器。

定义 SQGKTTrainer 类，用于训练和评估 SQGKT 模型。
"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("SQGKT")
class SQGKTModelParams(BaseParamConfig):
    """SQGKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "SQGKT Parameters"
        params = {
            "dim_emb": {
                "type": int,
                "default": 100,
                "short": "ed",
                "help": "Embedding dimension (default: 100)",
            },
            "agg_hops": {
                "type": int,
                "default": 3,
                "short": "nh",
                "help": "Number of GNN hops (default: 3)",
            },
            "dropout4lstm": {
                "type": float,
                "default": 0.2,
                "short": "dpl",
                "help": "LSTM dropout probability (default: 0.2)",
            },
            "dropout4gnn": {
                "type": float,
                "default": 0.4,
                "short": "dpg",
                "help": "GNN dropout probability (default: 0.4)",
            },
            "qs_question_neighbors": {
                "type": int,
                "default": 5,
                "help": "Question neighbors in question-skill graph (default: 5)",
            },
            "qs_skill_neighbors": {
                "type": int,
                "default": 10,
                "help": "Skill neighbors in question-skill graph (default: 10)",
            },
            "uq_user_neighbors": {
                "type": int,
                "default": 5,
                "help": "User neighbors in user-question graph (default: 5)",
            },
            "uq_question_neighbors": {
                "type": int,
                "default": 5,
                "help": "Question neighbors in user-question graph (default: 5)",
            },
            "rank_k": {
                "type": int,
                "default": 10,
                "help": "Top K for soft review mechanism (default: 10)",
            },
            "epochs": {
                "type": int,
                "default": 300,
                "short": "ep",
                "help": "Number of training epochs (default: 200)",
            },
            "learning_rate": {
                "type": float,
                "default": 0.001,
                "short": "lr",
                "help": "Learning rate for optimizer (default: 0.001)",
            },
            "lr_decay_factor": {
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


@TRAINERS.register("SQGKT")
class SQGKTTrainer(BaseTrainer):
    """SQGKT 模型训练器。

    负责初始化 SQGKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        args: 模型参数配置
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        # 1. 准备数据
        from model.SQGKT.SQGKT_data import SQGKTModelData

        model_data = SQGKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            qs_table,
            q_neighbors_qs,
            c_neighbors_qs,
            uq_table,
            u_neighbors_uq,
            q_neighbors_uq,
        ) = model_data.prepare_data(args)

        # 2. 初始化模型
        from model.SQGKT import SQGKT

        logger.info("Initializing SQGKT model...")
        model = SQGKT(args, data_src.get_metadata())

        # 3. 调用父类构造函数
        super().__init__(model)

        # 4. 创建优化器和损失函数
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # 5. 创建学习率调度器
        lr_scheduler = None
        if args.lr_decay_factor:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay_factor
            )

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
            batch_size=args.batch_size,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="SQGKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

        # 8. 移动静态数据到设备
        self.qs_table = self._move_tensor_to_device(qs_table)
        self.q_neighbors_qs = self._move_tensor_to_device(q_neighbors_qs)
        self.c_neighbors_qs = self._move_tensor_to_device(c_neighbors_qs)
        self.uq_table = self._move_tensor_to_device(uq_table)
        self.u_neighbors_uq = self._move_tensor_to_device(u_neighbors_uq)
        self.q_neighbors_uq = self._move_tensor_to_device(q_neighbors_uq)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """SQGKT 前向传播，使用基类辅助方法统一处理数据移动和预测生成。

        Args:
            batch_data: 包含 (users, sequence, response, mask) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        # 解包数据并移动到设备
        users, sequence, response, mask = batch_data
        users = self._move_tensor_to_device(users)
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # 模型前向传播
        y_hat_full = self.model(
            users,
            sequence,
            response,
            mask,
            self.qs_table,
            self.q_neighbors_qs,
            self.c_neighbors_qs,
            self.uq_table,
            self.u_neighbors_uq,
            self.q_neighbors_uq,
        )  # [B, S]

        # 提取有效位置的预测和标签（跳过第一个时间步）
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
        }
