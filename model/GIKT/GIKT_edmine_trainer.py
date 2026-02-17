"""
GIKT 模型训练器
定义 GIKT 模型特定的训练逻辑
"""

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["GIKTEdmineTrainer", "GIKTEdmineModelParams"]


@register_model_params("GIKTEdmine")
class GIKTEdmineModelParams(BaseParamConfig):
    """GIKT model-specific parameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "GIKT Parameters"
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
                "help": "Number of GNN hops (default: 3)",
            },
            "rank_k": {
                "type": int,
                "default": 10,
                "short": "rk",
                "help": "Rank K for low-rank approximation (default: 10)",
            },
            "dropout4gru": {
                "type": float,
                "default": 0.5,
                "help": "Dropout rate for GRU (default: 0.5)",
            },
            "dropout4gnn": {
                "type": float,
                "default": 0.5,
                "help": "Dropout rate for GNN (default: 0.5)",
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


@TRAINERS.register("GIKTEdmine")
class GIKTEdmineTrainer(BaseTrainer):
    """
    GIKT模型训练器
    """

    def __init__(
        self,
        args=None,
        data_src=None,
        exp_manager=None,
    ):
        # 1. 准备数据
        from model.GIKT import GIKTEdmineModelData

        model_data = GIKTEdmineModelData(data_src)
        train_dataset, val_dataset, question_neighbors, concept_neighbors, q_table = (
            model_data.prepare_data(args)
        )

        # 2. 初始化模型
        from model.GIKT.GIKT_edmine_model import GIKTEdmine

        logger.info("Initializing GIKT model...")
        model = GIKTEdmine(args, data_src.get_metadata())

        # 3. 调用父类构造函数
        super().__init__(model)

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

        # 6. 构建早停配置
        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=args.es_monitor,
                mode=args.es_mode,
                patience=es_patience,
                min_delta=args.es_min_delta,
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
            model_name="GIKTEdmine",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

        # 8. 移动静态数据到设备
        self.question_neighbors = self._move_tensor_to_device(question_neighbors)
        self.concept_neighbors = self._move_tensor_to_device(concept_neighbors)
        self.q_table = self._move_tensor_to_device(q_table)

    def forward_pass(self, batch_data):
        """GIKT 前向传播，使用基类辅助方法统一处理数据移动和预测生成"""
        # 解包数据并移动到设备
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # 模型前向传播
        # 模型在时刻 t 的输出预测的是 t+1 的标签
        y_hat_full = self.model(
            sequence,
            response,
            mask,
            self.question_neighbors,
            self.concept_neighbors,
            self.q_table,
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
        }
