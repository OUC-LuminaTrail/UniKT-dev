from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("DKT")
class DKTModelParams(BaseParamConfig):
    """DKT 模型参数配置

    Args:
        hidden_dim: 隐藏层维度
        embedding_dim: 嵌入维度
        dropout: Dropout概率
    """

    def define_params(self) -> tuple[str, dict]:
        """定义模型参数"""
        group_name = "DKT Parameters"
        params = {
            "hidden_dim": {
                "type": int,
                "default": 100,
                "help": "Hidden dimension of the model",
            },
            "embedding_dim": {
                "type": int,
                "default": 100,
                "help": "Embedding dimension of the model",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability",
            },
            "epochs": {
                "type": int,
                "default": 150,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 0.001,
                "short": "lr",
                "help": "Learning rate for optimizer",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch",
            },
            "weight_decay": {
                "type": float,
                "default": 0.0001,
                "short": "wd",
                "help": "Weight decay (L2 regularization) for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size for training",
            },
        }
        return group_name, params


@TRAINERS.register("DKT")
class DKTTrainer(BaseTrainer):
    """DKT 模型训练器

    负责初始化DKT模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        args: 模型参数配置
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        # 准备数据
        from model.DKT.DKT_data import DKTModelData

        model_data = DKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        # 初始化模型
        from model.DKT.DKT_model import DKT

        logger.info("Initializing DKT model...")
        metadata = data_src.get_metadata()
        model = DKT(
            num_c=metadata["num_skills"],
            emb_size=args.embedding_dim,
            dropout=args.dropout,
        )

        # 创建优化器和损失函数
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # 创建学习率调度器
        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        # 初始化基类训练器
        super().__init__(model)

        # 构建早停配置
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
            model_name="DKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """DKT 前向传播。

        Args:
            batch_data: 包含 (sequence, response, mask) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        # 解包数据并移动到设备
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # 模型前向传播
        y_hat_full = self.model(sequence, response, mask)  # [B, S]

        # 提取有效位置的预测和标签
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=False
        )

        # 处理空批次
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # 生成二分类预测
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

    def test_forward_pass(self, batch_data):
        sequence, response, mask, late_group_id = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)

        y_hat_full = self.model(sequence, response, mask)

        # DKT 模型在位置 t 的输出预测位置 t+1 的标签
        y_hat_aligned = y_hat_full[:, :-1]
        response_aligned = response.float()[:, 1:]
        mask_aligned = mask[:, 1:]
        group_id_aligned = late_group_id[:, 1:]

        y_hat = torch.masked_select(y_hat_aligned, mask_aligned)
        y_label = torch.masked_select(response_aligned, mask_aligned)
        results = self._aggregate_by_group(
            y_hat, y_label, group_id_aligned, mask_aligned
        )

        return {
            "y_hat": results["y_hat"],
            "y_label": results["y_label"],
            "y_predict": results["y_predict"],
            "y_score": results["y_hat"],
            "y_prob": results["y_hat"],
        }
