"""DKT+ 模型训练器模块"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("DKTPlus")
class DKTPlusModelParams(BaseParamConfig):
    """DKT+ 模型参数配置"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "DKTPlus Parameters"
        params = {
            "emb_size": {
                "type": int,
                "default": 200,
                "help": "Embedding and LSTM hidden dimension",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability",
            },
            "lambda_r": {
                "type": float,
                "default": 0.2,
                "help": "Weight for current-step consistency loss (loss_r)",
            },
            "lambda_w1": {
                "type": float,
                "default": 1.0,
                "help": "Weight for output smoothness L1 loss (loss_w1)",
            },
            "lambda_w2": {
                "type": float,
                "default": 10.0,
                "help": "Weight for output smoothness L2 loss (loss_w2)",
            },
            "epochs": {
                "type": int,
                "default": 150,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-3,
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
                "default": 0.0,
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


@register_trainer("DKTPlus")
class DKTPlusTrainer(BaseTrainer):
    """DKT+ 模型训练器"""

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        # 准备数据
        from model.DKTPlus.DKTPlus_data import DKTPlusModelData

        model_data = DKTPlusModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        # 初始化模型
        from model.DKTPlus.DKTPlus_model import DKTPlus

        metadata = data_src.get_metadata()

        logger.info("Initializing DKT+ model...")
        model = DKTPlus(
            num_c=metadata["num_skills"],
            emb_size=args.emb_size,
            lambda_r=args.lambda_r,
            lambda_w1=args.lambda_w1,
            lambda_w2=args.lambda_w2,
            dropout=args.dropout,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        super().__init__(model)

        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 0.0),
            )

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
            model_name="DKTPlus",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """训练 / 验证前向传播。

        Args:
            batch_data: (sequence, response, mask)
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full, reg_loss = self.model(sequence, response, mask)

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "reg_loss": reg_loss,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """总损失 = next-item BCE + DKT+ 正则化损失。"""
        reg_loss = outputs.get("reg_loss", 0)
        return self.loss(outputs["y_hat"], outputs["y_label"]) + reg_loss

    def test_forward_pass(self, batch_data):
        sequence, response, mask, late_group_id, true_labels, _ = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        y_hat_full, _ = self.model(sequence, response, mask)  # [B, S]，忽略 reg_loss

        y_hat_aligned = y_hat_full[:, 1:]
        true_labels_aligned = true_labels[:, 1:]
        mask_aligned = mask[:, 1:].bool()
        group_id_aligned = late_group_id[:, 1:]

        y_hat = torch.masked_select(y_hat_aligned, mask_aligned)
        y_label = torch.masked_select(true_labels_aligned, mask_aligned).float()
        group_ids = torch.masked_select(group_id_aligned, mask_aligned)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }
