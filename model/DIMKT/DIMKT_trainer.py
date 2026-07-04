"""DIMKT 模型训练器模块。"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("DIMKT")
class DIMKTModelParams(BaseParamConfig):
    """DIMKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "DIMKT Parameters"
        params = {
            "emb_size": {
                "type": int,
                "default": 128,
                "help": "Embedding size",
            },
            "dropout": {
                "type": float,
                "default": 0.2,
                "help": "Dropout probability",
            },
            "difficult_levels": {
                "type": int,
                "default": 100,
                "help": "Number of discrete difficulty levels D (sd/qd levels in [1, D+1])",
            },
            "epochs": {
                "type": int,
                "default": 100,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 5e-4,
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
                "default": 1e-4,
                "short": "wd",
                "help": "Weight decay (L2 regularization) for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size for training",
            },
        }
        return group_name, params


@register_trainer("DIMKT")
class DIMKTTrainer(BaseTrainer):
    """DIMKT 模型训练器。

    Args:
        args: 模型参数配置。
        data_src: 数据源实例。
        exp_manager: 实验管理器（可选）。
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        # 准备数据
        from model.DIMKT.DIMKT_data import DIMKTModelData

        model_data = DIMKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            skill_diff_table,
            question_diff_table,
        ) = model_data.prepare_data(args)

        # 初始化模型。
        from model.DIMKT.DIMKT_model import DIMKT

        metadata = data_src.get_metadata()
        num_q = metadata["num_questions"]
        num_c = metadata["num_skills"]
        logger.info(
            f"Initializing DIMKT model (emb_size={args.emb_size}, dropout={args.dropout}, "
            f"difficult_levels={args.difficult_levels}, num_q={num_q}, num_c={num_c})..."
        )

        model = DIMKT(
            num_q=num_q,
            num_c=num_c,
            dropout=args.dropout,
            emb_size=args.emb_size,
            batch_size=args.batch_size,
            difficult_levels=args.difficult_levels,
            skill_diff_table=skill_diff_table,
            question_diff_table=question_diff_table,
        )

        # 损失与优化器
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
            model_name="DIMKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def forward_pass(self, batch_data):
        """训练 / 验证前向传播。"""
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        y_hat_full = self._pad_to_full_sequence(
            self.model(sequence, question, response, mask)
        )
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=False
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

    def test_forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """测试前向传播，支持 windowlateauc_mean 评估。"""
        sequence, response, mask, late_group_id, true_labels, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        y = self.model(sequence, question, response, mask)  # [B, S-1]

        target_mask = mask[:, 1:].bool()
        y_hat = torch.masked_select(y, target_mask)
        y_label = torch.masked_select(true_labels[:, 1:], target_mask).float()
        group_ids = torch.masked_select(late_group_id[:, 1:], target_mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }
