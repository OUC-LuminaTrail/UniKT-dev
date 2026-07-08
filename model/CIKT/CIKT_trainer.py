"""CIKT 模型训练器模块。"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("CIKT")
class CIKTModelParams(BaseParamConfig):
    """CIKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "CIKT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 64,
                "help": "Hidden dimension (d_model).",
            },
            "dropout": {
                "type": float,
                "default": 0.5,
                "help": "GCN dropout probability (matches the reference implementation).",
            },
            "num_difficulty_levels": {
                "type": int,
                "default": 10,
                "help": "Number of question-difficulty bins (trivial head classes).",
            },
            "loss_w_causal": {
                "type": float,
                "default": 0.1,
                "help": "Loss weight for the causal branch (lambda_1 in paper).",
            },
            "loss_w_intervention": {
                "type": float,
                "default": 0.2,
                "help": "Loss weight for the response-invert intervention branch (lambda_3 in paper).",
            },
            "loss_w_trivial": {
                "type": float,
                "default": 0.6,
                "help": "Loss weight for the trivial (difficulty) branch (lambda_4 in paper).",
            },
            "loss_w_replace": {
                "type": float,
                "default": 0.3,
                "help": "Loss weight for the question-replace intervention branch (lambda_2 in paper).",
            },
            "epochs": {
                "type": int,
                "default": 100,
                "short": "ep",
                "help": "Number of training epochs.",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-3,
                "short": "lr",
                "help": "Learning rate for Adam.",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Exponential LR decay per epoch (None disables).",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-5,
                "short": "wd",
                "help": "Weight decay (L2).",
            },
            "batch_size": {
                "type": int,
                "default": 8,
                "short": "bs",
                "help": "Batch size.",
            },
        }
        return group_name, params


@register_trainer("CIKT")
class CIKTTrainer(BaseTrainer):
    """CIKT 模型训练器。"""

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.CIKT.CIKT_data import CIKTModelData
        from model.CIKT.CIKT_model import CIKT

        model_data = CIKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            difficulty_table,
            collate_fn,
        ) = model_data.prepare_data(args)

        metadata = data_src.get_metadata()
        num_questions = metadata["num_questions"]
        num_concepts = metadata["num_skills"]
        logger.info(
            f"Initializing CIKT model (d_model={args.d_model}, seq_len={args.max_seq_len}, "
            f"num_questions={num_questions}, num_concepts={num_concepts})..."
        )

        model = CIKT(
            num_questions=num_questions,
            num_concepts=num_concepts,
            d_model=args.d_model,
            seq_len=args.max_seq_len,
            dropout=args.dropout,
            num_difficulty_levels=args.num_difficulty_levels,
            difficulty_table=difficulty_table,
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

        self.w_causal = args.loss_w_causal
        self.w_intervention = args.loss_w_intervention
        self.w_trivial = args.loss_w_trivial
        self.w_replace = args.loss_w_replace
        self._ce_loss = torch.nn.CrossEntropyLoss()

        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 1e-3),
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
            collate_fn=collate_fn,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="CIKT",
            dataset_name=args.dataset,
        ).build()

    def forward_pass(self, batch_data):
        """训练 / 验证前向传播。

        batch_data: ``(Q, Y, mask, C, QR)``，模型输出 ``[B, L-1]``。
        """
        q, y, mask, c, qr = batch_data
        q = self._move_tensor_to_device(q)
        y = self._move_tensor_to_device(y)
        mask = self._move_tensor_to_device(mask)
        c = self._move_tensor_to_device(c)
        qr = self._move_tensor_to_device(qr)

        out = self.model(q, y, c, qr, mask)

        # pad 到 [B, L] 后由框架提取 valid_mask（mask[:-1] & mask[1:]）
        y_pred_full = self._pad_to_full_sequence(out["y_pred"])
        y_hat, y_label, valid_mask = self._extract_valid_predictions(
            y_pred_full, y, mask
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        a_true_full = self.model.difficulty_table[q][:, 1:]  # [B, L-1]

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "_aux_causal": out["y_causal"][valid_mask],
            "_aux_intervention": out["y_intervention"][valid_mask],
            "_aux_replace": out["y_replace"][valid_mask],
            "_aux_trivial": out["y_trivial"][valid_mask],
            "_aux_trivial_label": a_true_full[valid_mask],
        }

    def _compute_loss(self, outputs):
        """多任务损失"""
        y_label = outputs["y_label"]
        bce = self.loss
        loss_causal = bce(outputs["_aux_causal"], y_label)
        loss_intervention = bce(outputs["_aux_intervention"], y_label)
        loss_replace = bce(outputs["_aux_replace"], y_label)
        loss_trivial = self._ce_loss(
            outputs["_aux_trivial"], outputs["_aux_trivial_label"]
        )
        return (
            self.w_causal * loss_causal
            + self.w_intervention * loss_intervention
            + self.w_trivial * loss_trivial
            + self.w_replace * loss_replace
        )
