"""MIKT 模型训练器"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("MIKT")
class MIKTModelParams(BaseParamConfig):
    """MIKT 模型参数配置"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "MIKT Parameters"
        params = {
            "embed_dim": {
                "type": int,
                "default": 64,
                "short": "ed",
                "help": "Embedding dimension (default: 64)",
            },
            "state_dim": {
                "type": int,
                "default": 64,
                "short": "sd",
                "help": "State representation dimension (default: 64)",
            },
            "dropout": {
                "type": float,
                "default": 0.4,
                "short": "dp",
                "help": "Dropout rate (default: 0.4)",
            },
            "grad_clip": {
                "type": float,
                "default": 15.0,
                "help": "Gradient clipping norm (default: 15.0)",
            },
            "epochs": {
                "type": int,
                "default": 200,
                "short": "ep",
                "help": "Number of training epochs (default: 200)",
            },
            "learning_rate": {
                "type": float,
                "default": 0.002,
                "short": "lr",
                "help": "Learning rate for optimizer (default: 0.002)",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-5,
                "short": "wd",
                "help": "Weight decay for optimizer (default: 1e-5)",
            },
            "batch_size": {
                "type": int,
                "default": 80,
                "short": "bs",
                "help": "Batch size for training (default: 80)",
            },
        }
        return group_name, params


@register_trainer("MIKT")
class MIKTTrainer(BaseTrainer):
    """MIKT 模型训练器"""

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        from model.MIKT.MIKT_model import MIKT

        logger.info("Initializing MIKT model...")
        model = MIKT(args, data_src.get_metadata())

        from model.MIKT.MIKT_data import MIKTModelData

        model_data = MIKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.question_skill_matrix,
        ) = model_data.prepare_data(args)

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        lr_scheduler = None
        if getattr(args, "lr_decay", None):
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
            max_clip_grad_norm=args.grad_clip,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="MIKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

        self.question_skill_matrix = self.question_skill_matrix.to(self.device_)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """MIKT 前向传播

        模型输出 P[:, t] 预测 response[:, t+1]，即 [B, S-1] 的预测。
        对应标签为 response[:, 1:]，对应掩码为 mask[:, 1:]。
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # 模型输出 [B, S-1]，P[:, t] 预测 response[:, t+1]（next-item）；pad 到 [B, S] 后用内置函数
        y_hat_full = self._pad_to_full_sequence(
            self.model(sequence, response, mask, self.question_skill_matrix)
        )

        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

        return result
