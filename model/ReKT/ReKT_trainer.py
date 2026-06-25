from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["ReKTTrainer", "ReKTModelParams"]


@register_model_params("ReKT")
class ReKTModelParams(BaseParamConfig):
    def define_params(self) -> tuple[str, dict]:
        return "ReKT Parameters", {
            "hidden_dim": {
                "type": int,
                "default": 128,
                "short": "hd",
                "help": "Hidden layer dimension (default: 128)",
            },
            "dropout": {
                "type": float,
                "default": 0.4,
                "short": "dp",
                "help": "Dropout rate (default: 0.4)",
            },
            "epochs": {
                "type": int,
                "default": 70,
                "short": "ep",
                "help": "Number of training epochs (default: 70)",
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
                "help": "Weight decay (L2 regularization) (default: 1e-5)",
            },
            "batch_size": {
                "type": int,
                "default": 80,
                "short": "bs",
                "help": "Batch size for training (default: 80)",
            },
        }


@TRAINERS.register("ReKT")
class ReKTTrainer(BaseTrainer):
    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        from model.ReKT import ReKTModelData
        from model.ReKT.ReKT_model import ReKT

        model_data = ReKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            extra_metadata,
        ) = model_data.prepare_data(args)

        metadata = dict(data_src.get_metadata())
        metadata.update(extra_metadata)

        logger.info("Initializing ReKT model...")
        model = ReKT(args, metadata)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
            lr_scheduler=None,
            max_clip_grad_norm=15.0,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="ReKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def forward_pass(
        self,
        batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        question, skill, response, mask = batch_data
        question = self._move_tensor_to_device(question)
        skill = self._move_tensor_to_device(skill)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        logits = self.model(question, skill, response, mask)

        y_hat, y_label, _ = self._extract_valid_predictions(
            logits, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
        }
