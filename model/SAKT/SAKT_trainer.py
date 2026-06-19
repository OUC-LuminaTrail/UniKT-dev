"""SAKT trainer."""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("SAKT")
class SAKTModelParams(BaseParamConfig):
    """SAKT hyperparameter configuration."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "SAKT Parameters"
        params = {
            "emb_size": {
                "type": int,
                "default": 256,
                "help": "Embedding dimension of interaction and exercise embeddings",
            },
            "num_attn_heads": {
                "type": int,
                "default": 8,
                "help": "Number of multi-head attention heads",
            },
            "num_en": {
                "type": int,
                "default": 1,
                "help": "Number of SAKT attention blocks",
            },
            "dropout": {
                "type": float,
                "default": 0.2,
                "help": "Dropout probability",
            },
            "epochs": {
                "type": int,
                "default": 200,
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
                "help": "Weight decay for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size for training",
            },
        }
        return group_name, params


@TRAINERS.register("SAKT")
class SAKTTrainer(BaseTrainer):
    """SAKT model trainer."""

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.SAKT.SAKT_data import SAKTModelData

        model_data = SAKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        from model.SAKT.SAKT_model import SAKT

        logger.info("Initializing SAKT model...")
        metadata = data_src.get_metadata()
        model = SAKT(
            num_c=metadata["num_skills"],
            seq_len=args.max_seq_len,
            emb_size=args.emb_size,
            num_attn_heads=args.num_attn_heads,
            dropout=args.dropout,
            num_en=args.num_en,
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
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="SAKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run SAKT forward pass on train/validation batches.

        Batch shapes:
        - sequence: [B, S] skill/concept ids
        - response: [B, S] binary labels
        - mask: [B, S] valid sequence positions

        Model output shape is [B, S-1], aligned to response[:, 1:].
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(sequence, response)
        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

    def test_forward_pass(self, batch_data):
        """Run SAKT forward pass for windowlate evaluation.

        batch_data: (sequence, response, mask, late_group_id, true_labels, question)
        Windowlate mask marks target positions only, so it is shifted with the
        model output by using mask[:, 1:].
        """
        sequence, response, mask, late_group_id, true_labels, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        y_hat_full = self.model(sequence, response)
        target_mask = mask[:, 1:].bool()
        y_hat = torch.masked_select(y_hat_full[:, :-1], target_mask)
        y_label = torch.masked_select(true_labels[:, 1:], target_mask)
        group_ids = torch.masked_select(late_group_id[:, 1:], target_mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        if y_label.numel() == 0:
            return y_hat.sum() * 0.0
        return self.loss(y_hat, y_label)
