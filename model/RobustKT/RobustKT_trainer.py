"""RobustKT trainer."""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("RobustKT")
class RobustKTModelParams(BaseParamConfig):
    """RobustKT model parameter configuration."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "RobustKT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 256,
                "help": "Hidden dimension of the model",
            },
            "n_blocks": {
                "type": int,
                "default": 4,
                "help": "Number of transformer blocks",
            },
            "num_attn_heads": {
                "type": int,
                "default": 8,
                "help": "Number of attention heads",
            },
            "d_ff": {
                "type": int,
                "default": 512,
                "help": "Feed-forward network dimension",
            },
            "final_fc_dim": {
                "type": int,
                "default": 512,
                "help": "Final fully connected layer dimension",
            },
            "kernel_size": {
                "type": int,
                "default": 5,
                "help": "Causal smoothing kernel size",
            },
            "dropout": {
                "type": float,
                "default": 0.2,
                "help": "Dropout probability",
            },
            "kq_same": {
                "type": int,
                "default": 1,
                "help": "Whether key and query use the same linear transformation",
            },
            "separate_qa": {
                "type": int,
                "default": 0,
                "help": "Whether to use separate QA embeddings",
            },
            "l2": {
                "type": float,
                "default": 1e-5,
                "help": "Rasch difficulty regularization coefficient",
            },
            "epochs": {
                "type": int,
                "default": 150,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-4,
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
            "test_batch_size": {
                "type": int,
                "default": 512,
                "help": "Batch size for windowlate test evaluation",
            },
        }
        return group_name, params


@register_trainer("RobustKT")
class RobustKTTrainer(BaseTrainer):
    """Trainer for RobustKT."""

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.RobustKT.RobustKT_data import RobustKTModelData

        model_data = RobustKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        from model.RobustKT.RobustKT_model import RobustKT

        metadata = data_src.get_metadata()
        logger.info("Initializing RobustKT model...")
        if metadata.get("num_questions", 0) > 0:
            logger.info(
                "RobustKT: Using Problem ID (Rasch model) with "
                f"{metadata['num_questions']} questions"
            )
        else:
            logger.warning("RobustKT: Problem ID not available, using skill-only model")

        model = RobustKT(args=args, data_metadata=metadata)
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
            model_name="RobustKT",
            dataset_name=getattr(args, "dataset", ""),
            skip_test=getattr(args, "skip_test", False),
        ).build()

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        y_hat_full, c_reg_loss = self.model(
            sequence,
            response,
            mask,
            question=question,
        )

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
            "c_reg_loss": c_reg_loss,
        }

    def test_forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        sequence, response, mask, late_group_id, true_labels, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        valid_mask = late_group_id >= 0
        y_hat_full, _ = self.model(sequence, response, valid_mask, question=question)
        y_hat = torch.masked_select(y_hat_full, mask)
        y_label = torch.masked_select(true_labels, mask).float()
        group_ids = torch.masked_select(late_group_id, mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        bce_loss = self.loss(outputs["y_hat"], outputs["y_label"])
        return bce_loss + outputs.get(
            "c_reg_loss",
            torch.tensor(0.0, device=outputs["y_hat"].device),
        )
