"""DeepIRT trainer."""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("DeepIRT")
class DeepIRTModelParams(BaseParamConfig):
    """DeepIRT model parameter configuration."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "DeepIRT Parameters"
        params = {
            "dim_s": {
                "type": int,
                "default": 200,
                "help": "State dimension of key/value memory vectors",
            },
            "size_m": {
                "type": int,
                "default": 50,
                "help": "Number of memory slots",
            },
            "dropout": {
                "type": float,
                "default": 0.2,
                "help": "Dropout probability before ability/difficulty layers",
            },
            "emb_type": {
                "type": str,
                "default": "qid",
                "choices": ["qid"],
                "help": "Embedding type; PyKT DeepIRT qid path is supported",
            },
            "irt_scale": {
                "type": float,
                "default": 3.0,
                "help": "Scale applied to student ability in the IRT head",
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
                "help": "Learning rate for Adam optimizer",
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
                "help": "Weight decay for Adam optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size for training",
            },
            "test_batch_size": {
                "type": int,
                "default": 512,
                "help": "Batch size for windowlate test evaluation",
            },
            "test_num_workers": {
                "type": int,
                "default": 4,
                "help": "Number of DataLoader workers for windowlate test evaluation",
            },
            "test_pin_memory": {
                "type": bool,
                "default": True,
                "help": "Use pinned memory for windowlate test DataLoader",
            },
            "test_prefetch_factor": {
                "type": int,
                "default": 2,
                "help": "DataLoader prefetch factor for windowlate test evaluation",
            },
            "max_grad_norm": {
                "type": float,
                "default": 10.0,
                "help": "Max gradient norm for clipping",
            },
        }
        return group_name, params


@TRAINERS.register("DeepIRT")
class DeepIRTTrainer(BaseTrainer):
    """Trainer for DeepIRT."""

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.DeepIRT.DeepIRT_data import DeepIRTModelData
        from model.DeepIRT.DeepIRT_model import DeepIRT

        model_data = DeepIRTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        logger.info("Initializing DeepIRT model...")
        metadata = data_src.get_metadata()
        model = DeepIRT(
            num_c=metadata["num_skills"],
            dim_s=args.dim_s,
            size_m=args.size_m,
            dropout=args.dropout,
            emb_type=args.emb_type,
            irt_scale=args.irt_scale,
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
            max_clip_grad_norm=args.max_grad_norm,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            use_swanlab=getattr(args, "use_swanlab", True),
            model_name="DeepIRT",
            dataset_name=getattr(args, "dataset", ""),
            skip_test=getattr(args, "skip_test", False),
        ).build()

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(sequence, response, mask)

        # PyKT DeepIRT trains model(cc, cr) with y[:, 1:] against rshft.
        # The base helper's skip_first=True path would use y[:, :-1], which
        # is the wrong alignment for DKVMN-style same-position predictions.
        valid_mask = mask[:, 1:] & mask[:, :-1]
        y_hat = torch.masked_select(y_hat_full[:, 1:], valid_mask)
        y_label = torch.masked_select(response.float()[:, 1:], valid_mask)
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
        sequence, response, mask, late_group_id, true_labels, _ = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        y_hat_full = self.model(sequence, response, mask)
        mask_aligned = mask[:, 1:]
        y_hat = torch.masked_select(y_hat_full[:, 1:], mask_aligned)
        y_label = torch.masked_select(true_labels[:, 1:], mask_aligned).float()
        group_ids = torch.masked_select(late_group_id[:, 1:], mask_aligned)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }
