"""DeepIRT trainer."""

from dataclasses import field
from typing import Literal

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DeepIRT")
class DeepIRTConfig(ModelConfig):
    """DeepIRT model configuration.

    Args:
        dim_s: State dimension of key/value memory vectors.
        size_m: Number of memory slots.
        dropout: Dropout probability before ability/difficulty layers.
        emb_type: Embedding type; DeepIRT qid path is supported.
        irt_scale: Scale applied to student ability in the IRT head.
        epochs: Number of training epochs.
        learning_rate: Learning rate for Adam optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay for Adam optimizer.
        batch_size: Batch size for training.
        test_batch_size: Batch size for windowlate test evaluation.
        test_num_workers: Number of DataLoader workers for windowlate test evaluation.
        test_pin_memory: Use pinned memory for windowlate test DataLoader.
        test_prefetch_factor: DataLoader prefetch factor for windowlate test evaluation.
        max_grad_norm: Max gradient norm for clipping; None disables clipping.
    """

    dim_s: int = field(
        default=200,
        metadata={"optuna": {"type": "int", "low": 64, "high": 512}},
    )
    size_m: int = field(
        default=50,
        metadata={"optuna": {"type": "int", "low": 16, "high": 100}},
    )
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    emb_type: Literal["qid"] = "qid"
    irt_scale: float = field(
        default=3.0,
        metadata={"optuna": {"type": "float", "low": 1.0, "high": 6.0}},
    )
    epochs: int = 200
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}},
    )
    lr_decay: float | None = None
    # linear range: log sampling requires low > 0, default is 0.0
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 1e-2}},
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128, 256]}},
    )
    test_batch_size: int = 64
    test_num_workers: int = 0
    test_pin_memory: bool = False
    test_prefetch_factor: int | None = None
    max_grad_norm: float | None = None


@register_trainer("DeepIRT")
class DeepIRTTrainer(BaseTrainer):
    """Trainer for DeepIRT."""

    def build_components(self, rc, data_src):
        from model.DeepIRT.DeepIRT_data import DeepIRTModelData
        from model.DeepIRT.DeepIRT_model import DeepIRT

        model_data = DeepIRTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        logger.info("Initializing DeepIRT model...")
        metadata = data_src.get_metadata()
        m = rc.model
        model = DeepIRT(
            num_c=metadata["num_skills"],
            dim_s=m.dim_s,
            size_m=m.size_m,
            dropout=m.dropout,
            emb_type=m.emb_type,
            irt_scale=m.irt_scale,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            max_clip_grad_norm=m.max_grad_norm,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(sequence, response, mask)

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
        }

    def test_forward_pass(self, batch_data):
        sequence, response, mask, late_group_id, true_labels, _, _ = batch_data
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
