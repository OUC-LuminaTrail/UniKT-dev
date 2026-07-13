"""DeepIRT trainer."""

from dataclasses import dataclass, field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DeepIRT")
@dataclass
class DeepIRTConfig(ModelConfig):
    """DeepIRT model configuration."""

    dim_s: int = field(
        default=200, metadata={"help": "State dimension of key/value memory vectors"}
    )
    size_m: int = field(default=50, metadata={"help": "Number of memory slots"})
    dropout: float = field(
        default=0.2,
        metadata={"help": "Dropout probability before ability/difficulty layers"},
    )
    emb_type: str = field(
        default="qid",
        metadata={
            "choices": ["qid"],
            "help": "Embedding type; DeepIRT qid path is supported",
        },
    )
    irt_scale: float = field(
        default=3.0,
        metadata={"help": "Scale applied to student ability in the IRT head"},
    )
    epochs: int = field(
        default=200, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for Adam optimizer", "short": "lr"},
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for Adam optimizer", "short": "wd"}
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size for training", "short": "bs"}
    )
    test_batch_size: int = field(
        default=64, metadata={"help": "Batch size for windowlate test evaluation"}
    )
    test_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of DataLoader workers for windowlate test evaluation"
        },
    )
    test_pin_memory: bool = field(
        default=False,
        metadata={"help": "Use pinned memory for windowlate test DataLoader"},
    )
    test_prefetch_factor: int | None = field(
        default=None,
        metadata={"help": "DataLoader prefetch factor for windowlate test evaluation"},
    )
    max_grad_norm: float | None = field(
        default=None,
        metadata={"help": "Max gradient norm for clipping; None disables clipping"},
    )


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
