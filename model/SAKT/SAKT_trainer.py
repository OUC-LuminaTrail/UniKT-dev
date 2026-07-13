"""SAKT trainer."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("SAKT")
class SAKTConfig(ModelConfig):
    """SAKT model configuration."""

    emb_size: int = field(
        default=256,
        metadata={"help": "Embedding dimension of interaction and exercise embeddings"},
    )
    num_attn_heads: int = field(
        default=8, metadata={"help": "Number of multi-head attention heads"}
    )
    num_en: int = field(default=1, metadata={"help": "Number of SAKT attention blocks"})
    dropout: float = field(default=0.2, metadata={"help": "Dropout probability"})
    epochs: int = field(
        default=200, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-3, metadata={"help": "Learning rate for optimizer", "short": "lr"}
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for optimizer", "short": "wd"}
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size for training", "short": "bs"}
    )


@register_trainer("SAKT")
class SAKTTrainer(BaseTrainer):
    """SAKT model trainer."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.SAKT.SAKT_data import SAKTModelData

        model_data = SAKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.SAKT.SAKT_model import SAKT

        logger.info("Initializing SAKT model...")
        metadata = data_src.get_metadata()
        m = rc.model
        model = SAKT(
            num_c=metadata["num_skills"],
            seq_len=rc.data.max_seq_len,
            emb_size=m.emb_size,
            num_attn_heads=m.num_attn_heads,
            dropout=m.dropout,
            num_en=m.num_en,
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
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

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

        y_hat_full = self._pad_to_full_sequence(self.model(sequence, response))
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full,
            response,
            mask,
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
        y_hat = torch.masked_select(y_hat_full, target_mask)
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

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        if y_label.numel() == 0:
            return y_hat.sum() * 0.0
        return self.loss(y_hat, y_label)
