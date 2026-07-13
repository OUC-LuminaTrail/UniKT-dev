from dataclasses import dataclass, field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["ReKTTrainer", "ReKTConfig"]


@register_model_config("ReKT")
@dataclass
class ReKTConfig(ModelConfig):
    """ReKT model configuration."""

    hidden_dim: int = field(
        default=128, metadata={"help": "Hidden layer dimension", "short": "hd"}
    )
    dropout: float = field(
        default=0.4, metadata={"help": "Dropout rate", "short": "dp"}
    )
    epochs: int = field(
        default=70, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=0.002, metadata={"help": "Learning rate for optimizer", "short": "lr"}
    )
    weight_decay: float = field(
        default=1e-5,
        metadata={"help": "Weight decay (L2 regularization)", "short": "wd"},
    )
    batch_size: int = field(
        default=80, metadata={"help": "Batch size for training", "short": "bs"}
    )


@register_trainer("ReKT")
class ReKTTrainer(BaseTrainer):
    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.ReKT.ReKT_data import ReKTModelData
        from model.ReKT.ReKT_model import ReKT

        model_data = ReKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            extra_metadata,
        ) = model_data.prepare_data(rc)

        metadata = dict(data_src.get_metadata())
        metadata.update(extra_metadata)

        m = rc.model
        logger.info("Initializing ReKT model...")
        model = ReKT(metadata, hidden_dim=m.hidden_dim, dropout=m.dropout)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=None,
            max_clip_grad_norm=15.0,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

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
