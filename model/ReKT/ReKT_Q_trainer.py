import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["ReKT_QTrainer", "ReKT_QConfig"]


@register_model_config("ReKT_Q")
class ReKT_QConfig(ModelConfig):
    """ReKT_Q model configuration (question-level, original ReKT).

    Args:
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization).
        batch_size: Batch size for training.
    """

    hidden_dim: int = 128
    dropout: float = 0.4
    epochs: int = 70
    learning_rate: float = 0.002
    weight_decay: float = 1e-5
    batch_size: int = 80


@register_trainer("ReKT_Q")
class ReKT_QTrainer(BaseTrainer):
    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.ReKT.ReKT_model import ReKT
        from model.ReKT.ReKT_Q_data import ReKT_QModelData

        model_data = ReKT_QModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            extra_metadata,
        ) = model_data.prepare_data(rc)

        metadata = dict(data_src.get_metadata())
        metadata.update(extra_metadata)

        m = rc.model
        logger.info("Initializing ReKT_Q model...")
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
