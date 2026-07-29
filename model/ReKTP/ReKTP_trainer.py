"""Trainer and configuration for ReKTP."""

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("ReKTP")
class ReKTPConfig(ModelConfig):
    """ReKTP configuration.

    Args:
        hidden_dim: Shared event, local-state, and Mamba dimension.
        n_blocks: Number of global Mamba blocks.
        d_state: Mamba SSM state dimension.
        d_conv: Mamba local convolution width.
        expand: Mamba internal expansion factor.
        max_gap_bins: Number of logarithmic same-KC gap buckets.
        residual_scale: Maximum Frobenius scale of each 2x2 residual block.
        dropout: Dropout probability.
        epochs: Maximum training epochs.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        batch_size: Training batch size.
    """

    hidden_dim: int = 128
    n_blocks: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    max_gap_bins: int = 16
    residual_scale: float = 0.1
    dropout: float = 0.2
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 80


@register_trainer("ReKTP")
class ReKTPTrainer(BaseTrainer):
    """Train ReKTP with one next-item objective per original question."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.ReKTP.ReKTP_data import ReKTPModelData
        from model.ReKTP.ReKTP_model import ReKTP

        model_data = ReKTPModelData(data_src, cache=rc.general.cache)
        train_data, val_data, test_data, extra = model_data.prepare_data(rc)

        m = rc.model
        logger.info("Initializing ReKTP model...")
        model = ReKTP(
            data_metadata=data_src.get_metadata(),
            question_skill_ids=extra["question_skill_ids"],
            question_skill_mask=extra["question_skill_mask"],
            hidden_dim=m.hidden_dim,
            n_blocks=m.n_blocks,
            d_state=m.d_state,
            d_conv=m.d_conv,
            expand=m.expand,
            max_gap_bins=m.max_gap_bins,
            residual_scale=m.residual_scale,
            dropout=m.dropout,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            lr_scheduler=None,
            max_clip_grad_norm=15.0,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        questions, responses, mask = batch_data
        questions = self._move_tensor_to_device(questions)
        responses = self._move_tensor_to_device(responses)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        logits_full = self.model(questions, responses, mask)
        logits, labels, _ = self._extract_valid_predictions(
            logits_full, responses, mask, same_position=False
        )
        logits, labels = self._handle_empty_batch(logits, labels)
        probabilities = torch.sigmoid(logits)
        return {
            "y_hat": logits,
            "y_label": labels,
            "y_predict": self._generate_binary_predictions(logits, threshold=0.0),
            "y_score": logits,
            "y_prob": probabilities,
        }


__all__ = ["ReKTPConfig", "ReKTPTrainer"]
