"""Trainer and configuration for ReKTP."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("ReKTP")
class ReKTPConfig(ModelConfig):
    """ReKTP configuration.

    Args:
        hidden_dim: Shared event, local-state, and encoder dimension.
        n_blocks: Number of global encoder blocks.
        d_state: Mamba SSM state dimension (only used when encoder_type='mamba').
        d_conv: Mamba local convolution width (only used when encoder_type='mamba').
        expand: Mamba internal expansion factor (only used when encoder_type='mamba').
        encoder_type: Global history encoder: 'mamba', 'lstm', or 'transformer'.
            The rest of the model is identical across choices.
        n_heads: Number of attention heads (only used when encoder_type='transformer').
        question_embed_dim: Intrinsic width of the per-question embedding; -1
            (default) means ``hidden_dim`` and 0 removes the pathway, leaving
            question identity to the ``question_diff`` scalar and the KC side.
            Widths below ``hidden_dim`` are lifted back by a shared projection,
            cutting per-question parameters to ``num_questions * dim``.
        residual_scale: Maximum Frobenius scale of each 2x2 residual block.
        dropout: Dropout probability.
        use_skill_cross_effect: Enable the linear-time Hawkes-style skill branch.
        cross_effect_rank: Low-rank width per cross-effect timescale.
        cross_effect_num_scales: Number of fixed exponential timescales, including
            one persistent component.
        epochs: Maximum training epochs.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        batch_size: Training batch size.
    """

    hidden_dim: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    n_blocks: int = field(
        default=4, metadata={"optuna": {"type": "int", "low": 1, "high": 4}}
    )
    d_state: int = field(
        default=30, metadata={"optuna": {"type": "int", "low": 8, "high": 32}}
    )
    d_conv: int = field(
        default=4,
        metadata={"optuna": {"type": "categorical", "choices": [2, 4]}},
    )
    expand: int = field(
        default=4,
        metadata={"optuna": {"type": "categorical", "choices": [1, 2, 4]}},
    )
    encoder_type: str = field(
        default="mamba",
        metadata={
            "optuna": {
                "type": "categorical",
                "choices": ["mamba", "lstm", "transformer"],
            }
        },
    )
    n_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8]}},
    )
    residual_scale: float = field(
        default=0.04,
        metadata={"optuna": {"type": "float", "low": 0.02, "high": 0.3, "log": True}},
    )
    dropout: float = field(
        default=0.4, metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}}
    )
    use_skill_cross_effect: bool = False
    cross_effect_rank: int = field(
        default=16,
        metadata={"optuna": {"type": "categorical", "choices": [8, 16, 32]}},
    )
    cross_effect_num_scales: int = field(
        default=4,
        metadata={"optuna": {"type": "categorical", "choices": [2, 4, 8]}},
    )
    question_embed_dim: int = field(
        default=-1,
        metadata={
            "optuna": {"type": "categorical", "choices": [0, 8, 16, 32, 64, 128, 256]}
        },
    )
    epochs: int = 100
    learning_rate: float = field(
        default=2.7e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True}},
    )
    weight_decay: float = field(
        default=1.4e-4,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=32,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 80, 128]}},
    )


@register_trainer("ReKTP")
class ReKTPTrainer(BaseTrainer):
    """Train ReKTP with one next-item objective per original question."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.ReKTP.ReKTP_data import ReKTPModelData
        from model.ReKTP.ReKTP_model import ReKTP

        model_data = ReKTPModelData(data_src)
        train_data, val_data, test_data, extra = model_data.prepare_data(rc)

        m = rc.model
        max_gap_bins = int(extra["max_gap_bins"])
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
            max_gap_bins=max_gap_bins,
            residual_scale=m.residual_scale,
            dropout=m.dropout,
            encoder_type=m.encoder_type,
            n_heads=m.n_heads,
            question_embed_dim=(
                None if m.question_embed_dim < 0 else m.question_embed_dim
            ),
            use_skill_cross_effect=m.use_skill_cross_effect,
            cross_effect_rank=m.cross_effect_rank,
            cross_effect_num_scales=m.cross_effect_num_scales,
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
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        questions, responses, times, mask = batch_data
        questions = self._move_tensor_to_device(questions)
        responses = self._move_tensor_to_device(responses)
        times = self._move_tensor_to_device(times)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        logits_full = self.model(questions, responses, times, mask)
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
