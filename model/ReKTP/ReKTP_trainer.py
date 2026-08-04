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
        n_blocks: Number of global conv encoder blocks.
        conv_kernel_size: Causal-conv kernel size. Each block mixes the last
            ``kernel_size`` positions; the receptive field grows with dilation
            across blocks.
        conv_dilation_base: Dilation base per block; block ``i`` uses
            ``base**i``. 2 gives exponential receptive-field growth (TCN
            style), 1 gives uniform.
        use_global_film: If True, modulate the local KC input with the global
            state via FiLM (``local_global_film``). Off by default.
        state_block_size: Side of the square block in the local state
            transition. 2 (default) runs the fused Triton scan; 1 removes
            intra-block coupling (each dimension evolves independently) and
            larger sizes widen the coupled block, both via the serial scan
            fallback. Ablation knob for the block-affine transition.
        question_embed_dim: Intrinsic width of the per-question embedding; -1
            (default) means ``hidden_dim`` and 0 removes the pathway, leaving
            question identity to the ``question_diff`` scalar and the KC side.
            Widths below ``hidden_dim`` are lifted back by a shared projection,
            cutting per-question parameters to ``num_questions * dim``.
        residual_scale: Maximum Frobenius scale of each square residual block.
        dropout: Dropout probability.
        epochs: Maximum training epochs.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        batch_size: Training batch size.
        max_clip_grad_norm: Maximum gradient clipping norm.
        amp: Enable bf16 autocast for the forward pass. The matmul-heavy
            encoder and Linear layers run in bfloat16; backward is
            autograd-managed, and the custom triton scan keeps fp32 internally.
            Local to ReKTP for quick checking, not the framework-level
            precision node. Off by default.
    """

    hidden_dim: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    n_blocks: int = field(
        default=4, metadata={"optuna": {"type": "int", "low": 1, "high": 4}}
    )
    conv_kernel_size: int = field(
        default=3,
        metadata={"optuna": {"type": "categorical", "choices": [3, 5, 7]}},
    )
    conv_dilation_base: int = field(
        default=2,
        metadata={"optuna": {"type": "categorical", "choices": [1, 2]}},
    )
    use_global_film: bool = False
    state_block_size: int = field(
        default=2,
        metadata={"optuna": {"type": "categorical", "choices": [1, 2, 4]}},
    )
    amp: bool = False
    residual_scale: float = field(
        default=0.3,
        metadata={"optuna": {"type": "float", "low": 0.02, "high": 1.0, "log": True}},
    )
    dropout: float = field(
        default=0.4, metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}}
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
    max_clip_grad_norm: float = field(
        default=5.0,
        metadata={"optuna": {"type": "float", "low": 0.5, "high": 10, "log": True}},
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
            max_gap_bins=max_gap_bins,
            residual_scale=m.residual_scale,
            dropout=m.dropout,
            conv_kernel_size=m.conv_kernel_size,
            conv_dilation_base=m.conv_dilation_base,
            use_global_film=m.use_global_film,
            question_embed_dim=(
                None if m.question_embed_dim < 0 else m.question_embed_dim
            ),
            state_block_size=m.state_block_size,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            lr_scheduler=None,
            max_clip_grad_norm=m.max_clip_grad_norm,
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
        mask = self._move_tensor_to_device(mask)

        use_amp = bool(self.run_config.model.amp)
        with torch.autocast(
            device_type=self.device_.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits_full = self.model(questions, responses, times, mask)
        if use_amp:
            logits_full = logits_full.float()
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
