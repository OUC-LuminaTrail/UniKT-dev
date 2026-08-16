"""CL4KT trainer."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("CL4KT")
class CL4KTConfig(ModelConfig):
    """CL4KT model configuration.

    Args:
        hidden_size: Hidden dimension.
        num_blocks: Transformer blocks per encoder.
        num_attn_heads: Attention heads.
        kq_same: Share the key/query projection.
        final_fc_dim: Output MLP width.
        d_ff: Feed-forward dimension.
        dropout: Dropout probability.
        reg_cl: Contrastive loss weight.
        mask_prob: BERT-style masking probability.
        crop_prob: Cropping probability.
        permute_prob: Permutation probability.
        replace_prob: Skill-difficulty replacement probability.
        negative_prob: Response-flip probability for hard negatives.
        temp: Contrastive similarity temperature.
        hard_negative_weight: Weight added to hard-negative logits.
        max_grad_norm: Max gradient norm (0 disables clipping).
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        batch_size: Batch size; larger batches give more in-batch negatives.
    """

    # powers of two so hidden_size % num_attn_heads == 0 for every combination
    hidden_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128]}},
    )
    num_blocks: int = field(
        default=2,
        metadata={"optuna": {"type": "int", "low": 1, "high": 4}},
    )
    num_attn_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8]}},
    )
    kq_same: bool = True
    final_fc_dim: int = 512
    d_ff: int = 1024
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    reg_cl: float = 0.1
    mask_prob: float = 0.2
    crop_prob: float = 0.3
    permute_prob: float = 0.3
    replace_prob: float = 0.3
    negative_prob: float = 1.0
    temp: float = 0.05
    hard_negative_weight: float = 1.0
    max_grad_norm: float = 2.0
    epochs: int = 150
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    # categorical so the default 0.0 stays inside the space
    weight_decay: float = field(
        default=0.0,
        metadata={
            "optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4, 1e-3]}
        },
    )
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("CL4KT")
class CL4KTTrainer(BaseTrainer):
    """CL4KT trainer.

    Training batches carry two augmented views plus the original sequence; the
    forward pass computes a contrastive loss on the views and a BCE loss on the
    original. Evaluation and test batches carry only the original sequence.
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.CL4KT.CL4KT_data import CL4KTModelData

        model_data = CL4KTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.CL4KT.CL4KT_model import CL4KT

        logger.info("Initializing CL4KT model...")
        metadata = data_src.get_metadata()
        m = rc.model
        model = CL4KT(
            num_skills=metadata["num_skills"],
            hidden_size=m.hidden_size,
            num_blocks=m.num_blocks,
            num_attn_heads=m.num_attn_heads,
            kq_same=m.kq_same,
            final_fc_dim=m.final_fc_dim,
            d_ff=m.d_ff,
            dropout=m.dropout,
            temp=m.temp,
            hard_negative_weight=m.hard_negative_weight,
            negative_prob=m.negative_prob,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            max_clip_grad_norm=m.max_grad_norm if m.max_grad_norm > 0 else None,
        )

    def forward_pass(self, batch_data) -> dict:
        """Forward pass branching on the model's train/eval mode.

        Position t of the output predicts response t, so predictions are
        extracted with same-position alignment.
        """
        result = {}
        if self.model.training:
            (s_i, s_j, s), (r_i, r_j, r, neg_r), (m_i, m_j, m) = batch_data
            s_i = self._move_tensor_to_device(s_i)
            s_j = self._move_tensor_to_device(s_j)
            s = self._move_tensor_to_device(s)
            r_i = self._move_tensor_to_device(r_i)
            r_j = self._move_tensor_to_device(r_j)
            r = self._move_tensor_to_device(r)
            neg_r = self._move_tensor_to_device(neg_r)
            m_i = self._move_tensor_to_device(m_i)
            m_j = self._move_tensor_to_device(m_j)
            m = self._move_tensor_to_device(m)

            cl_loss = self.model.compute_cl_loss(
                s, s_i, s_j, r_i, r_j, neg_r, m_i, m_j, m
            )
            y_hat_full = self.model.predict(s, r)
            result["cl_loss"] = cl_loss
            ref_mask = m
        else:
            s, r, m = batch_data
            s = self._move_tensor_to_device(s)
            r = self._move_tensor_to_device(r)
            m = self._move_tensor_to_device(m)
            y_hat_full = self.model.predict(s, r)
            ref_mask = m

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, r, ref_mask.bool(), same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result.update(
            {
                "y_hat": y_hat,
                "y_label": y_label,
                "y_predict": y_predict,
                "y_score": y_hat,
                "y_prob": y_hat,
            }
        )
        return result

    def test_forward_pass(self, batch_data) -> dict:
        """Test forward pass over windowlate samples.

        batch_data: (sequence, response, mask, late_group_id, true_labels, question).
        """
        sequence, response, mask, late_group_id, true_labels, _ = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        y_hat_full = self.model.predict(sequence, response)
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
        """BCE loss on the original view plus the weighted contrastive loss."""
        bce_loss = self.loss(outputs["y_hat"], outputs["y_label"])
        if "cl_loss" in outputs:
            return bce_loss + self.run_config.model.reg_cl * outputs["cl_loss"]
        return bce_loss
