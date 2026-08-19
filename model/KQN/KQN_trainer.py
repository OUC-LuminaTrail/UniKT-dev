"""KQN baseline trainer."""

from dataclasses import field
from typing import Literal

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("KQN")
class KQNConfig(ModelConfig):
    """KQN model and optimization parameters.

    Args:
        n_hidden: Shared dimensionality of knowledge-state and skill vectors.
        n_rnn_hidden: Hidden size of the RNN knowledge encoder.
        n_mlp_hidden: Hidden size of the skill encoder MLP.
        n_rnn_layers: Number of RNN layers in the knowledge encoder.
        rnn_type: RNN cell type for the knowledge encoder.
        dropout: Dropout probability applied to encoded knowledge states.
        epochs: Number of training epochs.
        learning_rate: Learning rate for Adam optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay for Adam optimizer.
        batch_size: Batch size for training.
        max_grad_norm: Optional max gradient norm for clipping.
    """

    n_hidden: int = field(
        default=128,
        metadata={"optuna": {"type": "int", "low": 64, "high": 256}},
    )
    n_rnn_hidden: int = field(
        default=128,
        metadata={"optuna": {"type": "int", "low": 64, "high": 256}},
    )
    n_mlp_hidden: int = 128
    n_rnn_layers: int = 1
    rnn_type: Literal["lstm", "gru"] = "lstm"
    dropout: float = field(
        default=0.4,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    epochs: int = 150
    learning_rate: float = field(
        default=1e-3,
        metadata={
            "optuna": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
        },
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4]}},
    )
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    max_grad_norm: float | None = None


@register_trainer("KQN")
class KQNTrainer(BaseTrainer):
    """Trainer for the KC-level KQN baseline."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.KQN.KQN_data import KQNModelData
        from model.KQN.KQN_model import KQN

        model_data = KQNModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        metadata = data_src.get_metadata()
        m = rc.model
        logger.info(
            f"Initializing KQN model with {metadata['num_skills']} skills, "
            f"rnn_type={m.rnn_type}"
        )
        model = KQN(
            num_skills=metadata["num_skills"],
            n_hidden=m.n_hidden,
            n_rnn_hidden=m.n_rnn_hidden,
            n_mlp_hidden=m.n_mlp_hidden,
            n_rnn_layers=m.n_rnn_layers,
            rnn_type=m.rnn_type,
            dropout=m.dropout,
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

    @staticmethod
    def _build_shifted_inputs(
        concept: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build KQN inputs for predicting the next response.

        Input batch shapes:
            question: [B, S], retained by data/trainer but unused by KQN.
            concept: [B, S], 0-based KC ids.
            response: [B, S], binary labels.
            mask: [B, S], true for valid training/validation sequence positions.

        Shift semantics:
            current_concept/current_response are positions ``0..S-2`` and form
            Interaction input ``x_t``.
            next_concept/next_response are positions ``1..S-1`` and represent
            the query KC and BCE target for ``c_{t+1}``.
            target_mask is true only when both current and next positions are
            real sequence events.
        """
        current_concept = concept[:, :-1]
        current_response = response[:, :-1]
        next_concept = concept[:, 1:]
        next_response = response[:, 1:]
        target_mask = mask[:, :-1] & mask[:, 1:]
        return (
            current_concept,
            current_response,
            next_concept,
            next_response,
            target_mask,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """Run a training/validation batch.

        Batch format is ``(concept, response, mask, question)``. ``question`` is
        kept for explicit batch semantics but is not used by the KQN baseline.
        """
        concept, response, mask, _question = batch_data
        concept = self._move_tensor_to_device(concept)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # _build_shifted_inputs yields the current/next concept pairs the model needs
        # (it emits [B, S-1] next-item predictions); pad to [B, S] and feed original response/mask.
        current_concept, current_response, next_concept, _, _ = (
            self._build_shifted_inputs(concept, response, mask)
        )

        y_hat_full = self._pad_to_full_sequence(
            self.model(current_concept, current_response, next_concept)
        )
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

    def test_forward_pass(self, batch_data: tuple[torch.Tensor, ...]):
        """Run a windowlate test batch.

        Windowlate batch format:
            ``(concept, response, mask, late_group_id, true_labels, question, user_id)``.

        In windowlate data, history positions are context only and usually have
        ``mask=False``; target positions have ``mask=True``. Therefore test
        selection uses the shifted target-side mask ``mask[:, 1:]`` instead of
        the training mask ``mask[:, :-1] & mask[:, 1:]``.
        """
        concept, response, mask, late_group_id, true_labels, _question, _ = batch_data
        concept = self._move_tensor_to_device(concept)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        current_concept = concept[:, :-1]
        current_response = response[:, :-1]
        next_concept = concept[:, 1:]

        y_hat_full = self.model(current_concept, current_response, next_concept)

        target_mask = mask[:, 1:].bool()
        y_hat = torch.masked_select(y_hat_full, target_mask)
        y_label = torch.masked_select(true_labels[:, 1:], target_mask).float()
        group_ids = torch.masked_select(late_group_id[:, 1:], target_mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }
