"""SKT 模型训练器。"""

from dataclasses import field

import torch
from torch import nn

from utils.config import ModelConfig
from utils.core import register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents


@register_model_config("SKT")
class SKTConfig(ModelConfig):
    """SKT model configuration.

    Args:
        hidden_dim: Per-question state dimension h.
        latent_dim: Response embedding dimension.
        concept_dim: Concept embedding dimension.
        alpha: Aggregation weight between sync and propagation influence.
        dropout: Dropout probability before the output layer.
        self_dropout: Dropout probability after the self-influence update.
        graph_topk: Max successors / similar neighbors kept per question.
        use_checkpoint: Use gradient checkpointing per sequence step.
        epochs: Max training epochs (early stopping applies).
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    hidden_dim: int = field(
        default=16,
        metadata={"optuna": {"type": "categorical", "choices": [16, 32, 64]}},
    )
    latent_dim: int = 16
    concept_dim: int = 16
    alpha: float = 0.5
    dropout: float = 0.0
    self_dropout: float = 0.5
    graph_topk: int = 10
    use_checkpoint: bool = True
    epochs: int = 100
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    weight_decay: float = 1e-4
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("SKT")
class SKTTrainer(BaseTrainer):
    """SKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.SKT.SKT_data import SKTModelData
        from model.SKT.SKT_model import SKT

        train_dataset, val_dataset, test_dataset, info = SKTModelData(
            data_src
        ).prepare_data(rc)
        m = rc.model
        model = SKT(
            num_questions=info["num_questions"],
            hidden_dim=m.hidden_dim,
            latent_dim=m.latent_dim,
            concept_dim=m.concept_dim,
            alpha=m.alpha,
            neighbor_adj=info["neighbor_adj"],
            successor_adj=info["successor_adj"],
            dropout=m.dropout,
            self_dropout=m.self_dropout,
            use_checkpoint=m.use_checkpoint,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """前向传播：逐步全题 logits 中 gather 下一步题目，按 next-item 对齐。"""
        question, response, mask = batch_data
        question = self._move_tensor_to_device(question)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        logits = self.model(question, response, mask)  # [B, S, Q]

        # Keep only the next question's logit per step; sigmoid is elementwise,
        # so applying it after the gather is equivalent and much cheaper.
        next_idx = question[:, 1:].unsqueeze(-1)  # [B, S-1, 1]
        y_next = torch.sigmoid(
            torch.gather(logits[:, :-1], -1, next_idx).squeeze(-1)
        )  # [B, S-1]

        y_hat, y_label, _ = self._extract_valid_predictions(
            self._pad_to_full_sequence(y_next), response, mask
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

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """Clamp probabilities away from 0/1 before BCE for numerical safety."""
        y_hat = outputs["y_hat"].clamp(1e-7, 1.0 - 1e-7)
        return self.loss(y_hat, outputs["y_label"])
