"""PSKT trainer module."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("PSKT")
class PSKTConfig(ModelConfig):
    """PSKT model configuration.

    Args:
        embed_dim: Embedding dimension.
        max_time_interval: Maximum time interval in minutes.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    embed_dim: int = field(
        default=256,
        metadata={"optuna": {"type": "int", "low": 128, "high": 512}},
    )
    max_time_interval: int = 43200
    epochs: int = 200
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=1e-6,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("PSKT")
class PSKTTrainer(BaseTrainer):
    """PSKT model trainer."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.PSKT.PSKT_data import PSKTModelData
        from model.PSKT.PSKT_model import PSKT

        model_data = PSKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        metadata = data_src.get_metadata()
        num_q = metadata["num_questions"]
        num_c = metadata["num_skills"]
        m = rc.model
        logger.info(
            f"Initializing PSKT model: embed_dim={m.embed_dim}, "
            f"max_concepts={model_data.max_concepts}, num_q={num_q}, num_c={num_c}"
        )

        model = PSKT(
            num_questions=num_q,
            num_skills=num_c,
            embed_dim=m.embed_dim,
            max_concepts=model_data.max_concepts,
            max_time_interval=m.max_time_interval,
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

    def forward_pass(self, batch_data):
        """Args:
            batch_data: (question, response, mask, skills, timestamp) tuple.

        Returns:
            Dict with y_hat, y_label, y_predict.
        """
        sequence, response, mask, skills, timestamp = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        skills = self._move_tensor_to_device(skills)
        timestamp = self._move_tensor_to_device(timestamp)

        y = self.model(sequence, skills, response, timestamp, mask)
        y_hat, y_label, _ = self._extract_valid_predictions(
            y, response, mask, same_position=True
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
