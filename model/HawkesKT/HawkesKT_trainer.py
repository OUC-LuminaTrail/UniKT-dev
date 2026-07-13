from dataclasses import dataclass, field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("HawkesKT")
@dataclass
class HawkesKTConfig(ModelConfig):
    """HawkesKT model configuration."""

    emb_size: int = field(default=64, metadata={"help": "Size of embedding vectors"})
    time_log: float = field(
        default=2.718281828459045, metadata={"help": "Log base of time intervals"}
    )
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
        default=0.0,
        metadata={
            "help": "Weight decay (L2 regularization) for optimizer",
            "short": "wd",
        },
    )
    batch_size: int = field(
        default=128, metadata={"help": "Batch size for training", "short": "bs"}
    )


@register_trainer("HawkesKT")
class HawkesKTTrainer(BaseTrainer):
    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.HawkesKT.HawkesKT_data import HawkesKTModelData

        model_data = HawkesKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.HawkesKT.HawkesKT_model import HawkesKT

        logger.info("Initializing HawkesKT model...")
        metadata = data_src.get_metadata()
        m = rc.model
        model = HawkesKT(
            num_c=metadata["num_skills"],
            num_q=metadata["num_questions"],
            emb_size=m.emb_size,
            time_log=m.time_log,
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

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        skill, problem, time, label, mask = batch_data
        skill = self._move_tensor_to_device(skill)
        problem = self._move_tensor_to_device(problem)
        time = self._move_tensor_to_device(time)
        label = self._move_tensor_to_device(label)
        mask = self._move_tensor_to_device(mask)

        # Same-position output: y_hat_full[:, t] predicts label[:, t]; same_position=True applies built-in normalization
        y_hat_full = self.model(skill, problem, time, label)
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, label, mask, same_position=True
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
