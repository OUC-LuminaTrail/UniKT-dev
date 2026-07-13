from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["LBKTTrainer", "LBKTConfig"]


@register_model_config("LBKT")
class LBKTConfig(ModelConfig):
    """LBKT model configuration."""

    dim_tp: int = field(
        default=128,
        metadata={"help": "Topic (question) embedding dimension (default: 128)"},
    )
    dim_hidden: int = field(
        default=50, metadata={"help": "Response embedding dimension (default: 50)"}
    )
    num_units: int = field(
        default=128, metadata={"help": "Hidden dimension (default: 128)"}
    )
    dropout: float = field(
        default=0.2, metadata={"help": "Dropout rate (default: 0.2)"}
    )
    q_gamma: float = field(
        default=0.1, metadata={"help": "Q-matrix smoothing factor (default: 0.1)"}
    )
    epochs: int = field(
        default=100,
        metadata={"help": "Number of training epochs (default: 100)", "short": "ep"},
    )
    learning_rate: float = field(
        default=0.001,
        metadata={
            "help": "Learning rate for optimizer (default: 0.001)",
            "short": "lr",
        },
    )
    lr_decay_step: int = field(
        default=1, metadata={"help": "Learning rate decay step size (default: 1)"}
    )
    lr_decay_rate: float = field(
        default=0.5, metadata={"help": "Learning rate decay factor (default: 0.5)"}
    )
    weight_decay: float = field(
        default=1e-6, metadata={"help": "Weight decay for optimizer (default: 1e-6)"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training (default: 16)", "short": "bs"},
    )


@register_trainer("LBKT")
class LBKTTrainer(BaseTrainer):
    """LBKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.LBKT.LBKT_data import LBKTModelData
        from model.LBKT.LBKT_model import LBKT

        model_data = LBKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.q_matrix,
        ) = model_data.prepare_data(rc)

        logger.info("Initializing LBKT model...")
        metadata = data_src.get_metadata()
        m = rc.model
        model = LBKT(
            dim_tp=m.dim_tp,
            dim_hidden=m.dim_hidden,
            num_units=m.num_units,
            dropout=m.dropout,
            data_metadata=metadata,
        )

        # 优化器和损失函数
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
            eps=1e-8,
            betas=(0.1, 0.999),
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=m.lr_decay_step, gamma=m.lr_decay_rate
        )

        # build() hasn't run yet, so self.device_ is None. Resolve the target
        # device from rc to place the trainer-side q_matrix correctly.
        dev = rc.general.device
        device = (
            torch.device(dev)
            if dev
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.q_matrix = self.q_matrix.to(device)

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
        """LBKT 前向传播。

        Args:
            batch_data: 包含 (sequence, response, mask, time_factor, attempt_factor, hint_factor) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, time_factor, attempt_factor, hint_factor = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        time_factor = self._move_tensor_to_device(time_factor)
        attempt_factor = self._move_tensor_to_device(attempt_factor)
        hint_factor = self._move_tensor_to_device(hint_factor)

        preds = self.model(
            sequence,
            response,
            time_factor,
            attempt_factor,
            hint_factor,
            self.q_matrix,
        )

        y_hat, y_label, _ = self._extract_valid_predictions(
            preds, response, mask, same_position=True
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
