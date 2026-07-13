"""MIKT 模型训练器"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("MIKT")
class MIKTConfig(ModelConfig):
    """MIKT 模型配置。"""

    embed_dim: int = field(
        default=64,
        metadata={"help": "Embedding dimension (default: 64)", "short": "ed"},
    )
    state_dim: int = field(
        default=64,
        metadata={
            "help": "State representation dimension (default: 64)",
            "short": "sd",
        },
    )
    dropout: float = field(
        default=0.4, metadata={"help": "Dropout rate (default: 0.4)", "short": "dp"}
    )
    grad_clip: float = field(
        default=15.0, metadata={"help": "Gradient clipping norm (default: 15.0)"}
    )
    epochs: int = field(
        default=200,
        metadata={"help": "Number of training epochs (default: 200)", "short": "ep"},
    )
    learning_rate: float = field(
        default=0.002,
        metadata={
            "help": "Learning rate for optimizer (default: 0.002)",
            "short": "lr",
        },
    )
    weight_decay: float = field(
        default=1e-5,
        metadata={"help": "Weight decay for optimizer (default: 1e-5)", "short": "wd"},
    )
    batch_size: int = field(
        default=80,
        metadata={"help": "Batch size for training (default: 80)", "short": "bs"},
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )


@register_trainer("MIKT")
class MIKTTrainer(BaseTrainer):
    """MIKT 模型训练器"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.MIKT.MIKT_data import MIKTModelData
        from model.MIKT.MIKT_model import MIKT

        logger.info("Initializing MIKT model...")
        m = rc.model
        metadata = data_src.get_metadata()
        model = MIKT(
            embed_dim=m.embed_dim,
            state_dim=m.state_dim,
            dropout=m.dropout,
            data_metadata=metadata,
        )

        model_data = MIKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.question_skill_matrix,
        ) = model_data.prepare_data(rc)

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        # build() hasn't run yet, so self.device_ is None. Resolve the target
        # device from rc to place the trainer-side question_skill_matrix correctly.
        dev = rc.general.device
        device = (
            torch.device(dev)
            if dev
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.question_skill_matrix = self.question_skill_matrix.to(device)

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            max_clip_grad_norm=m.grad_clip,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """MIKT 前向传播

        模型输出 P[:, t] 预测 response[:, t+1]，即 [B, S-1] 的预测。
        对应标签为 response[:, 1:]，对应掩码为 mask[:, 1:]。
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # Model outputs [B, S-1]; P[:, t] predicts response[:, t+1] (next-item). Pad to [B, S] for built-in alignment.
        y_hat_full = self._pad_to_full_sequence(
            self.model(sequence, response, mask, self.question_skill_matrix)
        )

        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

        return result
