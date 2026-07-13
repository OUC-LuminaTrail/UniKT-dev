"""DKT+ 模型训练器模块"""

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DKTPlus")
class DKTPlusConfig(ModelConfig):
    """DKT+ 模型配置。

    Args:
        emb_size: Embedding and LSTM hidden dimension.
        dropout: Dropout probability.
        lambda_r: Weight for current-step consistency loss (loss_r).
        lambda_w1: Weight for output smoothness L1 loss (loss_w1).
        lambda_w2: Weight for output smoothness L2 loss (loss_w2).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    emb_size: int = 200
    dropout: float = 0.1
    lambda_r: float = 0.2
    lambda_w1: float = 1.0
    lambda_w2: float = 10.0
    epochs: int = 150
    learning_rate: float = 1e-3
    lr_decay: float | None = None
    weight_decay: float = 0.0
    batch_size: int = 128


@register_trainer("DKTPlus")
class DKTPlusTrainer(BaseTrainer):
    """DKT+ 模型训练器"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DKTPlus.DKTPlus_data import DKTPlusModelData

        model_data = DKTPlusModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.DKTPlus.DKTPlus_model import DKTPlus

        metadata = data_src.get_metadata()
        m = rc.model

        logger.info("Initializing DKT+ model...")
        model = DKTPlus(
            num_c=metadata["num_skills"],
            emb_size=m.emb_size,
            lambda_r=m.lambda_r,
            lambda_w1=m.lambda_w1,
            lambda_w2=m.lambda_w2,
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
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """训练 / 验证前向传播。

        Args:
            batch_data: (sequence, response, mask)
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full, reg_loss = self.model(sequence, response, mask)

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "reg_loss": reg_loss,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """总损失 = next-item BCE + DKT+ 正则化损失。"""
        reg_loss = outputs.get("reg_loss", 0)
        return self.loss(outputs["y_hat"], outputs["y_label"]) + reg_loss

    def test_forward_pass(self, batch_data):
        sequence, response, mask, late_group_id, true_labels, _ = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        y_hat_full, _ = self.model(sequence, response, mask)  # [B, S]

        y_hat_aligned = y_hat_full[:, 1:]
        true_labels_aligned = true_labels[:, 1:]
        mask_aligned = mask[:, 1:].bool()
        group_id_aligned = late_group_id[:, 1:]

        y_hat = torch.masked_select(y_hat_aligned, mask_aligned)
        y_label = torch.masked_select(true_labels_aligned, mask_aligned).float()
        group_ids = torch.masked_select(group_id_aligned, mask_aligned)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }
