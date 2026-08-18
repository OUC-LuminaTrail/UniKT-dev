"""DIMKT 模型训练器模块。"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DIMKT")
class DIMKTConfig(ModelConfig):
    """DIMKT model configuration.

    Args:
        emb_size: Embedding size.
        dropout: Dropout probability.
        difficult_levels: Number of discrete difficulty levels D (sd/qd levels in [1, D+1]).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    emb_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    dropout: float = field(
        default=0.2,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    # coupled to difficulty table construction in DIMKT_data, not searched
    difficult_levels: int = 100
    epochs: int = 100
    learning_rate: float = field(
        default=5e-4,
        metadata={
            "optuna": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
        },
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=1e-4,
        metadata={
            "optuna": {"type": "float", "low": 0.000001, "high": 0.001, "log": True}
        },
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("DIMKT")
class DIMKTTrainer(BaseTrainer):
    """DIMKT 模型训练器。

    Args:
        rc: RunConfig (OmegaConf DictConfig)。
        data_src: 数据源实例。
        exp_manager: 实验管理器（可选）。
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DIMKT.DIMKT_data import DIMKTModelData

        model_data = DIMKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            skill_diff_table,
            question_diff_table,
        ) = model_data.prepare_data(rc)

        from model.DIMKT.DIMKT_model import DIMKT

        metadata = data_src.get_metadata()
        num_q = metadata["num_questions"]
        num_c = metadata["num_skills"]
        m = rc.model
        logger.info(
            f"Initializing DIMKT model (emb_size={m.emb_size}, dropout={m.dropout}, "
            f"difficult_levels={m.difficult_levels}, num_q={num_q}, num_c={num_c})..."
        )

        model = DIMKT(
            num_q=num_q,
            num_c=num_c,
            dropout=m.dropout,
            emb_size=m.emb_size,
            batch_size=m.batch_size,
            difficult_levels=m.difficult_levels,
            skill_diff_table=skill_diff_table,
            question_diff_table=question_diff_table,
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
        """训练 / 验证前向传播。"""
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        y_hat_full = self._pad_to_full_sequence(
            self.model(sequence, question, response, mask)
        )
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=False
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

    def test_forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """测试前向传播，支持 windowlateauc_mean 评估。"""
        sequence, response, mask, late_group_id, true_labels, question, _ = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        y = self.model(sequence, question, response, mask)  # [B, S-1]

        target_mask = mask[:, 1:].bool()
        y_hat = torch.masked_select(y, target_mask)
        y_label = torch.masked_select(true_labels[:, 1:], target_mask).float()
        group_ids = torch.masked_select(late_group_id[:, 1:], target_mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }
