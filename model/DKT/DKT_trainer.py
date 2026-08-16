from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DKT")
class DKTConfig(ModelConfig):
    """DKT model configuration.

    Args:
        hidden_dim: Hidden dimension of the model (alias for emb_size).
        embedding_dim: Embedding dimension of the model.
        dropout: Dropout probability.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    hidden_dim: int = 200
    embedding_dim: int = field(
        default=200,
        metadata={"optuna": {"type": "categorical", "choices": [128, 200, 256]}},
    )
    dropout: float = field(
        default=0.2,
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
    # categorical so the default 0.0 stays inside the space
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4]}},
    )
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("DKT")
class DKTTrainer(BaseTrainer):
    """DKT 模型训练器

    负责初始化DKT模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src):
        from model.DKT.DKT_data import DKTModelData
        from model.DKT.DKT_model import DKT

        train_dataset, val_dataset, test_dataset = DKTModelData(data_src).prepare_data(
            rc
        )
        metadata = data_src.get_metadata()
        m = rc.model
        logger.info("Initializing DKT model...")
        model = DKT(
            num_c=metadata["num_skills"], emb_size=m.embedding_dim, dropout=m.dropout
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
        """DKT 前向传播。

        Args:
            batch_data: 包含 (sequence, response, mask) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(sequence, response, mask)  # [B, S]

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
        }

    def test_forward_pass(self, batch_data):
        """
        测试前向传播，支持 windowlateauc_mean 评估。

        数据格式说明：
        - sequence: [技能历史, 目标技能]
        - response: [历史标签, 0]  # 目标位置 response=0 避免数据泄露
        - mask: [0, ..., 0, 1]  # 只有最后一个位置需要预测
        - late_group_id: [g1, ..., gN]  # 最后一个位置是当前题目的 group_id
        - true_labels: [历史标签, 真实标签]  # 用于评估

        DKT 预测语义：
        - y_hat[:, t] 预测的是 response[t]
        - y_hat[:, 0] = 0（无有效预测）
        - 铍要对齐：y_hat[:, 1:] 对应 true_labels[:, 1:]
        """
        sequence, response, mask, late_group_id, true_labels, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        y_hat_full = self.model(sequence, response, mask)  # [B, S]

        # DKT prediction alignment: y_hat[:, t] predicts response[t], but y_hat[:, 0] is 0
        # (no valid prediction), so valid predictions are y_hat[:, 1:] aligned to true_labels[:, 1:].
        y_hat_aligned = y_hat_full[:, 1:]
        true_labels_aligned = true_labels[:, 1:]
        mask_aligned = mask[:, 1:]
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
