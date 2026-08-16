from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DKVMN")
class DKVMNConfig(ModelConfig):
    """DKVMN model configuration.

    Args:
        dim_s: State dimension of memory vectors.
        size_m: Number of memory slots.
        dropout: Dropout probability.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    dim_s: int = field(
        default=200,
        metadata={"optuna": {"type": "categorical", "choices": [128, 200, 256]}},
    )
    size_m: int = field(
        default=50,
        metadata={"optuna": {"type": "int", "low": 20, "high": 100}},
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
    # log float cannot include 0, so categorical for the default of 0.0
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4]}},
    )
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("DKVMN")
class DKVMNTrainer(BaseTrainer):
    """DKVMN 模型训练器

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src):
        from model.DKVMN.DKVMN_data import DKVMNModelData

        model_data = DKVMNModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.DKVMN.DKVMN_model import DKVMN

        logger.info("Initializing DKVMN model...")
        metadata = data_src.get_metadata()
        m = rc.model
        model = DKVMN(
            num_c=metadata["num_skills"],
            dim_s=m.dim_s,
            size_m=m.size_m,
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
        """DKVMN 前向传播

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

        # DKVMN same-position output: p[:, t] predicts response[t] from history 0..t-1; same_position=True delegates alignment to the helper.
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
        """测试前向传播，支持 windowlateauc_mean 评估

        数据格式：
        - sequence: [技能历史, 目标技能]
        - response: [历史标签, 0]
        - mask: [0, ..., 0, 1]
        - late_group_id: [g1, ..., gN]
        - true_labels: [历史标签, 真实标签]

        DKVMN 预测语义：
        - p[:, t] 使用历史 0..t-1 预测位置 t
        - 所有位置均有有效预测
        """
        sequence, response, mask, late_group_id, true_labels, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        y_hat_full = self.model(sequence, response, mask)  # [B, S]

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
