"""DKT-Forget 模型训练器模块"""

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DKTForget")
class DKTForgetConfig(ModelConfig):
    """DKT-Forget 模型配置

    Args:
        emb_size: Embedding and LSTM hidden dimension.
        dropout: Dropout probability.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    emb_size: int = 100
    dropout: float = 0.1
    epochs: int = 100
    learning_rate: float = 1e-3
    lr_decay: float | None = None
    weight_decay: float = 0.0
    batch_size: int = 128


@register_trainer("DKTForget")
class DKTForgetTrainer(BaseTrainer):
    """DKT-Forget 模型训练器"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        # 准备数据
        from model.DKTForget.DKTForget_data import DKTForgetModelData

        model_data = DKTForgetModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        # 初始化模型
        from model.DKTForget.DKTForget_model import DKTForget

        m = rc.model
        logger.info(
            f"Initializing DKTForget model (emb_size={m.emb_size}, "
            f"num_rgap={model_data.num_rgap}, num_sgap={model_data.num_sgap}, "
            f"num_pcount={model_data.num_pcount})..."
        )
        metadata = data_src.get_metadata()
        model = DKTForget(
            num_c=metadata["num_skills"],
            num_rgap=model_data.num_rgap,
            num_sgap=model_data.num_sgap,
            num_pcount=model_data.num_pcount,
            emb_size=m.emb_size,
            dropout=m.dropout,
        )

        # 创建优化器和损失函数
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        # 创建学习率调度器
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
        self,
        batch_data: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> dict[str, torch.Tensor]:
        """DKT-Forget 前向传播

        Args:
            batch_data: (sequence, response, mask, rgaps, sgaps, pcounts)

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, rgaps, sgaps, pcounts = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        rgaps = self._move_tensor_to_device(rgaps)
        sgaps = self._move_tensor_to_device(sgaps)
        pcounts = self._move_tensor_to_device(pcounts)

        y_hat_full = self.model(sequence, response, rgaps, sgaps, pcounts)  # [B, S]

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
        """测试前向传播，支持 windowlateauc_mean 评估"""
        (
            sequence,
            response,
            mask,
            late_group_id,
            true_labels,
            question,
            rgaps,
            sgaps,
            pcounts,
        ) = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)
        rgaps = self._move_tensor_to_device(rgaps)
        sgaps = self._move_tensor_to_device(sgaps)
        pcounts = self._move_tensor_to_device(pcounts)

        y_hat_full = self.model(sequence, response, rgaps, sgaps, pcounts)  # [B, S]

        y_hat = torch.masked_select(y_hat_full, mask)
        y_label = torch.masked_select(true_labels, mask).float()
        group_ids = torch.masked_select(late_group_id, mask)
        question_ids = torch.masked_select(question, mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
            "question_id": question_ids,
        }
