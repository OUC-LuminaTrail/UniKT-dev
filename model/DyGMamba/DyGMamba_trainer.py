"""DyGMamba 模型训练器。"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["DyGMambaTrainer", "DyGMambaConfig"]


@register_model_config("DyGMamba")
class DyGMambaConfig(ModelConfig):
    """DyGMamba model configuration.

    Args:
        channel_embedding_dim: Dimension per channel embedding.
        time_feat_dim: Time encoding dimension.
        num_layers: Number of Mamba layers.
        dropout: Dropout rate.
        max_input_sequence_length: Max input sequence length for neighbor padding.
        dual_view_time: Enable dual-view time encoder.
        d_state: Mamba SSM state dimension.
        d_conv: Mamba conv1d kernel size.
        expand: Mamba hidden state expansion factor.
        time_mamba: B/C computed from dts (time-mamba mode).
        no_selective: Disable selective SSM mechanism.
        remove_time_channel: Remove time channel from node features.
        hawkes_cross_dim: Hawkes cross-effect embedding dim (0 to disable).
        plain_mamba: Use plain Mamba without time modulation.
        num_neighbor: Number of neighbors for history.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization).
        batch_size: Batch size.
    """

    channel_embedding_dim: int = field(
        default=50, metadata={"optuna": {"type": "int", "low": 32, "high": 128}}
    )
    time_feat_dim: int = field(
        default=100, metadata={"optuna": {"type": "int", "low": 32, "high": 128}}
    )
    num_layers: int = field(
        default=2, metadata={"optuna": {"type": "int", "low": 1, "high": 4}}
    )
    dropout: float = field(
        default=0.1, metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}}
    )
    max_input_sequence_length: int = 512
    dual_view_time: bool = True
    d_state: int = field(
        default=16, metadata={"optuna": {"type": "int", "low": 8, "high": 64}}
    )
    d_conv: int = 4
    expand: int = field(
        default=2, metadata={"optuna": {"type": "int", "low": 1, "high": 4}}
    )
    time_mamba: bool = False
    no_selective: bool = False
    remove_time_channel: bool = False
    hawkes_cross_dim: int = field(
        default=32, metadata={"optuna": {"type": "int", "low": 0, "high": 64}}
    )
    plain_mamba: bool = False
    num_neighbor: int = field(
        default=50, metadata={"optuna": {"type": "int", "low": 10, "high": 100}}
    )
    epochs: int = 50
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
        default=2000,
        metadata={
            "optuna": {"type": "categorical", "choices": [512, 1024, 2000, 4096]}
        },
    )


@register_trainer("DyGMamba")
class DyGMambaTrainer(BaseTrainer):
    """DyGMamba 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DyGMamba.DyGMamba_data import DyGMambaModelData
        from model.DyGMamba.DyGMamba_model import DyGMamba

        m = rc.model

        model_data = DyGMambaModelData(data_src, cache=rc.general.cache)
        train_dataset, val_dataset, test_dataset, model_metadata = (
            model_data.prepare_data(rc)
        )

        # build() 前尚未设置 self.device_，手动推导以在目标设备上构建 Mamba 内核
        dev = rc.general.device
        device = (
            torch.device(dev)
            if dev
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("Initializing DyGMamba model...")
        model = DyGMamba(
            model_metadata,
            device=device,
            time_feat_dim=m.time_feat_dim,
            channel_embedding_dim=m.channel_embedding_dim,
            num_layers=m.num_layers,
            dropout=m.dropout,
            max_input_sequence_length=m.max_input_sequence_length,
            remove_time_channel=m.remove_time_channel,
            dual_view_time=m.dual_view_time,
            hawkes_cross_dim=m.hawkes_cross_dim,
            d_state=m.d_state,
            d_conv=m.d_conv,
            expand=m.expand,
            time_mamba=m.time_mamba,
            no_selective=m.no_selective,
            plain_mamba=m.plain_mamba,
        )

        optimizer = torch.optim.AdamW(
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
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            lr_scheduler=lr_scheduler,
            max_clip_grad_norm=10.0,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            collate_fn=train_dataset.get_batch,
            val_collate_fn=val_dataset.get_batch,
            test_collate_fn=test_dataset.get_batch,
        )

    def forward_pass(self, batch_data: dict) -> dict[str, torch.Tensor]:
        """DyGMamba 前向传播（交互级预测，无需 next-item 序列对齐）。"""
        batch = {
            key: self._move_tensor_to_device(value)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch_data.items()
        }

        outputs = self.model(batch)

        y_hat = outputs["logits"]
        y_label = batch["correctness"].float()
        y_prob = torch.sigmoid(y_hat)
        y_predict = self._generate_binary_predictions(y_prob, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_prob,
            "y_prob": y_prob,
            "src_embeddings": outputs["src_embeddings"],
            "dst_embeddings": outputs["dst_embeddings"],
            "dst_node_ids": batch["question"].long(),
        }
