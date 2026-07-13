"""DenoiseKT 模型训练器。"""

from dataclasses import dataclass, field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DenoiseKT")
@dataclass
class DenoiseKTConfig(ModelConfig):
    """DenoiseKT 模型配置。"""

    d_model: int = field(
        default=256, metadata={"help": "Hidden dimension of the model"}
    )
    n_blocks: int = field(default=1, metadata={"help": "Number of transformer blocks"})
    num_attn_heads: int = field(
        default=8, metadata={"help": "Number of attention heads"}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout probability for the transformer"}
    )
    dropout1: float = field(
        default=0.1, metadata={"help": "Dropout probability for the GCN"}
    )
    d_ff: int = field(default=64, metadata={"help": "Feed-forward network dimension"})
    final_fc_dim: int = field(
        default=256, metadata={"help": "First output MLP dimension"}
    )
    final_fc_dim2: int = field(
        default=256, metadata={"help": "Second output MLP dimension"}
    )
    bf: float = field(
        default=0.9,
        metadata={"help": "Distance-decay base for same-concept boost focus"},
    )
    kq_same: int = field(
        default=1,
        metadata={
            "help": "Whether key and query share the linear projection (1=yes, 0=no)"
        },
    )
    epochs: int = field(
        default=200, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for optimizer", "short": "lr"},
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for optimizer", "short": "wd"}
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size for training", "short": "bs"}
    )


@register_trainer("DenoiseKT")
class DenoiseKTTrainer(BaseTrainer):
    """DenoiseKT 模型训练器。"""

    def build_components(self, rc, data_src):
        from model.DenoiseKT.DenoiseKT_data import DenoiseKTModelData

        model_data = DenoiseKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            question_concepts,
            question_graph,
        ) = model_data.prepare_data(rc)

        from model.DenoiseKT.DenoiseKT_model import DenoiseKT

        metadata = data_src.get_metadata()
        num_q = metadata["num_questions"]
        num_c = metadata["num_skills"]
        logger.info("Initializing DenoiseKT model...")

        m = rc.model
        model = DenoiseKT(
            num_c=num_c,
            num_q=num_q,
            question_concepts=question_concepts,
            question_graph=question_graph,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            dropout1=m.dropout1,
            bf=m.bf,
            d_ff=m.d_ff,
            seq_len=rc.data.max_seq_len,
            kq_same=m.kq_same,
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            num_attn_heads=m.num_attn_heads,
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
    ) -> dict:
        """前向传播。

        预测语义：
        ``preds[:, t]`` 使用 ``question[:, t]`` 与历史 ``qa[:, :t]`` 预测 ``response[:, t]``。

        batch_data: ``(question, response, mask)``
        """
        question, response, mask = batch_data
        question = self._move_tensor_to_device(question)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        preds = self.model(question, response)  # [B, S]

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
