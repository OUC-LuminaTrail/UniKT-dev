"""extraKT 模型训练器"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("extraKT")
class extraKTConfig(ModelConfig):
    """extraKT model configuration."""

    d_model: int = field(
        default=256, metadata={"help": "Hidden dimension of the model"}
    )
    n_blocks: int = field(default=4, metadata={"help": "Number of transformer blocks"})
    num_attn_heads: int = field(
        default=8, metadata={"help": "Number of attention heads"}
    )
    dropout: float = field(default=0.05, metadata={"help": "Dropout probability"})
    d_ff: int = field(default=256, metadata={"help": "Feed-forward network dimension"})
    final_fc_dim: int = field(
        default=512, metadata={"help": "Final fully connected layer dimension"}
    )
    kq_same: int = field(
        default=1,
        metadata={
            "help": "Whether key and query use the same linear transformation (1=yes, 0=no)"
        },
    )
    separate_qa: int = field(
        default=0,
        metadata={"help": "Whether to use separate QA embeddings (1=yes, 0=no)"},
    )
    l2: float = field(
        default=1e-5,
        metadata={"help": "L2 regularization coefficient for Rasch model"},
    )
    num_buckets: int = field(
        default=32, metadata={"help": "Number of buckets for ALiBi relative position"}
    )
    max_distance: int = field(
        default=100,
        metadata={"help": "Maximum distance for ALiBi relative position"},
    )
    epochs: int = field(
        default=150, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for optimizer", "short": "lr"},
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
        default=64, metadata={"help": "Batch size for training", "short": "bs"}
    )


@register_trainer("extraKT")
class extraKTTrainer(BaseTrainer):
    """extraKT 模型训练器

    负责初始化extraKT模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.extraKT.extraKT_data import extraKTModelData

        model_data = extraKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.extraKT.extraKT_model import extraKT

        logger.info("Initializing extraKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)

        if n_pid > 0:
            logger.info(
                f"extraKT: Using Problem ID (Rasch model) with {n_pid} questions"
            )
        else:
            logger.info("extraKT: Problem ID not available, using skill-only model")

        m = rc.model
        model = extraKT(
            num_c=metadata["num_skills"],
            n_pid=n_pid,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            final_fc_dim=m.final_fc_dim,
            num_attn_heads=m.num_attn_heads,
            kq_same=m.kq_same,
            separate_qa=bool(m.separate_qa),
            l2=m.l2,
            num_buckets=m.num_buckets,
            max_distance=m.max_distance,
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

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build extraKT Rasch pid data with 0 reserved for padding."""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """extraKT 前向传播。

        extraKT预测语义：
        - y_hat[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]
        - y_hat[:, t] 直接对应 response[t]，需要对齐

        Args:
            batch_data: 包含 (sequence, response, mask, question) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else None

        y_hat_full, c_reg_loss = self.model(
            sequence, response, mask, pid_data
        )  # [B, S]

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
        )

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

        if use_pid:
            result["c_reg_loss"] = c_reg_loss

        return result

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估。

        batch_data: (sequence, response, mask, late_group_id, true_labels, question)
        """
        sequence, response, mask, late_group_id, true_labels, question = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else None

        y_hat_full, _ = self.model(sequence, response, mask, pid_data)  # [B, S]

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

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失，包含BCE损失和Rasch正则化损失"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            return bce_loss + outputs["c_reg_loss"]

        return bce_loss
