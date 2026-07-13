"""StableKT 模型训练器模块"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("StableKT")
class StableKTConfig(ModelConfig):
    """StableKT model configuration."""

    d_model: int = field(default=256, metadata={"help": "Dimension of the model"})
    n_blocks: int = field(default=2, metadata={"help": "Number of transformer blocks"})
    n_heads: int = field(
        default=4,
        metadata={"help": "Number of attention heads (must be even for HAKT)"},
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    d_ff: int = field(
        default=256, metadata={"help": "Dimension of feed-forward network"}
    )
    kq_same: int = field(
        default=1,
        metadata={"help": "Whether to share key and query weights (1 yes, 0 no)"},
    )
    separate_qa: int = field(
        default=0,
        metadata={
            "help": "Whether to use separate interaction embedding (1 yes, 0 no)"
        },
    )
    final_fc_dim: int = field(
        default=512,
        metadata={"help": "First fully connected layer dimension in output"},
    )
    final_fc_dim2: int = field(
        default=256,
        metadata={"help": "Second fully connected layer dimension in output"},
    )
    emb_type: str = field(
        default="qid",
        metadata={
            "help": "Embedding type: qid, qid_woha, qid_sin, qid_t5, qid_rotary, qid_wha, etc."
        },
    )
    r: float = field(
        default=1.0, metadata={"help": "Penumbral cone radius for HAKT attention"}
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Penumbral cone temperature parameter for HAKT attention"},
    )
    num_buckets: int = field(
        default=32, metadata={"help": "Number of buckets for T5 relative position bias"}
    )
    max_distance: int = field(
        default=100,
        metadata={"help": "Maximum distance for T5 relative position bias"},
    )
    epochs: int = field(
        default=100, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Learning rate for optimizer", "short": "lr"}
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
        default=128, metadata={"help": "Batch size for training", "short": "bs"}
    )


@register_trainer("StableKT")
class StableKTTrainer(BaseTrainer):
    """StableKT 模型训练器

    负责初始化 StableKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.StableKT.StableKT_data import StableKTModelData

        model_data = StableKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.StableKT.StableKT_model import StableKT

        logger.info("Initializing StableKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata["num_questions"]
        logger.info(f"StableKT: Using Problem ID (Rasch model) with {n_pid} questions")

        m = rc.model
        model = StableKT(
            num_skills=metadata["num_skills"],
            n_pid=n_pid,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            n_heads=m.n_heads,
            seq_len=rc.data.max_seq_len,
            kq_same=m.kq_same,
            separate_qa=bool(m.separate_qa),
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            emb_type=m.emb_type,
            r=m.r,
            gamma=m.gamma,
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
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """StableKT 前向传播

        StableKT 预测语义：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - 第一个位置：y_hat[:, 0] 基于 sequence[0:1] 和空历史预测 response[0]
        - y_hat[:, t] 直接对应 response[t]，需要对齐

        Args:
            batch_data: 包含 (sequence, response, mask, question) 的元组（question 必选，用于Rasch pid）

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        pid_data = self._build_pid_data(question, mask)

        y_hat_full = self.model(sequence, response, mask, pid_data)

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

        数据格式说明：
        - sequence: [技能历史, 目标技能]
        - response: [历史标签, 0]  # 目标位置 response=0 用于避免数据泄露
        - mask: [0, ..., 0, 1]  # 只有最后一个位置需要预测
        - late_group_id: [g1, ..., gN]  # 最后一个位置是当前题目的 group_id
        - true_labels: [历史标签, 真实标签]  # 用于评估
        - question: [题目历史, 目标题目]  # 用于Rasch pid

        StableKT 预测语义：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - y_hat[:, t] 直接对应 response[t]
        """
        sequence, response, mask, late_group_id, true_labels, question = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask)

        y_hat_full = self.model(sequence, response, mask, pid_data)

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
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        return self.loss(y_hat, y_label)
