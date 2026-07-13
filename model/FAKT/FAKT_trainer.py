from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("FAKT")
class FAKTConfig(ModelConfig):
    """FAKT model configuration."""

    d_model: int = field(
        default=256, metadata={"help": "Hidden dimension of the model"}
    )
    n_blocks: int = field(default=2, metadata={"help": "Number of transformer blocks"})
    num_attn_heads: int = field(
        default=4, metadata={"help": "Number of attention heads"}
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    d_ff: int = field(default=256, metadata={"help": "Feed-forward network dimension"})
    final_fc_dim: int = field(
        default=256, metadata={"help": "Final fully connected layer dimension"}
    )
    final_fc_dim2: int = field(
        default=256,
        metadata={"help": "Second final fully connected layer dimension"},
    )
    kernel_size1: int = field(
        default=5,
        metadata={"help": "Kernel size for the first frequency-band causal conv"},
    )
    kernel_size2: int = field(
        default=5,
        metadata={"help": "Kernel size for the second frequency-band causal conv"},
    )
    kq_same: bool = field(
        default=True,
        metadata={"help": "Whether key and query share the linear transform"},
    )
    separate_qa: bool = field(
        default=False,
        metadata={"help": "Whether to use separate QA embeddings"},
    )
    emb_type: str = field(
        default="qidband",
        metadata={
            "help": "Embedding type (default enables frequency-band enhancement)"
        },
    )
    use_moe: bool = field(
        default=True, metadata={"help": "Whether to enable Mixture-of-Experts"}
    )
    num_experts: int = field(default=4, metadata={"help": "Number of experts in MoE"})
    confidence_thresholds: str = field(
        default="[0.8, 0.6, 0.4]",
        metadata={"help": "Confidence thresholds for adaptive expert selection"},
    )
    mamba_d_state: int = field(
        default=16, metadata={"help": "Mamba SSM state dimension"}
    )
    mamba_d_conv: int = field(
        default=4, metadata={"help": "Mamba local convolution width"}
    )
    mamba_expand: int = field(default=2, metadata={"help": "Mamba expansion factor"})
    min_experts: int = field(
        default=1, metadata={"help": "Minimum number of experts selected adaptively"}
    )
    max_experts: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of experts selected adaptively (None=num_experts)"
        },
    )
    epochs: int = field(
        default=200, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-4,
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
        default=128, metadata={"help": "Batch size for training", "short": "bs"}
    )


@register_trainer("FAKT")
class FAKTTrainer(BaseTrainer):
    """FAKT 模型训练器

    负责初始化 FAKT 模型、优化器和训练数据，并实现前向传播逻辑。
    """

    def build_components(self, rc, data_src):
        from model.FAKT.FAKT_data import FAKTModelData

        model_data = FAKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.FAKT.FAKT_model import FAKT

        logger.info("Initializing FAKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)
        if n_pid > 0:
            logger.info(f"FAKT: Using Problem ID (Rasch model) with {n_pid} questions")
        else:
            logger.info("FAKT: Problem ID not available, using skill-only model")

        m = rc.model
        model = FAKT(
            n_question=metadata["num_skills"],
            n_pid=n_pid,
            num_rgap=model_data.num_rgap,
            num_sgap=model_data.num_sgap,
            num_pcount=model_data.num_pcount,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            seq_len=metadata["max_seq_len"],
            kernel_size1=m.kernel_size1,
            kernel_size2=m.kernel_size2,
            freq=True,
            kq_same=m.kq_same,
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            num_attn_heads=m.num_attn_heads,
            separate_qa=m.separate_qa,
            emb_type=m.emb_type,
            use_moe=m.use_moe,
            num_experts=m.num_experts,
            confidence_thresholds=m.confidence_thresholds,
            mamba_d_state=m.mamba_d_state,
            mamba_d_conv=m.mamba_d_conv,
            mamba_expand=m.mamba_expand,
            min_experts=m.min_experts,
            max_experts=m.max_experts,
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
        self, question: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """构造 Rasch pid 数据：question + 1，padding 位置置 0。"""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """FAKT 前向传播"""
        sequence, response, mask, rgaps, sgaps, pcounts, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        rgaps = self._move_tensor_to_device(rgaps)
        sgaps = self._move_tensor_to_device(sgaps)
        pcounts = self._move_tensor_to_device(pcounts)
        question = self._move_tensor_to_device(question)

        # Rasch pid: question + 1 offset, padding (mask=0) positions set to 0.
        pid_data = (
            self._build_pid_data(question, mask) if self.model.n_pid > 0 else None
        )

        preds = self.model(sequence, response, rgaps, sgaps, pcounts, pid_data)

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

    def test_forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """测试前向传播，支持 windowlateauc_mean 评估。

        batch_data: (sequence, response, mask, late_group_id, true_labels, question,
                     rgaps, sgaps, pcounts)
        """
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

        # Rasch pid: question + 1 offset, padding (late_group_id<0) positions set to 0.
        valid_mask = late_group_id >= 0
        pid_data = (
            self._build_pid_data(question, valid_mask) if self.model.n_pid > 0 else None
        )

        preds = self.model(sequence, response, rgaps, sgaps, pcounts, pid_data)

        y_hat = torch.masked_select(preds, mask)
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
