"""DKT2 模型训练器。"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["DKT2Trainer"]


@register_model_config("DKT2")
class DKT2Config(ModelConfig):
    """DKT2 模型配置。"""

    factor: float = field(
        default=1.3, metadata={"help": "mLSTM/sLSTM feed-forward up-projection factor"}
    )
    num_blocks: int = field(default=1, metadata={"help": "Number of xLSTM blocks"})
    num_heads: int = field(
        default=2, metadata={"help": "Number of attention heads per block"}
    )
    slstm_at: list[int] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "Block indices using sLSTM (rest use mLSTM)",
            "nargs": "+",
        },
    )
    slstm_backend: str = field(
        default="cuda",
        metadata={"help": "sLSTM backend: cuda or vanilla (auto-fallback to vanilla)"},
    )
    conv1d_kernel_size: int = field(
        default=4, metadata={"help": "Conv1d kernel size in xLSTM blocks"}
    )
    qkv_proj_blocksize: int = field(
        default=4, metadata={"help": "Block size of the mLSTM qkv projection"}
    )
    embedding_size: int = field(
        default=64, metadata={"help": "Embedding / hidden dimension", "short": "ed"}
    )
    dropout: float = field(default=0.2, metadata={"help": "Dropout probability"})
    length: int = field(
        default=1, metadata={"help": "Prediction horizon (next-item when 1)"}
    )
    epochs: int = field(
        default=300, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-3, metadata={"help": "Learning rate", "short": "lr"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay", "short": "wd"}
    )
    max_grad_norm: float = field(
        default=2.0,
        metadata={
            "help": "Max gradient norm for clipping, 0 to disable",
            "short": "mgn",
        },
    )
    batch_size: int = field(default=512, metadata={"help": "Batch size", "short": "bs"})


@register_trainer("DKT2")
class DKT2Trainer(BaseTrainer):
    """DKT2 训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DKT2.DKT2_data import DKT2ModelData
        from model.DKT2.DKT2_model import DKT2

        model_data = DKT2ModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_skills,
            num_questions,
        ) = model_data.prepare_data(rc)
        max_seq_len = data_src.get_metadata("max_seq_len")

        m = rc.model
        logger.info("Initializing DKT2 model...")
        model = DKT2(
            num_skills=num_skills,
            num_questions=num_questions,
            batch_size=m.batch_size,
            seq_len=max_seq_len,
            factor=m.factor,
            num_blocks=m.num_blocks,
            num_heads=m.num_heads,
            slstm_at=m.slstm_at,
            conv1d_kernel_size=m.conv1d_kernel_size,
            qkv_proj_blocksize=m.qkv_proj_blocksize,
            embedding_size=m.embedding_size,
            dropout=m.dropout,
            slstm_backend=m.slstm_backend,
            length=m.length,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        logger.info(
            f"DKT2 Trainer: {num_skills} concepts (incl. padding), "
            f"{num_questions} questions, sLSTM backend = {model.slstm_backend}"
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(),
            max_clip_grad_norm=m.max_grad_norm or None,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(self, batch_data: dict) -> dict[str, torch.Tensor]:
        """next-item 前向传播。

        模型输出已过 sigmoid 的概率，长度 S-1，output[t] 预测 response[t+1]。
        用 _pad_to_full_sequence 补一列后走基类的 next-item 对齐。
        """
        questions = self._move_tensor_to_device(batch_data["questions"])
        responses = self._move_tensor_to_device(batch_data["responses"])
        masks = self._move_tensor_to_device(batch_data["masks"])
        skills = self._move_tensor_to_device(batch_data["skills"])

        output, _ = self.model(questions, skills, responses, masks)  # [B, S-1] probs

        y_hat_full = self._pad_to_full_sequence(output)
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, responses, masks
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
        }
