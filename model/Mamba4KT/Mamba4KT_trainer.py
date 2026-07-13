"""Mamba4KT 模型训练器"""

from dataclasses import dataclass, field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("Mamba4KT")
@dataclass
class Mamba4KTConfig(ModelConfig):
    """Mamba4KT model configuration."""

    d_model: int = field(
        default=128,
        metadata={"help": "Hidden dimension of the model (paper: {64,128,256})"},
    )
    n_blocks: int = field(
        default=5, metadata={"help": "Number of Mamba blocks (paper N=5)"}
    )
    d_state: int = field(default=16, metadata={"help": "SSM latent state dimension"})
    d_conv: int = field(
        default=4, metadata={"help": "Conv1D kernel width in Mamba block"}
    )
    expand: int = field(
        default=2,
        metadata={
            "help": "Mamba internal expansion factor (Conv1D out channels = expand*d_model)"
        },
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    l2: float = field(
        default=1e-5,
        metadata={
            "help": "L2 regularization coefficient for Rasch difficulty parameter (lambda in Eq.11)"
        },
    )
    epochs: int = field(
        default=150, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={
            "help": "Learning rate (paper: {0.003,0.002,0.001,0.0001})",
            "short": "lr",
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for optimizer", "short": "wd"}
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size (paper=64)", "short": "bs"}
    )


@register_trainer("Mamba4KT")
class Mamba4KTTrainer(BaseTrainer):
    """Mamba4KT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.Mamba4KT.Mamba4KT_data import Mamba4KTModelData
        from model.Mamba4KT.Mamba4KT_model import Mamba4KT

        train_dataset, val_dataset, test_dataset = Mamba4KTModelData(
            data_src
        ).prepare_data(rc)

        logger.info("Initializing Mamba4KT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)
        if n_pid > 0:
            logger.info(f"Mamba4KT: Using Rasch embeddings with {n_pid} questions")
        else:
            logger.warning("Mamba4KT: Problem ID not available, using skill-only model")

        m = rc.model
        model = Mamba4KT(
            num_c=metadata["num_skills"],
            n_pid=n_pid,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            d_state=m.d_state,
            d_conv=m.d_conv,
            expand=m.expand,
            dropout=m.dropout,
            l2=m.l2,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def _build_pid_data(
        self, question: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """构建 Rasch pid 数据，0 保留给填充位置。"""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """Mamba4KT 前向传播（next-item 约定）。

        out[t] 利用历史 0..t 预测 response[t+1]，待预测题目 q_{t+1} 已在模型内部左移注入。
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

        # next-item alignment
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=False
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

        # Windowlate convention: mask=1 only at the last position (target_pos=p) and the
        # target response is zeroed to prevent leakage. Under next-item, the prediction
        # for target p lives at y_hat_full[p-1] (using history 0..p-1 and question q_p),
        # so mask[:, 1:] selects target positions aligned with the preceding prediction.
        target_mask = mask[:, 1:]
        y_hat = torch.masked_select(y_hat_full[:, :-1], target_mask)
        y_label = torch.masked_select(true_labels[:, 1:].float(), target_mask)
        group_ids = torch.masked_select(late_group_id[:, 1:], target_mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失，包含 BCE 损失与 Rasch 正则化损失（论文 Eq. 11）。"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            return bce_loss + outputs["c_reg_loss"]
        return bce_loss
