"""MCSKT 模型训练器"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("MCSKT")
class MCSKTConfig(ModelConfig):
    """MCSKT model configuration."""

    d_model: int = field(
        default=256,
        metadata={
            "help": "Hidden dimension (256 with n_blocks=5 reproduces the paper's "
            "~5.6M param count)"
        },
    )
    n_blocks: int = field(
        default=5, metadata={"help": "Number of Mamba blocks per encoder (Q/K)"}
    )
    num_heads: int = field(
        default=8, metadata={"help": "Number of dynamic k-sparse attention heads"}
    )
    d_state: int = field(
        default=16, metadata={"help": "SSM latent state dimension in Mamba"}
    )
    d_conv: int = field(
        default=4, metadata={"help": "Conv1D kernel width in Mamba block"}
    )
    expand: int = field(default=2, metadata={"help": "Mamba internal expansion factor"})
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout probability (applied to embeddings, Mamba blocks, "
            "prediction head)"
        },
    )
    l2: float = field(
        default=1e-5,
        metadata={"help": "L2 reg coefficient for Rasch difficulty parameter"},
    )
    num_rgap: int = field(
        default=100,
        metadata={"help": "Number of review-gap (repeated time gap) buckets"},
    )
    num_sgap: int = field(
        default=100, metadata={"help": "Number of sequence-gap buckets"}
    )
    num_pcount: int = field(
        default=15, metadata={"help": "Number of past-trial-count buckets"}
    )
    delta1: float = field(
        default=0.25,
        metadata={
            "help": "Lower bound of dynamic sparsity interval [d1, d2] (paper: 1/4)"
        },
    )
    delta2: float = field(
        default=0.667,
        metadata={
            "help": "Upper bound of dynamic sparsity interval [d1, d2] (paper: 2/3)"
        },
    )
    epochs: int = field(
        default=200,
        metadata={"help": "Number of training epochs (paper: 200)", "short": "ep"},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={
            "help": "Learning rate (paper: 1e-5; raised for stable Adam)",
            "short": "lr",
        },
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay for optimizer", "short": "wd"},
    )
    max_clip_grad_norm: float = field(
        default=1.0,
        metadata={
            "help": "Max gradient norm for clipping (None to disable)",
            "short": "clip",
        },
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size (paper: 64)", "short": "bs"}
    )


@register_trainer("MCSKT")
class MCSKTTrainer(BaseTrainer):
    """MCSKT 模型训练器。

    预测语义（same_position）：
        - preds[:, t] 使用历史 0..t-1 与当前题目 x_t 预测 response[t]
        - trainer 用 ``same_position=True`` 由内置函数归一化为 next-item 视图
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.MCSKT.MCSKT_data import MCSKTModelData
        from model.MCSKT.MCSKT_model import MCSKT

        model_data = MCSKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        logger.info("Initializing MCSKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)

        if n_pid > 0:
            logger.info(f"MCSKT: Using Rasch embeddings with {n_pid} questions")
        else:
            logger.warning("MCSKT: Problem ID not available, using skill-only model")

        m = rc.model
        model = MCSKT(
            num_c=metadata["num_skills"],
            n_pid=n_pid,
            num_rgap=m.num_rgap,
            num_sgap=m.num_sgap,
            num_pcount=m.num_pcount,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            num_heads=m.num_heads,
            d_state=m.d_state,
            d_conv=m.d_conv,
            expand=m.expand,
            dropout=m.dropout,
            l2=m.l2,
            delta1=m.delta1,
            delta2=m.delta2,
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
            max_clip_grad_norm=m.max_clip_grad_norm,
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
        """MCSKT 前向传播（same_position 约定）。

        batch_data: (sequence, response, mask, question, rgap, sgap, pcount)
        """
        sequence, response, mask, question, rgap, sgap, pcount = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)
        rgap = self._move_tensor_to_device(rgap)
        sgap = self._move_tensor_to_device(sgap)
        pcount = self._move_tensor_to_device(pcount)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else question

        y_hat_full, c_reg_loss = self.model(
            sequence, response, mask, pid_data, rgap, sgap, pcount
        )

        # same_position: out[t] predicts response[t], normalized to a next-item view
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

    def test_forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """测试前向传播，支持 windowlateauc_mean 评估。

        batch_data: (sequence, response, mask, late_group_id, true_labels,
                     question, rgap, sgap, pcount)  9-元组
        遗忘特征由窗口内时间戳实时计算（非零填充）。
        """
        (
            sequence,
            response,
            mask,
            late_group_id,
            true_labels,
            question,
            rgap,
            sgap,
            pcount,
        ) = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)
        rgap = self._move_tensor_to_device(rgap)
        sgap = self._move_tensor_to_device(sgap)
        pcount = self._move_tensor_to_device(pcount)

        use_pid = self.model.n_pid > 0
        # In windowlate, mask is 1 only at target positions; attention needs all valid
        # positions as the key mask, so build valid_mask from late_group_id >= 0 and
        # reserve the target mask for selecting evaluation predictions.
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else question

        y_hat_full, _ = self.model(
            sequence, response, valid_mask, pid_data, rgap, sgap, pcount
        )

        # Windowlate evaluates only at target positions (mask=1); under same_position,
        # out[target] predicts response[target] using history 0..target-1.
        y_hat = torch.masked_select(y_hat_full, mask)
        y_label = torch.masked_select(true_labels.float(), mask)
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
        """计算损失：BCE + Rasch 正则化（论文 Eq.9 / Eq.11 风格）。"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            return bce_loss + outputs["c_reg_loss"]
        return bce_loss
