"""DTransformer 模型训练器"""

import random
from dataclasses import field

import torch
import torch.nn.functional as F

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

MIN_SEQ_LEN = 5


@register_model_config("DTransformer")
class DTransformerConfig(ModelConfig):
    """DTransformer 模型配置

    Args:
        d_model: Hidden dimension of the model.
        d_ff: Feed-forward network dimension.
        num_attn_heads: Number of attention heads.
        n_know: Number of learnable knowledge parameters.
        n_blocks: Number of transformer blocks (1-3).
        dropout: Dropout probability.
        separate_qa: Whether to use separate QA embeddings (1=yes, 0=no).
        l2: L2 regularization coefficient for Rasch model.
        shortcut: Use AKT-like shortcut mode (1=yes, 0=no).
        lambda_cl: Contrastive learning loss weight (0 = disabled).
        hard_neg: Use hard negatives for contrastive learning (1=yes, 0=no).
        proj: Use projection layer for contrastive learning (1=yes, 0=no).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        batch_size: Batch size for training.
    """

    # powers of two so d_model % num_attn_heads == 0 for every combination
    d_model: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    d_ff: int = 256
    num_attn_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8, 16]}},
    )
    n_know: int = 16
    n_blocks: int = field(
        default=3,
        metadata={"optuna": {"type": "int", "low": 1, "high": 3}},
    )
    dropout: float = field(
        default=0.3,
        metadata={"optuna": {"type": "float", "low": 0.1, "high": 0.5}},
    )
    separate_qa: int = 0
    l2: float = 1e-3
    shortcut: int = 0
    lambda_cl: float = 0.0
    hard_neg: int = 0
    proj: int = 0
    epochs: int = 150
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True}},
    )
    weight_decay: float = field(
        default=1e-5,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-4, "log": True}},
    )
    batch_size: int = field(
        default=32,
        metadata={"optuna": {"type": "categorical", "choices": [16, 32, 64, 128]}},
    )


@register_trainer("DTransformer")
class DTransformerTrainer(BaseTrainer):
    """DTransformer 模型训练器"""

    def build_components(self, rc, data_src):
        from model.DTransformer.DTransformer_data import DTransformerModelData

        model_data = DTransformerModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.DTransformer.DTransformer_model import DTransformer

        logger.info("Initializing DTransformer model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)

        if n_pid > 0:
            logger.info(
                f"DTransformer: Using Problem ID (Rasch model) with {n_pid} questions"
            )
        else:
            logger.info(
                "DTransformer: Problem ID not available, using skill-only model"
            )

        m = rc.model
        model = DTransformer(
            num_c=metadata["num_skills"],
            n_pid=n_pid,
            d_model=m.d_model,
            d_ff=m.d_ff,
            num_attn_heads=m.num_attn_heads,
            n_know=m.n_know,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            separate_qa=bool(m.separate_qa),
            l2=m.l2,
            shortcut=bool(m.shortcut),
            proj=bool(m.proj),
        )

        self.lambda_cl = m.lambda_cl
        self.hard_neg = bool(m.hard_neg)

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=None,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build Rasch pid data with 0 reserved for padding."""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def _augment_for_cl(self, sequence, response, pid_data, lens):
        """通过交换相邻元素并翻转部分响应进行数据增强"""
        bs = sequence.size(0)
        q_ = sequence.clone()
        s_ = response.clone()
        pid_ = pid_data.clone() if pid_data is not None else None

        for b in range(bs):
            seq_len = int(lens[b].item())
            n_swap = min(seq_len - 1, max(1, int(seq_len * self.model.dropout_rate)))
            idx = random.sample(range(seq_len - 1), n_swap)
            for i in idx:
                q_[b, i], q_[b, i + 1] = q_[b, i + 1].clone(), q_[b, i].clone()
                s_[b, i], s_[b, i + 1] = s_[b, i + 1].clone(), s_[b, i].clone()
                if pid_ is not None:
                    pid_[b, i], pid_[b, i + 1] = (
                        pid_[b, i + 1].clone(),
                        pid_[b, i].clone(),
                    )

            if not self.hard_neg:
                n_flip = min(seq_len, max(1, int(seq_len * self.model.dropout_rate)))
                flip_idx = random.sample(range(seq_len), n_flip)
                for i in flip_idx:
                    s_[b, i] = 1 - s_[b, i]

        return q_, s_, pid_

    def _flip_responses(self, response, lens):
        """翻转部分响应作为硬负样本"""
        s_flip = response.clone()
        bs = response.size(0)
        for b in range(bs):
            seq_len = int(lens[b].item())
            n_flip = min(seq_len, max(1, int(seq_len * self.model.dropout_rate)))
            idx = random.sample(range(seq_len), n_flip)
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]
        return s_flip

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """DTransformer 前向传播

        DTransformer 预测语义:
        - output[t] 使用 sequence[0..t] 和 response[0..t-1]（通过知识状态）
          加上当前问题 embedding 来预测 response[t]
        - output[t] 直接对应 response[t]，需要对齐
        """
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else None

        use_cl = self.lambda_cl > 0 and self.model.training

        if use_cl:
            output, z, q_emb, reg_loss = self.model.predict(
                sequence, response, pid_data
            )
            y_hat_full = torch.sigmoid(output)

            lens = mask.sum(dim=1)
            minlen = lens.min().item()

            if minlen >= MIN_SEQ_LEN:
                q_aug, s_aug, pid_aug = self._augment_for_cl(
                    sequence, response, pid_data, lens
                )
                _, z_aug, _, _ = self.model.predict(q_aug, s_aug, pid_aug)

                cl_loss = self._compute_cl_loss(
                    z, z_aug, lens, minlen, sequence, response, pid_data
                )
            else:
                cl_loss = torch.tensor(0.0, device=sequence.device)
        else:
            y_hat_full, reg_loss = self.model(sequence, response, mask, pid_data)

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
            result["reg_loss"] = reg_loss
        if use_cl:
            result["cl_loss"] = cl_loss

        return result

    def _compute_cl_loss(self, z, z_aug, lens, minlen, sequence, response, pid_data):
        """计算对比学习损失"""
        sim_input = self.model.sim(z[:, :minlen, :], z_aug[:, :minlen, :])

        if self.hard_neg:
            s_flip = self._flip_responses(response, lens)
            _, z_hard, _, _ = self.model.predict(sequence, s_flip, pid_data)
            hard_sim = self.model.sim(z[:, :minlen, :], z_hard[:, :minlen, :])
            sim_input = torch.cat([sim_input, hard_sim], dim=1)

        target = torch.arange(z.size(0))[:, None].to(z.device).expand(-1, minlen)
        cl_loss = F.cross_entropy(sim_input, target)
        return cl_loss * self.lambda_cl

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估"""
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

        y_hat_full, _ = self.model(sequence, response, mask, pid_data)

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
        """计算损失，包含BCE损失、正则化损失和对比学习损失"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        loss = self.loss(y_hat, y_label)

        if "reg_loss" in outputs and self.model.n_pid > 0:
            loss = loss + outputs["reg_loss"]

        if "cl_loss" in outputs:
            loss = loss + outputs["cl_loss"]

        return loss
