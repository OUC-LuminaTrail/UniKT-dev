"""DGMKT 模型训练器"""

from dataclasses import field

import torch
from torch import nn

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("DGMKT")
class DGMKTConfig(ModelConfig):
    """DGMKT model configuration.

    Args:
        d_model: Hidden dimension of the Mamba backbones and HGNN/GCN branches.
        n_layer: Number of Mamba blocks per backbone (H and D branches).
        lr_decay_step: StepLR interval (epochs) between learning-rate decays.
        lr_decay_rate: StepLR gamma (multiplicative decay factor).
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
    """

    d_model: int = field(
        default=512,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256, 512]}},
    )
    n_layer: int = field(
        default=4,
        metadata={"optuna": {"type": "int", "low": 2, "high": 6}},
    )
    lr_decay_step: int = 50
    lr_decay_rate: float = 0.5
    epochs: int = 200
    batch_size: int = field(
        default=24,
        metadata={"optuna": {"type": "categorical", "choices": [16, 24, 32, 64]}},
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    # linear range: log sampling requires low > 0, default is 0.0
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 1e-3}},
    )


@register_trainer("DGMKT")
class DGMKTTrainer(BaseTrainer):
    """DGMKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DGMKT.DGMKT_data import DGMKTModelData
        from model.DGMKT.DGMKT_model import DGMKT

        (
            train_dataset,
            val_dataset,
            test_dataset,
            H,
            dv,
            de,
            num_users,
            seq_len,
        ) = DGMKTModelData(data_src).prepare_data(rc)

        num_skills = data_src.get_metadata()["num_skills"]
        logger.info("Initializing DGMKT model...")
        m = rc.model
        model = DGMKT(
            num_c_raw=num_skills,
            num_users=num_users,
            max_seq_len=seq_len,
            H=H,
            dv=dv,
            de=de,
            d_model=m.d_model,
            n_layer=m.n_layer,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=m.lr_decay_step, gamma=m.lr_decay_rate
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            lr_scheduler=scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            max_clip_grad_norm=1.0,
        )

    def forward_pass(self, batch_data: tuple) -> dict:
        """DGMKT 前向传播。

        模型返回三路 raw logits（h / d / ensemble），out[t] 预测 response[t+1]；
        指标与 eval loss 使用三路 sigmoid 的均值。
        """
        sequence, response, mask, user_id = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        user_id = self._move_tensor_to_device(user_id)

        device = self.device_
        skill = torch.where(
            mask, sequence, torch.tensor(self.model.num_c_raw, device=device)
        )
        answer = torch.where(mask, response, torch.tensor(2, device=device))
        student = user_id.unsqueeze(1).expand_as(sequence)

        logit_h, logit_d, logit_e = self.model(student, skill, answer)  # [B, S-1]
        eps = 1e-7
        p_h = torch.sigmoid(logit_h).clamp(eps, 1 - eps)
        p_d = torch.sigmoid(logit_d).clamp(eps, 1 - eps)
        p_e = torch.sigmoid(logit_e).clamp(eps, 1 - eps)
        p_mean = (p_h + p_d + p_e) / 3.0

        y_hat, y_label, valid_mask = self._extract_valid_predictions(
            self._pad_to_full_sequence(p_mean), response, mask, same_position=False
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "p_h": p_h,
            "p_d": p_d,
            "p_e": p_e,
            "valid_mask": valid_mask,
        }
        return result

    def test_forward_pass(self, batch_data: tuple) -> dict:
        """windowlate 测试前向"""
        sequence, response, mask, late_group_id, true_labels, _, user_id = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        student = self._move_tensor_to_device(user_id)

        device = self.device_
        seq_len = sequence.shape[1]
        target_pos = mask.long().argmax(dim=1)  # [B]
        pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]
        pad = pos > target_pos.unsqueeze(1)  # [B, S] pad 区

        num_c_raw = self.model.num_c_raw
        skill = torch.where(pad, torch.tensor(num_c_raw, device=device), sequence)
        answer = torch.where(pad, torch.tensor(2, device=device), response)

        logit_h, logit_d, logit_e = self.model(student, skill, answer)  # [B, S-1]
        eps = 1e-7
        p = (
            torch.sigmoid(logit_h).clamp(eps, 1 - eps)
            + torch.sigmoid(logit_d).clamp(eps, 1 - eps)
            + torch.sigmoid(logit_e).clamp(eps, 1 - eps)
        ) / 3.0

        target_sel = mask[:, 1:].bool()
        y_hat = torch.masked_select(p, target_sel)
        y_label = torch.masked_select(true_labels[:, 1:].float(), target_sel)
        group_ids = torch.masked_select(late_group_id[:, 1:], target_sel)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """三路 BCE 之和 + KD 一致性项"""
        y_label = outputs["y_label"]
        valid_mask = outputs["valid_mask"]
        y_h = torch.masked_select(outputs["p_h"], valid_mask)
        y_d = torch.masked_select(outputs["p_d"], valid_mask)
        y_e = torch.masked_select(outputs["p_e"], valid_mask)
        kd = (outputs["p_e"] - outputs["p_d"]).abs() + (
            outputs["p_e"] - outputs["p_h"]
        ).abs()
        kd_loss = kd[valid_mask].sum() / valid_mask.sum().clamp(min=1)
        return (
            self.loss(y_h, y_label)
            + self.loss(y_d, y_label)
            + self.loss(y_e, y_label)
            + kd_loss
        )
