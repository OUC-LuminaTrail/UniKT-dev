"""DGEKT 模型训练器。

监督损失为三路（concept / transition / ensemble）BCE 之和——按每生有效步
取均值、batch 内求和；另有在线蒸馏项：ensemble 路温度软化后与两分支路
logits 的全分布 L1 距离，权重 ``kd_alpha``。评估用三路概率平均。
"""

from dataclasses import field

import torch
import torch.nn.functional as F
from torch import nn

from utils.config import ModelConfig
from utils.core import register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents


@register_model_config("DGEKT")
class DGEKTConfig(ModelConfig):
    """DGEKT model configuration.

    Args:
        emb_dim: Interaction embedding dimension (must be even).
        hidden_dim: GRU hidden dimension.
        num_layers: Number of GRU layers.
        kd_alpha: Weight of the online knowledge distillation loss.
        kd_temperature: Temperature softening the logits for distillation.
        learning_rate: Learning rate.
        weight_decay: Adam weight decay.
        epochs: Max training epochs (early stopping applies).
        batch_size: Batch size.
    """

    emb_dim: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256]}},
    )
    hidden_dim: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128]}},
    )
    num_layers: int = 1
    kd_alpha: float = field(
        default=1.25e-6,
        metadata={"optuna": {"type": "float", "low": 1e-8, "high": 1e-5, "log": True}},
    )
    kd_temperature: float = field(
        default=0.5,
        metadata={"optuna": {"type": "categorical", "choices": [0.25, 0.5, 1.0]}},
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    weight_decay: float = 0.0
    epochs: int = 20
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("DGEKT")
class DGEKTTrainer(BaseTrainer):
    """DGEKT 训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DGEKT.DGEKT_data import DGEKTModelData
        from model.DGEKT.DGEKT_model import DGEKT

        train_dataset, val_dataset, test_dataset, info = DGEKTModelData(
            data_src
        ).prepare_data(rc)
        m = rc.model
        model = DGEKT(
            num_questions=info["num_questions"],
            emb_dim=m.emb_dim,
            hidden_dim=m.hidden_dim,
            num_layers=m.num_layers,
            hyper_g=info["hyper_g"],
            adj_out=info["adj_out"],
            adj_in=info["adj_in"],
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """前向传播：三路 logits 按 next-item 对齐，取三路概率平均作预测。

        Returns:
            含 ``y_hat`` / ``y_label`` / ``y_predict`` / ``y_score`` / ``y_prob``
            与辅助损失 ``_sup_loss`` / ``_kd_loss`` 的字典。
        """
        question, response, mask = batch_data
        question = self._move_tensor_to_device(question)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        logit_c, logit_t, logit_e = self.model(question, response, mask)  # [B, S, Q]

        # Keep only the next question's logit per step; sigmoid is elementwise,
        # so applying it after the gather is equivalent and much cheaper.
        next_idx = question[:, 1:].unsqueeze(-1)  # [B, S-1, 1]
        p_c = torch.sigmoid(torch.gather(logit_c[:, :-1], -1, next_idx).squeeze(-1))
        p_t = torch.sigmoid(torch.gather(logit_t[:, :-1], -1, next_idx).squeeze(-1))
        p_e = torch.sigmoid(torch.gather(logit_e[:, :-1], -1, next_idx).squeeze(-1))

        y_hat, y_label, valid_mask = self._extract_valid_predictions(
            self._pad_to_full_sequence((p_c + p_t + p_e) / 3.0), response, mask
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        sup_loss = kd_loss = None
        if self.model.training:
            # Supervised loss: per-student mean over valid steps, summed over batch.
            labels = response.float()[:, 1:]
            counts = valid_mask.sum(dim=1).clamp(min=1)
            sup_loss = sum(
                (
                    (
                        F.binary_cross_entropy(
                            p.clamp(1e-7, 1.0 - 1e-7), labels, reduction="none"
                        )
                        * valid_mask
                    ).sum(dim=1)
                    / counts
                ).sum()
                for p in (p_c, p_t, p_e)
            )

            # Online distillation: L1 between the softened ensemble distribution
            # and both branch distributions.
            t = self.run_config.model.kd_temperature
            soft_e = torch.sigmoid(logit_e / t)
            kd_loss = (soft_e - torch.sigmoid(logit_c / t)).abs().sum() + (
                soft_e - torch.sigmoid(logit_t / t)
            ).abs().sum()

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "_sup_loss": sup_loss,
            "_kd_loss": kd_loss,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        return (
            outputs["_sup_loss"] + self.run_config.model.kd_alpha * outputs["_kd_loss"]
        )
