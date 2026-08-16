"""ATKT 模型训练器"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("ATKT")
class ATKTConfig(ModelConfig):
    """ATKT model configuration.

    Args:
        skill_emb_dim: Skill embedding dimension.
        answer_emb_dim: Answer embedding dimension.
        hidden_dim: LSTM hidden dimension.
        attention_dim: Attention intermediate dimension.
        adversarial_beta: Weight of the adversarial loss.
        adversarial_epsilon: Perturbation magnitude on the interaction embedding.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay_step: StepLR step size.
        lr_decay_rate: StepLR gamma.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    skill_emb_dim: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [128, 256]}},
    )
    answer_emb_dim: int = field(
        default=96,
        metadata={"optuna": {"type": "categorical", "choices": [64, 96, 128]}},
    )
    hidden_dim: int = field(
        default=80,
        metadata={"optuna": {"type": "categorical", "choices": [64, 80, 128]}},
    )
    attention_dim: int = field(
        default=80,
        metadata={"optuna": {"type": "categorical", "choices": [64, 80, 128]}},
    )
    adversarial_beta: float = 0.2
    adversarial_epsilon: float = 10.0
    epochs: int = 150
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    lr_decay_step: int = 50
    lr_decay_rate: float = 0.5
    # categorical so the default 0.0 stays inside the space
    weight_decay: float = field(
        default=0.0,
        metadata={
            "optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4, 1e-3]}
        },
    )
    batch_size: int = field(
        default=24,
        metadata={"optuna": {"type": "categorical", "choices": [16, 24, 32, 64]}},
    )


def _l2_normalize_adv(d: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalization of the adversarial gradient.

    Args:
        d: Gradient of the clean loss w.r.t. the interaction embedding
            [B, S, emb_dim].

    Returns:
        Unit-norm perturbation direction, same shape as ``d``.
    """
    return d / (d.norm(dim=(1, 2), keepdim=True) + 1e-16)


@register_trainer("ATKT")
class ATKTTrainer(BaseTrainer):
    """ATKT 模型训练器

    负责初始化ATKT模型、优化器和训练数据，并实现前向传播与对抗训练逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.ATKT.ATKT_data import ATKTModelData
        from model.ATKT.ATKT_model import ATKT

        train_dataset, val_dataset, test_dataset = ATKTModelData(data_src).prepare_data(
            rc
        )

        metadata = data_src.get_metadata()
        m = rc.model
        logger.info("Initializing ATKT model...")
        model = ATKT(
            num_c=metadata["num_skills"],
            skill_emb_dim=m.skill_emb_dim,
            answer_emb_dim=m.answer_emb_dim,
            hidden_dim=m.hidden_dim,
            attention_dim=m.attention_dim,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=m.lr_decay_step, gamma=m.lr_decay_rate
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
    ) -> dict[str, torch.Tensor]:
        """ATKT 前向传播（训练时附带对抗分支）。

        ATKT预测语义：
        - y_hat[:, t] 基于 0..t-1 的交互预测 response[t]（同位对齐）

        对抗训练（仅梯度开启时，与原始实现一致）：
        1. 对 clean loss 关于交互嵌入求梯度并 L2 归一化
        2. 以 epsilon 倍归一化方向作为扰动做第二次前向
        3. adv_loss 由 _compute_loss 以 beta 加权合并

        Args:
            batch_data: 包含 (sequence, response, mask) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典（训练时额外含 adv_loss）
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full, features = self.model(sequence, response)  # [B, S]

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

        if torch.is_grad_enabled():
            clean_loss = self.loss(y_hat, y_label)
            result["adv_loss"] = self._compute_adversarial_loss(
                sequence, response, mask, features, clean_loss, y_label
            )

        return result

    def _compute_adversarial_loss(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        features: torch.Tensor,
        clean_loss: torch.Tensor,
        y_label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the adversarial loss on perturbed interaction embeddings."""
        epsilon = self.run_config.model.adversarial_epsilon

        features_grad = torch.autograd.grad(clean_loss, features, retain_graph=True)[0]
        p_adv = epsilon * _l2_normalize_adv(features_grad)

        y_hat_adv_full, _ = self.model(sequence, response, perturbation=p_adv)
        y_hat_adv, _, _ = self._extract_valid_predictions(
            y_hat_adv_full, response, mask, same_position=True
        )

        return self.loss(y_hat_adv, y_label)

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估。

        batch_data: (sequence, response, mask, late_group_id, true_labels, question)
        """
        sequence, response, mask, late_group_id, true_labels, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        y_hat_full, _ = self.model(sequence, response)  # [B, S]

        # Same-position alignment: y_hat[:, t] predicts response[t]
        y_hat_aligned = y_hat_full[:, 1:]
        true_labels_aligned = true_labels[:, 1:]
        mask_aligned = mask[:, 1:]
        group_id_aligned = late_group_id[:, 1:]

        y_hat = torch.masked_select(y_hat_aligned, mask_aligned)
        y_label = torch.masked_select(true_labels_aligned, mask_aligned).float()
        group_ids = torch.masked_select(group_id_aligned, mask_aligned)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失：BCE损失加对抗损失（训练时）"""
        loss = self.loss(outputs["y_hat"], outputs["y_label"])

        if "adv_loss" in outputs:
            beta = self.run_config.model.adversarial_beta
            loss = loss + beta * outputs["adv_loss"]

        return loss
