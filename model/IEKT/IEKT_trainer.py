from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("IEKT")
class IEKTConfig(ModelConfig):
    """IEKT 模型配置

    Args:
        emb_size: Embedding dimension.
        dropout: Dropout probability.
        n_layer: Number of hidden layers in MLP heads.
        cog_levels: Number of cognitive estimation levels (m_t action space).
        acq_levels: Number of knowledge acquisition sensitivity levels (s_t action space).
        lamb: Weight of the reinforcement learning loss.
        gamma: Reward discount factor for policy gradient.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    emb_size: int = field(
        default=64,
        metadata={"optuna": {"type": "int", "low": 64, "high": 256}},
    )
    dropout: float = field(
        default=0.0,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    n_layer: int = field(
        default=1,
        metadata={"optuna": {"type": "int", "low": 1, "high": 3}},
    )
    cog_levels: int = 10
    acq_levels: int = 10
    lamb: int = 40
    gamma: float = 0.93
    epochs: int = 100
    learning_rate: float = field(
        default=1e-3,
        metadata={
            "optuna": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
        },
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4]}},
    )
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("IEKT")
class IEKTTrainer(BaseTrainer):
    """IEKT 模型训练器

    实现 REINFORCE 策略梯度与 BCE 联合损失。
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.IEKT.IEKT_data import IEKTModelData
        from model.IEKT.IEKT_model import IEKT

        model_data = IEKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        metadata = data_src.get_metadata()
        logger.info(
            f"Initializing IEKT model: {metadata['num_questions']} questions, "
            f"{metadata['num_skills']} skills, max_concepts={model_data.max_concepts}"
        )

        m = rc.model
        model = IEKT(
            num_questions=metadata["num_questions"],
            num_skills=metadata["num_skills"],
            emb_size=m.emb_size,
            max_concepts=model_data.max_concepts,
            lamb=m.lamb,
            n_layer=m.n_layer,
            cog_levels=m.cog_levels,
            acq_levels=m.acq_levels,
            dropout=m.dropout,
            gamma=m.gamma,
        )

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        self.lamb = m.lamb
        self.gamma = m.gamma

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(self, batch_data: tuple) -> dict:
        """IEKT 前向传播

        Args:
            batch_data: (sequence, response, mask, skills) 元组

        Returns:
            包含 y_hat / y_label / y_predict 等指标键，以及 ``_rl_*`` 损失细节键的字典
        """
        sequence, response, mask, skills = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        skills = self._move_tensor_to_device(skills)

        out = self.model(sequence, response, mask, skills)
        logits = out["logits"]  # [B, S]; predict_current[t] predicts response[t]
        probs = torch.sigmoid(logits)  # [B, S]

        # Same-position output: probs[t] predicts response[t]; same_position=True applies built-in normalization
        y_hat, y_label, _ = self._extract_valid_predictions(
            probs, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "_rl_logits": logits,
            "_rl_p_actions": out["p_actions"],
            "_rl_emb_actions": out["emb_actions"],
            "_rl_pre_states": out["pre_states"],
            "_rl_states": out["states"],
            "_rl_response": response,
            "_rl_mask": mask,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """REINFORCE 策略梯度 + BCE 联合损失

        总损失 = ``lamb * rl_loss + bce``，其中 ``rl_loss`` 为认知/获取两个策略的
        策略梯度损失均值，``bce`` 为主预测损失。损失仅在有效预测位（拥有历史）上计算。
        """
        logits = outputs["_rl_logits"]  # [B, S]
        response = outputs["_rl_response"]  # [B, S]
        mask = outputs["_rl_mask"]  # [B, S] bool
        p_actions = outputs["_rl_p_actions"]
        emb_actions = outputs["_rl_emb_actions"]
        pre_states = outputs["_rl_pre_states"]
        states = outputs["_rl_states"]

        B, S = logits.shape
        device = logits.device

        # Valid prediction: predict_current[t] is valid iff both t and t-1 are real positions (has history)
        # valid[t] = mask[t] & mask[t-1], with valid[0] = False
        mask_prev = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device), mask[:, :-1]], dim=1
        )
        valid = mask & mask_prev  # [B, S]
        valid_f = valid.float()

        # Reward: whether the prediction matches the ground truth
        pred01 = (logits > 0).float()
        reward = (pred01 == response.float()).float()  # [B, S]
        reward = reward * valid_f

        seq_len = valid_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # [B, 1]
        reward_norm = reward / seq_len

        # Discounted return (advantage) via backward recursion
        advantage = torch.zeros(B, S, device=device)
        advantage[:, -1] = reward_norm[:, -1]
        for t in range(S - 2, -1, -1):
            advantage[:, t] = self.gamma * advantage[:, t + 1] + reward_norm[:, t]
        advantage = (
            advantage.detach()
        )  # REINFORCE: advantage is treated as a constant (no gradient)

        pi_cog = self.model.pi_cog(pre_states)  # [B, S, cog_levels]
        pi_sens = self.model.pi_sens(states)  # [B, S, acq_levels]

        eps = 1e-30
        log_pi_cog = torch.log(
            pi_cog.gather(-1, p_actions.unsqueeze(-1)).squeeze(-1) + eps
        )  # [B, S]
        log_pi_sens = torch.log(
            pi_sens.gather(-1, emb_actions.unsqueeze(-1)).squeeze(-1) + eps
        )

        # Policy loss only at valid positions
        loss_cog = -(log_pi_cog * advantage) * valid_f  # [B, S]
        loss_sens = -(log_pi_sens * advantage) * valid_f

        num_valid = valid_f.sum().clamp(min=1.0)
        rl_loss = (loss_cog.sum() + loss_sens.sum()) / num_valid

        logits_flat = torch.masked_select(logits, valid)
        labels_flat = torch.masked_select(response.float(), valid)
        bce = self.loss(logits_flat, labels_flat)  # BCEWithLogitsLoss(reduction='mean')

        return self.lamb * rl_loss + bce
