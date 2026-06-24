from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("IEKT")
class IEKTModelParams(BaseParamConfig):
    """IEKT 模型参数配置"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "IEKT Parameters"
        params = {
            "emb_size": {
                "type": int,
                "default": 64,
                "help": "Embedding dimension",
            },
            "dropout": {
                "type": float,
                "default": 0.0,
                "help": "Dropout probability",
            },
            "n_layer": {
                "type": int,
                "default": 1,
                "help": "Number of hidden layers in MLP heads",
            },
            "cog_levels": {
                "type": int,
                "default": 10,
                "help": "Number of cognitive estimation levels (m_t action space)",
            },
            "acq_levels": {
                "type": int,
                "default": 10,
                "help": "Number of knowledge acquisition sensitivity levels (s_t action space)",
            },
            "lamb": {
                "type": int,
                "default": 40,
                "help": "Weight of the reinforcement learning loss",
            },
            "gamma": {
                "type": float,
                "default": 0.93,
                "help": "Reward discount factor for policy gradient",
            },
            "epochs": {
                "type": int,
                "default": 100,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-3,
                "short": "lr",
                "help": "Learning rate for optimizer",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch",
            },
            "weight_decay": {
                "type": float,
                "default": 0,
                "short": "wd",
                "help": "Weight decay (L2 regularization) for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size for training",
            },
        }
        return group_name, params


@TRAINERS.register("IEKT")
class IEKTTrainer(BaseTrainer):
    """IEKT 模型训练器

    实现 REINFORCE 策略梯度与 BCE 联合损失。
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.IEKT.IEKT_data import IEKTModelData
        from model.IEKT.IEKT_model import IEKT

        # 准备数据
        model_data = IEKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        # 初始化模型
        metadata = data_src.get_metadata()
        logger.info(
            f"Initializing IEKT model: {metadata['num_questions']} questions, "
            f"{metadata['num_skills']} skills, max_concepts={model_data.max_concepts}"
        )

        model = IEKT(
            num_questions=metadata["num_questions"],
            num_skills=metadata["num_skills"],
            emb_size=args.emb_size,
            max_concepts=model_data.max_concepts,
            lamb=args.lamb,
            n_layer=args.n_layer,
            cog_levels=args.cog_levels,
            acq_levels=args.acq_levels,
            dropout=args.dropout,
            gamma=args.gamma,
        )

        # 损失函数
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        super().__init__(model)

        self.lamb = args.lamb
        self.gamma = args.gamma

        # 构建早停配置
        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 0.0),
            )

        # 配置训练器
        self.with_training(
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
        ).with_data(
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            batch_size=args.batch_size,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="IEKT",
            dataset_name=args.dataset,
        ).build()

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

        # 模型前向传播
        out = self.model(sequence, response, mask, skills)
        logits = out["logits"]  # [B, S]，predict_current[t] 预测 response[t]
        probs = torch.sigmoid(logits)  # [B, S]

        # 左移：y_hat_full[t] = probs[t+1]，预测 response[t+1]
        B, S = probs.shape
        dummy = torch.zeros(B, 1, device=probs.device)
        y_hat_full = torch.cat([probs[:, 1:], dummy], dim=1)

        # 提取有效预测（跳过首位置，对齐 next 预测）
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            # RL 相关中间结果
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

        # 有效预测位：predict_current[t] 有效当且仅当 t 与 t-1 均为真实位置（有历史）
        # valid[t] = mask[t] & mask[t-1]，且 valid[0] = False
        mask_prev = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device), mask[:, :-1]], dim=1
        )
        valid = mask & mask_prev  # [B, S]
        valid_f = valid.float()

        # 奖励：预测是否等于真值
        pred01 = (logits > 0).float()
        reward = (pred01 == response.float()).float()  # [B, S]
        reward = reward * valid_f

        # 按序列有效长度归一化
        seq_len = valid_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # [B, 1]
        reward_norm = reward / seq_len

        # 折扣回报（advantage），反向递归
        advantage = torch.zeros(B, S, device=device)
        advantage[:, -1] = reward_norm[:, -1]
        for t in range(S - 2, -1, -1):
            advantage[:, t] = self.gamma * advantage[:, t + 1] + reward_norm[:, t]
        advantage = advantage.detach()  # REINFORCE：advantage 视为常数

        # 策略 log-probs
        pi_cog = self.model.pi_cog(pre_states)  # [B, S, cog_levels]
        pi_sens = self.model.pi_sens(states)  # [B, S, acq_levels]

        eps = 1e-30
        log_pi_cog = torch.log(
            pi_cog.gather(-1, p_actions.unsqueeze(-1)).squeeze(-1) + eps
        )  # [B, S]
        log_pi_sens = torch.log(
            pi_sens.gather(-1, emb_actions.unsqueeze(-1)).squeeze(-1) + eps
        )

        # 仅在有效位计算策略损失
        loss_cog = -(log_pi_cog * advantage) * valid_f  # [B, S]
        loss_sens = -(log_pi_sens * advantage) * valid_f

        num_valid = valid_f.sum().clamp(min=1.0)
        rl_loss = (loss_cog.sum() + loss_sens.sum()) / num_valid

        # BCE 主损失
        logits_flat = torch.masked_select(logits, valid)
        labels_flat = torch.masked_select(response.float(), valid)
        bce = self.loss(logits_flat, labels_flat)  # BCEWithLogitsLoss(reduction='mean')

        return self.lamb * rl_loss + bce
