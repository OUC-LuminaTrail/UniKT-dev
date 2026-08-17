"""MCKT trainer."""

from dataclasses import field
from typing import Literal

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("MCKT")
class MCKTConfig(ModelConfig):
    """MCKT model configuration.

    Args:
        d_model: Embedding and hidden dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        temperature: Contrastive learning temperature.
        sim_threshold: Question cosine-similarity threshold for topic state attention.
        cl_batch_size: Chunk size for question/interaction contrastive loss.
        cl_exp_mode: Use 'source' for the released MCKT code path's double-exp question/interaction CL, or 'paper' for exp(cos/tau).
        pro_loss_weight: Question-level contrastive loss weight.
        react_loss_weight: Interaction-level contrastive loss weight.
        state_loss_weight: Knowledge-state contrastive loss weight.
        pos_strategy: How to build pos_matrix from question-KC relations.
        pos_include_self: Include diagonal self-positive pairs in pos_matrix.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay for optimizer.
        batch_size: Batch size for training.
        max_grad_norm: Max gradient norm for clipping.
    """

    # powers of two so d_model % n_heads == 0 for every combination
    d_model: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    n_heads: int = field(
        default=8,
        metadata={"optuna": {"type": "categorical", "choices": [4, 8, 16]}},
    )
    dropout: float = field(
        default=0.1,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    temperature: float = field(
        default=0.8,
        metadata={"optuna": {"type": "float", "low": 0.1, "high": 1.0}},
    )
    sim_threshold: float = field(
        default=0.8,
        metadata={"optuna": {"type": "float", "low": 0.5, "high": 0.95}},
    )
    cl_batch_size: int = 10000
    cl_exp_mode: Literal["paper", "source"] = "source"
    pro_loss_weight: float = 1.0
    react_loss_weight: float = 1.0
    state_loss_weight: float = 0.0001
    pos_strategy: Literal["same_kc_set", "shared_kc"] = "shared_kc"
    pos_include_self: bool = True
    epochs: int = 70
    learning_rate: float = field(
        default=0.002,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=1e-5,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=80,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 80, 128]}},
    )
    max_grad_norm: float = 15.0


@register_trainer("MCKT")
class MCKTTrainer(BaseTrainer):
    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.MCKT.MCKT_data import MCKTModelData
        from model.MCKT.MCKT_model import MCKT

        model_data = MCKTModelData(data_src)
        train_dataset, val_dataset, test_dataset, pos_matrix = model_data.prepare_data(
            rc
        )

        metadata = data_src.get_metadata()
        logger.info(
            f"Initializing MCKT model: {metadata['num_questions']} questions, "
            f"{metadata['num_skills']} skills"
        )
        m = rc.model
        model = MCKT(
            num_questions=metadata["num_questions"],
            num_skills=metadata["num_skills"],
            d_model=m.d_model,
            dropout=m.dropout,
            n_heads=m.n_heads,
            temperature=m.temperature,
            sim_threshold=m.sim_threshold,
            cl_batch_size=m.cl_batch_size,
            cl_exp_mode=m.cl_exp_mode,
            pro_loss_weight=m.pro_loss_weight,
            react_loss_weight=m.react_loss_weight,
            state_loss_weight=m.state_loss_weight,
            pos_matrix=pos_matrix,
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
            max_clip_grad_norm=m.max_grad_norm,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def _build_mckt_inputs(
        self,
        question: torch.Tensor,
        response: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        last_problem = question[:, :-1]
        last_ans = response[:, :-1]
        next_problem = question[:, 1:]
        next_ans = response[:, 1:]
        return last_problem, last_ans, next_problem, next_ans

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        question, response, mask = batch_data
        question = self._move_tensor_to_device(question)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        last_problem, last_ans, next_problem, next_ans = self._build_mckt_inputs(
            question, response
        )
        y_hat_full, state_loss, pro_loss, react_loss = self.model(
            last_problem, last_ans, next_problem, next_ans
        )

        # y_hat_full is [B, S-1] next-item (out[t] predicts response[t+1]); pad to
        # [B, S] and reuse the unified extraction contract (adjacent-pair mask).
        y_hat_full = self._pad_to_full_sequence(y_hat_full)
        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "_state_loss": state_loss,
            "_pro_loss": pro_loss,
            "_react_loss": react_loss,
        }

    def _compute_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Total loss = BCE + training-only contrastive auxiliary terms.

        Eval logging goes through the base ``_compute_eval_loss`` (pure BCE);
        the model forward returns ``None`` aux terms in eval mode.
        """
        return (
            self.loss(outputs["y_hat"], outputs["y_label"])
            + outputs["_state_loss"]
            + outputs["_pro_loss"]
            + outputs["_react_loss"]
        )
