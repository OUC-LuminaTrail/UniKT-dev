"""MCKT trainer."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("MCKT")
class MCKTConfig(ModelConfig):
    """MCKT model configuration."""

    d_model: int = field(
        default=128, metadata={"help": "Embedding and hidden dimension"}
    )
    n_heads: int = field(default=8, metadata={"help": "Number of attention heads"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability"})
    temperature: float = field(
        default=0.8, metadata={"help": "Contrastive learning temperature"}
    )
    sim_threshold: float = field(
        default=0.8,
        metadata={
            "help": "Question cosine-similarity threshold for topic state attention"
        },
    )
    cl_batch_size: int = field(
        default=10000,
        metadata={"help": "Chunk size for question/interaction contrastive loss"},
    )
    cl_exp_mode: str = field(
        default="source",
        metadata={
            "choices": ["paper", "source"],
            "help": (
                "Use 'source' for the released MCKT code path's double-exp "
                "question/interaction CL, or 'paper' for exp(cos/tau)"
            ),
        },
    )
    pro_loss_weight: float = field(
        default=1.0, metadata={"help": "Question-level contrastive loss weight"}
    )
    react_loss_weight: float = field(
        default=1.0, metadata={"help": "Interaction-level contrastive loss weight"}
    )
    state_loss_weight: float = field(
        default=0.0001, metadata={"help": "Knowledge-state contrastive loss weight"}
    )
    pos_strategy: str = field(
        default="shared_kc",
        metadata={
            "choices": ["same_kc_set", "shared_kc"],
            "help": "How to build pos_matrix from question-KC relations",
        },
    )
    pos_include_self: bool = field(
        default=True,
        metadata={"help": "Include diagonal self-positive pairs in pos_matrix"},
    )
    epochs: int = field(
        default=70, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=0.002, metadata={"help": "Learning rate for optimizer", "short": "lr"}
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )
    weight_decay: float = field(
        default=1e-5, metadata={"help": "Weight decay for optimizer", "short": "wd"}
    )
    batch_size: int = field(
        default=80, metadata={"help": "Batch size for training", "short": "bs"}
    )
    max_grad_norm: float = field(
        default=15.0, metadata={"help": "Max gradient norm for clipping"}
    )


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
        bce_loss = self.loss(outputs["y_hat"], outputs["y_label"])
        if not self.model.training:
            return bce_loss
        return (
            bce_loss
            + outputs["_state_loss"]
            + outputs["_pro_loss"]
            + outputs["_react_loss"]
        )
