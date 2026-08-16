from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["GRKTTrainer", "GRKTConfig"]


@register_model_config("GRKT")
class GRKTConfig(ModelConfig):
    """GRKT model configuration.

    Args:
        d_hidden: Dimension of embedding and hidden states.
        k_hidden: Dimension of hidden knowledge mastery.
        pos_mode: Positive projection mode: sigmoid|softplus|relu|softmax|none.
        k_hop: Hops of graph operation.
        thresh: Threshold for relevance/prerequisite graph sparsity.
        tau: Gumbel softmax temperature.
        alpha: Time interval scaling factor.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        batch_size: Batch size.
    """

    d_hidden: int = field(
        default=128,
        metadata={"optuna": {"type": "int", "low": 64, "high": 256}},
    )
    k_hidden: int = field(
        default=16,
        metadata={"optuna": {"type": "int", "low": 8, "high": 32}},
    )
    pos_mode: str = "softmax"
    k_hop: int = field(
        default=1,
        metadata={"optuna": {"type": "int", "low": 1, "high": 3}},
    )
    thresh: float = field(
        default=0.6,
        metadata={"optuna": {"type": "float", "low": 0.3, "high": 0.8}},
    )
    tau: float = 0.2
    alpha: float = 0.01
    epochs: int = 200
    learning_rate: float = field(
        default=0.001,
        metadata={
            "optuna": {"type": "float", "low": 0.0001, "high": 0.01, "log": True}
        },
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4]}},
    )
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("GRKT")
class GRKTTrainer(BaseTrainer):
    def build_components(self, rc, data_src) -> RuntimeComponents:
        # 1. Prepare data
        from model.GRKT.GRKT_data import GRKTModelData

        model_data = GRKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_questions,
            num_skills,
            max_k,
            rel_map,
            pre_map,
        ) = model_data.prepare_data(rc)

        logger.info(
            f"GRKT data prepared: n_questions={num_questions}, "
            f"n_skills={num_skills}, max_skills_per_q={max_k}"
        )

        # 2. Build metadata dict for model construction
        metadata = data_src.get_metadata()
        metadata["num_questions"] = num_questions
        metadata["num_skills"] = num_skills

        # 3. Initialize model
        from model.GRKT.GRKT_model import GRKT

        logger.info("Initializing GRKT model...")
        m = rc.model
        model = GRKT(
            metadata,
            rel_map,
            pre_map,
            d_hidden=m.d_hidden,
            k_hidden=m.k_hidden,
            k_hop=m.k_hop,
            tau=m.tau,
            alpha=m.alpha,
            pos_mode=m.pos_mode,
            thresh=m.thresh,
        )

        # 4. Create optimizer and loss function
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

    def forward_pass(self, batch_data):
        questions, knows, responses, times, mask = batch_data
        questions = self._move_tensor_to_device(questions)
        knows = self._move_tensor_to_device(knows)
        responses = self._move_tensor_to_device(responses)
        times = self._move_tensor_to_device(times)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # Forward: scores[i] predicts response[i], shape [B, S]
        scores = self.model(questions, knows, responses, times)

        # Drop first prediction (no prior history), evaluate on 1..S-1
        y_hat = scores[:, 1:]
        y_label = responses.float()[:, 1:]
        valid_mask = mask[:, 1:]

        y_hat = torch.masked_select(y_hat, valid_mask)
        y_label = torch.masked_select(y_label, valid_mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }
