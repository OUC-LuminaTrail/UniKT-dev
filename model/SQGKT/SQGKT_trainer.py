"""SQGKT 模型训练器。"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["SQGKTTrainer", "SQGKTConfig"]


@register_model_config("SQGKT")
class SQGKTConfig(ModelConfig):
    """SQGKT model configuration.

    Args:
        n_hop: Number of GNN aggregation hops.
        skill_neighbor_num: Number of skill neighbors sampled per hop.
        question_neighbor_num: Number of question neighbors sampled per hop.
        user_neighbor_num: Sampled students per question in the student-question graph.
        hist_neighbor_num: Number of historical neighbor samples M.
        next_neighbor_num: Number of next-question neighbor samples N.
        att_bound: Similarity threshold.
        aggregator: Aggregator type: sum or concat.
        variant: History sampling variant: hssi/hsei (same skill) or ssei/dkt (similarity).
        sim_emb: Similarity embedding: skill_emb/question_emb/feature.
        embedding_dim: Embedding dimension.
        hidden_neurons: Hidden sizes for each LSTM layer; last layer must equal embedding_dim.
        dropout_probs: Dropout probabilities for [LSTM, GNN, eval].
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay.
        batch_size: Batch size.
    """

    n_hop: int = field(
        default=3,
        metadata={"optuna": {"type": "int", "low": 1, "high": 5}},
    )
    skill_neighbor_num: int = field(
        default=4,
        metadata={"optuna": {"type": "int", "low": 2, "high": 10}},
    )
    question_neighbor_num: int = 4
    user_neighbor_num: int = 5
    hist_neighbor_num: int = 3
    next_neighbor_num: int = 4
    att_bound: float = field(
        default=0.7,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.8}},
    )
    aggregator: str = "sum"
    variant: str = "hsei"
    sim_emb: str = "question_emb"
    embedding_dim: int = 100
    hidden_neurons: list[int] = field(default_factory=lambda: [200, 100])
    dropout_probs: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.0])
    epochs: int = 100
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=1e-8,
        metadata={"optuna": {"type": "float", "low": 1e-8, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=32,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128, 256]}},
    )


@register_trainer("SQGKT")
class SQGKTTrainer(BaseTrainer):
    """SQGKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.SQGKT.SQGKT_data import SQGKTModelData
        from model.SQGKT.SQGKT_model import SQGKT

        model_data = SQGKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.graph_data,
            self.num_skills,
            self.num_questions,
            self.num_users,
            train_collate_fn,
            val_collate_fn,
        ) = model_data.prepare_data(rc)

        logger.info("Initializing SQGKT model...")
        metadata = dict(data_src.get_metadata())
        metadata["num_users"] = self.num_users
        m = rc.model
        model = SQGKT(
            data_metadata=metadata,
            embedding_dim=m.embedding_dim,
            hidden_neurons=list(m.hidden_neurons),
            dropout_probs=list(m.dropout_probs),
            n_hop=m.n_hop,
            skill_neighbor_num=m.skill_neighbor_num,
            question_neighbor_num=m.question_neighbor_num,
            hist_neighbor_num=m.hist_neighbor_num,
            next_neighbor_num=m.next_neighbor_num,
            att_bound=m.att_bound,
            aggregator=m.aggregator,
            variant=m.variant,
            sim_emb=m.sim_emb,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        lr_scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=m.lr_decay)
            if m.lr_decay
            else None
        )

        # Move static graph data to device and bind the shared embedding table.
        device = (
            torch.device(rc.general.device) if rc.general.device else self._try_gpu()
        )
        self.graph_data = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.graph_data.items()
        }
        self.graph_data["feature_embedding"] = model.feature_embedding.weight

        logger.info(
            f"SQGKT Trainer: {self.num_skills} skills, {self.num_questions} questions"
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            collate_fn=train_collate_fn,
            val_collate_fn=val_collate_fn,
        )

    def forward_pass(self, batch_data):
        sequence = self._move_tensor_to_device(batch_data["sequence"])
        response = self._move_tensor_to_device(batch_data["response"])
        mask = self._move_tensor_to_device(batch_data["mask"])
        user_id = self._move_tensor_to_device(batch_data["user_id"])
        skills = self._move_tensor_to_device(batch_data["skills"])
        hist_neighbor_index = self._move_tensor_to_device(
            batch_data["hist_neighbor_index"]
        )

        y_hat_full = self._pad_to_full_sequence(
            self.model(
                user_sequence=sequence,
                user_response=response,
                user_mask=mask,
                user_ids=user_id,
                skills=skills,
                graph_data=self.graph_data,
                hist_neighbor_index=hist_neighbor_index,
            )
        )
        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.0),
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
        }
