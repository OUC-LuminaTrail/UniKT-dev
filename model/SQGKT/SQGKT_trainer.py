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
    """SQGKT model configuration."""

    n_hop: int = field(
        default=3,
        metadata={"help": "Number of GNN aggregation hops", "short": "nh"},
    )
    skill_neighbor_num: int = field(
        default=4, metadata={"help": "Number of skill neighbors sampled per hop"}
    )
    question_neighbor_num: int = field(
        default=4, metadata={"help": "Number of question neighbors sampled per hop"}
    )
    user_neighbor_num: int = field(
        default=5,
        metadata={
            "help": "Sampled students per question in the student-question graph"
        },
    )
    hist_neighbor_num: int = field(
        default=3,
        metadata={"help": "Number of historical neighbor samples M", "short": "hn"},
    )
    next_neighbor_num: int = field(
        default=4,
        metadata={"help": "Number of next-question neighbor samples N", "short": "nn"},
    )
    att_bound: float = field(default=0.7, metadata={"help": "Similarity threshold"})
    aggregator: str = field(
        default="sum", metadata={"help": "Aggregator type: sum or concat"}
    )
    variant: str = field(
        default="hsei",
        metadata={
            "help": "History sampling variant: hssi/hsei (same skill) or ssei/dkt (similarity)"
        },
    )
    sim_emb: str = field(
        default="question_emb",
        metadata={"help": "Similarity embedding: skill_emb/question_emb/feature"},
    )
    embedding_dim: int = field(
        default=100, metadata={"help": "Embedding dimension", "short": "ed"}
    )
    hidden_neurons: list[int] = field(
        default_factory=lambda: [200, 100],
        metadata={
            "help": "Hidden sizes for each LSTM layer; last layer must equal embedding_dim",
            "nargs": "+",
        },
    )
    dropout_probs: list[float] = field(
        default_factory=lambda: [0.2, 0.2, 0.0],
        metadata={
            "help": "Dropout probabilities for [LSTM, GNN, eval]",
            "nargs": "+",
        },
    )
    epochs: int = field(
        default=100, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-3, metadata={"help": "Learning rate", "short": "lr"}
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )
    weight_decay: float = field(
        default=1e-8, metadata={"help": "Weight decay", "short": "wd"}
    )
    batch_size: int = field(default=32, metadata={"help": "Batch size", "short": "bs"})


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
