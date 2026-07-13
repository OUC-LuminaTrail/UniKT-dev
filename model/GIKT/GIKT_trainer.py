"""GIKT 模型训练器。"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["GIKTTrainer", "GIKTConfig"]


@register_model_config("GIKT")
class GIKTConfig(ModelConfig):
    """GIKT model configuration."""

    n_hop: int = field(
        default=3,
        metadata={
            "help": "Number of GNN aggregation hops",
            "short": "nh",
            "optuna": {"type": "int", "low": 1, "high": 5},
        },
    )
    skill_neighbor_num: int = field(
        default=4,
        metadata={
            "help": "Number of skill neighbors sampled per hop",
            "optuna": {"type": "int", "low": 2, "high": 10},
        },
    )
    question_neighbor_num: int = field(
        default=4,
        metadata={
            "help": "Number of question neighbors sampled per hop",
            "optuna": {"type": "int", "low": 2, "high": 10},
        },
    )
    hist_neighbor_num: int = field(
        default=3,
        metadata={
            "help": "Number of historical neighbor samples M",
            "short": "hn",
            "optuna": {"type": "int", "low": 3, "high": 10},
        },
    )
    next_neighbor_num: int = field(
        default=4,
        metadata={
            "help": "Number of next-question neighbor samples N",
            "short": "nn",
            "optuna": {"type": "int", "low": 2, "high": 10},
        },
    )
    att_bound: float = field(
        default=0.7,
        metadata={
            "help": "Similarity threshold",
            "optuna": {"type": "float", "low": 0.0, "high": 0.8},
        },
    )
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
        default=100,
        metadata={
            "help": "Embedding dimension",
            "short": "ed",
            "optuna": {"type": "int", "low": 64, "high": 256, "log": True},
        },
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
        default=1e-3,
        metadata={
            "help": "Learning rate",
            "short": "lr",
            "optuna": {"type": "float", "low": 0.0001, "high": 0.01, "log": True},
        },
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )
    weight_decay: float = field(
        default=1e-8,
        metadata={
            "help": "Weight decay",
            "short": "wd",
            "optuna": {"type": "float", "low": 0.000001, "high": 0.001, "log": True},
        },
    )
    batch_size: int = field(
        default=32,
        metadata={
            "help": "Batch size",
            "short": "bs",
            "optuna": {"type": "categorical", "choices": [32, 64, 128, 256]},
        },
    )


@register_trainer("GIKT")
class GIKTTrainer(BaseTrainer):
    """GIKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.GIKT.GIKT_data import GIKTModelData
        from model.GIKT.GIKT_model import GIKT

        model_data = GIKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.graph_data,
            self.num_skills,
            self.num_questions,
            train_collate_fn,
            val_collate_fn,
        ) = model_data.prepare_data(rc)

        logger.info("Initializing GIKT model...")
        m = rc.model
        model = GIKT(
            data_metadata=data_src.get_metadata(),
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
        # build_components runs before BaseTrainer.build() sets self.device_, so
        # derive the device from rc the same way build() does.
        dev = rc.general.device
        device = (
            torch.device(dev)
            if dev
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.graph_data = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.graph_data.items()
        }
        self.graph_data["feature_embedding"] = model.feature_embedding.weight

        logger.info(
            f"GIKT Trainer: {self.num_skills} skills, {self.num_questions} questions"
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
        """返回 dict：y_hat/y_label/y_predict/y_score/y_prob。模型输出 [B, S-1]（next-item）。"""
        sequence = self._move_tensor_to_device(batch_data["sequence"])
        response = self._move_tensor_to_device(batch_data["response"])
        mask = self._move_tensor_to_device(batch_data["mask"])
        skills = self._move_tensor_to_device(batch_data["skills"])
        hist_neighbor_index = self._move_tensor_to_device(
            batch_data["hist_neighbor_index"]
        )

        y_hat_full = self._pad_to_full_sequence(
            self.model(
                user_sequence=sequence,
                user_response=response,
                user_mask=mask,
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
