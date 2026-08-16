"""SGKT model trainer.

Defines training logic for Session Graph-based Knowledge Tracing model.
"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["SGKTTrainer", "SGKTConfig"]


@register_model_config("SGKT")
class SGKTConfig(ModelConfig):
    """SGKT model configuration.

    Args:
        n_hop: Number of GCN layers for HRG graph (default: 3).
        sg_layers: Number of GatedGraphConv layers for session graph (default: 2).
        hist_neighbor_num: Number of historical neighbors to sample (default: 3).
        next_neighbor_num: Number of next question neighbors to sample (default: 4).
        att_bound: Similarity threshold for historical neighbor sampling (default: 0.7).
        cooc_neighbor_num: Max number of co-occurrence neighbors per question in HRG graph (default: 0).
        skill_neighbor_num: Number of skill neighbors to sample per hop (default: 4).
        question_neighbor_num: Number of question neighbors to sample per hop (default: 4).
        aggregator: Aggregator type: sum or concat (default: sum).
        select_index: Feature indices used for model inputs (default: [0, 1, 2]).
        sim_emb: Embedding type for similarity (default: question_emb).
        embedding_dim: Embedding dimension (default: 100).
        hidden_dim: Hidden layer dimension (default: 100).
        gnn_dropout_keep: Keep probability of the HRG GNN aggregators (default: 0.8).
        epochs: Number of training epochs (default: 100).
        learning_rate: Learning rate for optimizer (default: 0.00025).
        lr_decay: Learning rate decay factor per epoch (default: 0.92).
        weight_decay: Weight decay (L2 regularization) for optimizer (default: 1e-8).
        batch_size: Batch size for training (default: 6).
    """

    n_hop: int = field(
        default=3,
        metadata={"optuna": {"type": "int", "low": 1, "high": 5}},
    )
    sg_layers: int = 2
    hist_neighbor_num: int = 3
    next_neighbor_num: int = 4
    att_bound: float = field(
        default=0.7,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.8}},
    )
    cooc_neighbor_num: int = 0
    skill_neighbor_num: int = field(
        default=4,
        metadata={"optuna": {"type": "int", "low": 2, "high": 10}},
    )
    question_neighbor_num: int = 4
    aggregator: str = "sum"
    select_index: list[int] = field(default_factory=lambda: [0, 1, 2])
    sim_emb: str = "question_emb"
    embedding_dim: int = 100
    hidden_dim: int = 100
    # keep-prob of the GNN aggregators; single effective position of the old list
    gnn_dropout_keep: float = field(
        default=0.8,
        metadata={"optuna": {"type": "float", "low": 0.5, "high": 0.95}},
    )
    epochs: int = 100
    learning_rate: float = field(
        default=0.00025,
        metadata={"optuna": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True}},
    )
    lr_decay: float = 0.92
    weight_decay: float = field(
        default=1e-8,
        metadata={"optuna": {"type": "float", "low": 1e-8, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=6,
        metadata={"optuna": {"type": "categorical", "choices": [4, 6, 8, 16]}},
    )


@register_trainer("SGKT")
class SGKTTrainer(BaseTrainer):
    """SGKT 模型训练器。"""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.SGKT.SGKT_data import SGKTModelData

        model_data = SGKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.hrg_data,
            self.num_skills,
            self.num_questions,
            train_collate_fn,
            val_collate_fn,
        ) = model_data.prepare_data(rc)

        from model.SGKT.SGKT_model import SGKT

        logger.info("Initializing SGKT model...")
        metadata = data_src.get_metadata()
        m = rc.model
        model = SGKT(
            data_metadata=metadata,
            embedding_dim=m.embedding_dim,
            hidden_dim=m.hidden_dim,
            keep_prob_gnn=m.gnn_dropout_keep,
            question_neighbor_num=m.question_neighbor_num,
            skill_neighbor_num=m.skill_neighbor_num,
            n_hop=m.n_hop,
            aggregator=m.aggregator,
            hist_neighbor_num=m.hist_neighbor_num,
            att_bound=m.att_bound,
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

        # Move static graph data to device and bind the shared embedding table.
        device = (
            torch.device(rc.general.device) if rc.general.device else self._try_gpu()
        )
        self.hrg_data = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.hrg_data.items()
        }
        self.hrg_data["feature_embedding"] = model.feature_embedding.weight

        logger.info(
            f"SGKT Trainer initialized with {self.num_skills} skills and {self.num_questions} questions"
        )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            collate_fn=train_collate_fn,
            val_collate_fn=val_collate_fn,
            test_collate_fn=val_collate_fn,
        )

    def forward_pass(self, batch_data):
        """SGKT 前向传播。

        Args:
            batch_data: Dictionary with keys 'sequence', 'response', 'mask', 'hist_neighbor_index'

        Returns:
            Dictionary with 'y_hat', 'y_label', 'y_predict'
        """
        batch_dict = batch_data
        sequence = batch_dict["sequence"]
        response = batch_dict["response"]
        mask = batch_dict["mask"]
        hist_neighbor_index = batch_dict.get("hist_neighbor_index", None)

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        if hist_neighbor_index is not None:
            hist_neighbor_index = self._move_tensor_to_device(hist_neighbor_index)

        # Model forward pass: output [B, S-1], y[:, t] predicts response[:, t+1] (next-item)
        y_hat_full = self._pad_to_full_sequence(
            self.model(
                user_sequence=sequence,
                user_response=response,
                user_mask=mask,
                hrg_data=self.hrg_data,
                hist_neighbor_index=hist_neighbor_index,
            )
        )

        # Extract valid predictions (pad to [B, S], then use built-in next-item alignment)
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full,
            response,
            mask,
        )

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
        }
