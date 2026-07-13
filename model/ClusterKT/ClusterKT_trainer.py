from dataclasses import dataclass, field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["ClusterKTTrainer", "ClusterKTConfig"]


@register_model_config("ClusterKT")
@dataclass
class ClusterKTConfig(ModelConfig):
    """ClusterKT model configuration."""

    d_model: int = field(
        default=256, metadata={"help": "Hidden dimension of the model", "short": "dm"}
    )
    n_blocks: int = field(
        default=4, metadata={"help": "Number of transformer blocks", "short": "nb"}
    )
    n_heads: int = field(
        default=8, metadata={"help": "Number of attention heads", "short": "nh"}
    )
    dropout: float = field(
        default=0.05, metadata={"help": "Dropout probability", "short": "dp"}
    )
    d_ff: int = field(
        default=1024,
        metadata={"help": "Feed-forward network dimension", "short": "df"},
    )
    cluster_size: int = field(
        default=10, metadata={"help": "Number of cluster centers", "short": "cs"}
    )
    final_fc_dim: int = field(
        default=512,
        metadata={"help": "Final fully connected layer dimension", "short": "fc"},
    )
    kq_same: int = field(
        default=1,
        metadata={
            "help": "Whether key and query use same linear transform (1=yes, 0=no)"
        },
    )
    separate_qa: int = field(
        default=0,
        metadata={"help": "Whether to use separate QA embeddings (1=yes, 0=no)"},
    )
    n_st: int = field(
        default=300,
        metadata={"help": "Spent time embedding vocabulary size"},
    )
    n_et: int = field(
        default=1440,
        metadata={"help": "Elapsed time embedding vocabulary size"},
    )
    cluster_loss_weight: float = field(
        default=0.001, metadata={"help": "Weight for cluster regularization loss"}
    )
    epochs: int = field(
        default=100, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "Learning rate for optimizer", "short": "lr"},
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for optimizer", "short": "wd"},
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size for training", "short": "bs"}
    )


@register_trainer("ClusterKT")
class ClusterKTTrainer(BaseTrainer):
    """ClusterKT model trainer."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.ClusterKT.ClusterKT_data import ClusterKTModelData

        model_data = ClusterKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.ClusterKT.ClusterKT_model import ClusterKT

        logger.info("Initializing ClusterKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)
        num_skill_groups = model_data.num_skill_groups

        logger.info(
            f"ClusterKT: n_question(skill_groups)={num_skill_groups}, "
            f"n_pid(questions)={n_pid}"
        )

        m = rc.model
        model = ClusterKT(
            n_question=num_skill_groups,
            n_pid=n_pid,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            kq_same=m.kq_same,
            dropout=m.dropout,
            cluster_size=m.cluster_size,
            final_fc_dim=m.final_fc_dim,
            n_heads=m.n_heads,
            d_ff=m.d_ff,
            n_st=m.n_st,
            n_et=m.n_et,
            separate_qa=bool(m.separate_qa),
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

        self.cluster_loss_weight = m.cluster_loss_weight

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(self, batch_data):
        """ClusterKT forward pass.

        Args:
            batch_data: (skill_group_seq, question_seq, response, mask, lagtime)

        Returns:
            dict with y_hat (logits), y_label, y_predict, y_prob, cluster_loss
        """
        skill_group_seq, question_seq, response, mask, lagtime = batch_data
        skill_group_seq = self._move_tensor_to_device(skill_group_seq)
        question_seq = self._move_tensor_to_device(question_seq)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        lagtime = self._move_tensor_to_device(lagtime)

        # dotime: padding value (no spent_time data)
        dotime = torch.full_like(skill_group_seq, self.model.time + 1)

        # pid_data for Rasch model
        pid_data = question_seq if self.model.n_pid > 0 else None

        y_hat_full, cluster_loss = self.model(
            q_data=skill_group_seq,
            response=response,
            mask=mask,
            pid_data=pid_data,
            dotime=dotime,
            lagtime=lagtime,
        )

        # Extract valid predictions
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
            "cluster_loss": cluster_loss,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """Compute BCE loss + cluster regularization."""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)
        cluster_loss = outputs.get("cluster_loss", 0.0)
        return bce_loss + self.cluster_loss_weight * cluster_loss
