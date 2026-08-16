"""SimpleKTQ trainer: question-level SimpleKT for the skill-vs-question ablation."""

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("SimpleKTQ")
class SimpleKTQConfig(ModelConfig):
    """SimpleKTQ configuration.

    Same architecture as SimpleKT, but the question id is the concept embedding
    unit (``num_skills=num_questions``) and Rasch pid is disabled.
    Hyperparameters mirror SimpleKT so the ablation varies only the modeling
    granularity.

    Args:
        d_model: Dimension of the model.
        n_blocks: Number of transformer blocks.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        d_ff: Dimension of feed-forward network.
        kq_same: Whether to share key and query weights (1 for yes, 0 for no).
        separate_qa: Whether to use separate interaction embedding (1 for yes, 0 for no).
        final_fc_dim: First fully connected layer dimension in output.
        final_fc_dim2: Second fully connected layer dimension in output.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    d_model: int = 256
    n_blocks: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    d_ff: int = 256
    kq_same: int = 1
    separate_qa: int = 0
    final_fc_dim: int = 512
    final_fc_dim2: int = 256
    epochs: int = 100
    learning_rate: float = 1e-4
    lr_decay: float | None = None
    weight_decay: float = 0.0
    batch_size: int = 128


@register_trainer("SimpleKTQ")
class SimpleKTQTrainer(BaseTrainer):
    """Question-level SimpleKT trainer (ablation variant).

    Reuses the SimpleKT model with question-level concept embeddings and no
    Rasch pid. Test follows the standard K-fold split instead of windowlate.
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.SimpleKT.SimpleKT_model import SimpleKT
        from model.SimpleKT.SimpleKTQ_data import SimpleKTQModelData

        model_data = SimpleKTQModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        num_questions = data_src.get_metadata("num_questions")
        logger.info(
            f"Initializing SimpleKTQ (question-level) with {num_questions} questions"
        )

        m = rc.model
        model = SimpleKT(
            num_skills=num_questions,
            n_pid=0,
            d_model=m.d_model,
            n_blocks=m.n_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            n_heads=m.n_heads,
            seq_len=rc.data.max_seq_len,
            kq_same=m.kq_same,
            separate_qa=bool(m.separate_qa),
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
        )

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
            loss_fn=torch.nn.BCELoss(),
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Question-level forward pass.

        ``sequence`` is the question id and no pid is used. Output alignment
        matches SimpleKT (same_position=True): ``y_hat[:, t]`` predicts
        ``response[t]``.
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(sequence, response, mask, None)

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }
