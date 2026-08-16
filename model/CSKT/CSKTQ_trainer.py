"""CSKTQ trainer: question-level CSKT for the skill-vs-question ablation."""

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("CSKTQ")
class CSKTQConfig(ModelConfig):
    """CSKTQ configuration.

    Same architecture as CSKT, but the question id is the concept embedding
    unit (``num_c=num_questions``) and Rasch pid is disabled. Hyperparameters
    mirror CSKT so the ablation varies only the modeling granularity.

    Args:
        d_model: Hidden dimension of the model.
        num_blocks: Number of transformer blocks.
        num_attn_heads: Number of attention heads.
        dropout: Dropout probability.
        d_ff: Feed-forward network dimension.
        r: Cone attention radius parameter.
        gamma: Cone attention temperature parameter.
        kq_same: Whether key and query share the linear projection (1=yes, 0=no).
        separate_qa: Whether to use a separate interaction embedding (1=yes, 0=no).
        final_fc_dim: First fully connected layer dimension in output head.
        final_fc_dim2: Second fully connected layer dimension in output head.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    d_model: int = 128
    num_blocks: int = 2
    num_attn_heads: int = 4
    dropout: float = 0.1
    d_ff: int = 256
    r: float = 0.6
    gamma: float = 1.0
    kq_same: int = 1
    separate_qa: int = 0
    final_fc_dim: int = 512
    final_fc_dim2: int = 256
    epochs: int = 100
    learning_rate: float = 1e-4
    lr_decay: float | None = None
    weight_decay: float = 0.0
    batch_size: int = 64


@register_trainer("CSKTQ")
class CSKTQTrainer(BaseTrainer):
    """Question-level CSKT trainer (ablation variant).

    Reuses the CSKT model with question-level concept embeddings and no
    Rasch pid. Test follows the standard K-fold split instead of windowlate.
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.CSKT.CSKT_model import CSKT
        from model.CSKT.CSKTQ_data import CSKTQModelData

        model_data = CSKTQModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        num_questions = data_src.get_metadata("num_questions")
        logger.info(
            f"Initializing CSKTQ (question-level) with {num_questions} questions"
        )

        m = rc.model
        model = CSKT(
            num_c=num_questions,
            max_seq_len=rc.data.max_seq_len,
            n_pid=0,
            d_model=m.d_model,
            num_blocks=m.num_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            num_attn_heads=m.num_attn_heads,
            r=m.r,
            gamma=m.gamma,
            kq_same=bool(m.kq_same),
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            separate_qa=bool(m.separate_qa),
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

        ``sequence`` is the question id and no pid is used. CSKT's forward
        takes no mask arg (mask is applied externally via
        ``_extract_valid_predictions``). same_position=True: ``y_hat[:, t]``
        predicts ``response[t]``.
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(sequence, response, None)

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
