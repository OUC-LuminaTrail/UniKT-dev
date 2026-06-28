"""KQN baseline trainer."""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("KQN")
class KQNModelParams(BaseParamConfig):
    """KQN model and optimization parameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "KQN Parameters"
        params = {
            "n_hidden": {
                "type": int,
                "default": 128,
                "help": "Shared dimensionality of knowledge-state and skill vectors",
            },
            "n_rnn_hidden": {
                "type": int,
                "default": 128,
                "help": "Hidden size of the RNN knowledge encoder",
            },
            "n_mlp_hidden": {
                "type": int,
                "default": 128,
                "help": "Hidden size of the skill encoder MLP",
            },
            "n_rnn_layers": {
                "type": int,
                "default": 1,
                "help": "Number of RNN layers in the knowledge encoder",
            },
            "rnn_type": {
                "type": str,
                "default": "lstm",
                "choices": ["lstm", "gru"],
                "help": "RNN cell type for the knowledge encoder",
            },
            "dropout": {
                "type": float,
                "default": 0.4,
                "help": "Dropout probability applied to encoded knowledge states",
            },
            "epochs": {
                "type": int,
                "default": 150,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-3,
                "short": "lr",
                "help": "Learning rate for Adam optimizer",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch",
            },
            "weight_decay": {
                "type": float,
                "default": 0.0,
                "short": "wd",
                "help": "Weight decay for Adam optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size for training",
            },
            "max_grad_norm": {
                "type": float,
                "default": None,
                "help": "Optional max gradient norm for clipping",
            },
        }
        return group_name, params


@TRAINERS.register("KQN")
class KQNTrainer(BaseTrainer):
    """Trainer for the KC-level KQN baseline."""

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.KQN.KQN_data import KQNModelData
        from model.KQN.KQN_model import KQN

        model_data = KQNModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        metadata = data_src.get_metadata()
        logger.info(
            f"Initializing KQN model with {metadata['num_skills']} skills, "
            f"rnn_type={args.rnn_type}"
        )
        model = KQN(
            num_skills=metadata["num_skills"],
            n_hidden=args.n_hidden,
            n_rnn_hidden=args.n_rnn_hidden,
            n_mlp_hidden=args.n_mlp_hidden,
            n_rnn_layers=args.n_rnn_layers,
            rnn_type=args.rnn_type,
            dropout=args.dropout,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        super().__init__(model)

        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 0.0),
            )

        self.with_training(
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
        ).with_data(
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            batch_size=args.batch_size,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            max_clip_grad_norm=args.max_grad_norm,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="KQN",
            dataset_name=getattr(args, "dataset", ""),
            skip_test=getattr(args, "skip_test", False),
        ).build()

    @staticmethod
    def _build_shifted_inputs(
        concept: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build KQN inputs for predicting the next response.

        Input batch shapes:
            question: [B, S], retained by data/trainer but unused by KQN.
            concept: [B, S], 0-based KC ids.
            response: [B, S], binary labels.
            mask: [B, S], true for valid training/validation sequence positions.

        Shift semantics:
            current_concept/current_response are positions ``0..S-2`` and form
            Interaction input ``x_t``.
            next_concept/next_response are positions ``1..S-1`` and represent
            the query KC and BCE target for ``c_{t+1}``.
            target_mask is true only when both current and next positions are
            real sequence events.
        """
        current_concept = concept[:, :-1]
        current_response = response[:, :-1]
        next_concept = concept[:, 1:]
        next_response = response[:, 1:]
        target_mask = mask[:, :-1] & mask[:, 1:]
        return (
            current_concept,
            current_response,
            next_concept,
            next_response,
            target_mask,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """Run a training/validation batch.

        Batch format is ``(concept, response, mask, question)``. ``question`` is
        kept for explicit batch semantics but is not used by the KQN baseline.
        """
        concept, response, mask, _question = batch_data
        concept = self._move_tensor_to_device(concept)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # _build_shifted_inputs 产出模型所需的 current/next concept 对（模型吐
        # [B, S-1] 的 next-item 预测）；提取时 pad 到 [B, S] 并喂原始 response/mask。
        current_concept, current_response, next_concept, _, _ = (
            self._build_shifted_inputs(concept, response, mask)
        )

        y_hat_full = self._pad_to_full_sequence(
            self.model(current_concept, current_response, next_concept)
        )
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full,
            response,
            mask,
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

    def test_forward_pass(self, batch_data: tuple[torch.Tensor, ...]):
        """Run a windowlate test batch.

        Windowlate batch format:
            ``(concept, response, mask, late_group_id, true_labels, question)``.

        In windowlate data, history positions are context only and usually have
        ``mask=False``; target positions have ``mask=True``. Therefore test
        selection uses the shifted target-side mask ``mask[:, 1:]`` instead of
        the training mask ``mask[:, :-1] & mask[:, 1:]``.
        """
        concept, response, mask, late_group_id, true_labels, _question = batch_data
        concept = self._move_tensor_to_device(concept)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        current_concept = concept[:, :-1]
        current_response = response[:, :-1]
        next_concept = concept[:, 1:]

        y_hat_full = self.model(current_concept, current_response, next_concept)

        target_mask = mask[:, 1:].bool()
        y_hat = torch.masked_select(y_hat_full, target_mask)
        y_label = torch.masked_select(true_labels[:, 1:], target_mask).float()
        group_ids = torch.masked_select(late_group_id[:, 1:], target_mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }
