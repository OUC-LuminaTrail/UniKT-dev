"""Trainer for HGIKT_SimpleFusion variant."""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("HGIKT_SimpleFusion")
class HGIKTSimpleFusionModelParams(BaseParamConfig):
    """HGIKT_SimpleFusion model parameters - inherits from HGIKT."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "HGIKT_SimpleFusion Parameters"
        params = {
            "hidden_dim": {
                "type": int,
                "default": 250,
                "short": "hd",
                "help": "Hidden layer dimension",
            },
            "n_hop": {
                "type": int,
                "default": 4,
                "short": "nh",
                "help": "Number of GNN hops",
            },
            "heads": {
                "type": int,
                "default": 1,
                "short": "hs",
                "help": "Number of attention heads",
            },
            "lstm_layers": {
                "type": int,
                "default": 1,
                "short": "ll",
                "help": "Number of LSTM layers",
            },
            "history_neighbour": {
                "type": int,
                "default": 5,
                "short": "hn",
                "help": "History neighbor count",
            },
            "att_bound": {
                "type": float,
                "default": 0.1,
                "short": "ab",
                "help": "Attention bound",
            },
            "num_difficulty_clusters": {
                "type": int,
                "default": 5,
                "help": "Number of difficulty clusters for weighted hypergraph",
            },
            "epochs": {
                "type": int,
                "default": 120,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 0.0003,
                "short": "lr",
                "help": "Learning rate for optimizer",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch",
            },
            "dropout": {
                "type": float,
                "default": 0.25,
                "help": "Dropout rate",
            },
            "weight_decay": {
                "type": float,
                "default": 0.00001,
                "short": "wd",
                "help": "Weight decay (L2 regularization) for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size for training",
            },
        }
        return group_name, params


@TRAINERS.register("HGIKT_SimpleFusion")
class HGIKTSimpleFusionTrainer(BaseTrainer):
    """Trainer for HGIKT with simple addition fusion.

    Uses same data preparation as HGIKT, but variant model without MoE.
    """

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        from model.HGIKT.HGIKT_data import HGIKTModelData

        model_data = HGIKTModelData(data_src)
        data_dict = model_data.prepare_data(args)

        train_dataset = data_dict["train_dataset"]
        val_dataset = data_dict["val_dataset"]
        test_dataset = data_dict.get("test_dataset")
        self.hypergraph = data_dict["skill_hypergraph"]
        self.hetero_graph = data_dict["hetero_graph"]
        self.question_skill_matrix = data_dict["question_skill_matrix"]

        from model.HGIKT.variants.hgikt_simple_fusion import HGIKT_SimpleFusion

        logger.info("Initializing HGIKT_SimpleFusion model...")
        model = HGIKT_SimpleFusion(
            args, data_src.get_metadata(), self.hetero_graph.metadata()
        )

        super().__init__(model)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

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
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="HGIKT_SimpleFusion",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

        self.hetero_graph = self.hetero_graph.to(self.device_)
        self.hypergraph = self.hypergraph.to(self.device_)
        self.question_skill_matrix = self.question_skill_matrix.to(self.device_)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward pass matching HGIKT interface."""
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(
            sequence,
            response,
            mask,
            self.hetero_graph,
            self.hypergraph,
            self.question_skill_matrix,
        )

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=True
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
