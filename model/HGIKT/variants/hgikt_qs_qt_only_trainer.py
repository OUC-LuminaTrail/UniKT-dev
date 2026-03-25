"""Trainer for HGIKT_QS_QT_Only variant."""

from typing import Any

import torch
from torch_geometric.data import HeteroData

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("HGIKT_QS_QT_Only")
class HGIKTQSQTOnlyModelParams(BaseParamConfig):
    """HGIKT_QS_QT_Only model parameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "HGIKT_QS_QT_Only Parameters"
        params = {
            "hidden_dim": {"type": int, "default": 250, "help": "Hidden layer dimension"},
            "n_hop": {"type": int, "default": 4, "help": "Number of GNN hops"},
            "heads": {"type": int, "default": 1, "help": "Number of attention heads"},
            "lstm_layers": {"type": int, "default": 1, "help": "Number of LSTM layers"},
            "history_neighbour": {"type": int, "default": 5, "help": "History neighbor count"},
            "att_bound": {"type": float, "default": 0.1, "help": "Attention bound"},
            "epochs": {"type": int, "default": 120, "help": "Number of training epochs"},
            "learning_rate": {"type": float, "default": 0.0003, "help": "Learning rate"},
            "lr_decay": {"type": float, "default": None, "help": "LR decay factor"},
            "dropout": {"type": float, "default": 0.25, "help": "Dropout rate"},
            "weight_decay": {"type": float, "default": 0.00001, "help": "Weight decay"},
            "batch_size": {"type": int, "default": 64, "help": "Batch size"},
        }
        return group_name, params


@TRAINERS.register("HGIKT_QS_QT_Only")
class HGIKTQSQTOnlyTrainer(BaseTrainer):
    """Trainer for HGIKT with QS and QT edges only."""

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        # 1. Prepare data
        from model.HGIKT.variants.hgikt_qs_qt_only_data import HGIKTQSQTOnlyData
        model_data = HGIKTQSQTOnlyData(data_src)
        data_dict = model_data.prepare_data(args)

        self.hypergraph = data_dict["skill_hypergraph"]
        self.hetero_graph = data_dict["hetero_graph"]
        self.question_skill_matrix = data_dict["question_skill_matrix"]

        # 2. Initialize model
        from model.HGIKT.variants.hgikt_qs_qt_only import HGIKT_QS_QT_Only
        logger.info("Initializing HGIKT_QS_QT_Only model...")
        model = HGIKT_QS_QT_Only(
            args, data_src.get_metadata(), self.hetero_graph.metadata()
        )

        # 3. Call parent constructor
        super().__init__(model)

        # 4. Optimizer and Loss
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

        early_stopping_cfg = None
        if getattr(args, "es_patience", None) is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=args.es_patience
            )

        self.with_training(epochs=args.epochs, seed=args.seed, device=args.device, checkpoint_path=args.checkpoint_path) \
            .with_data(train_data=data_dict["train_dataset"], val_data=data_dict["val_dataset"], batch_size=args.batch_size) \
            .with_optimization(optimizer=optimizer, loss_fn=loss_fn, lr_scheduler=lr_scheduler, early_stopping=early_stopping_cfg) \
            .with_experiment(exp_manager=exp_manager, hyperparams=args, model_name="HGIKT_QS_QT_Only", dataset_name=getattr(args, "dataset", "")) \
            .build()

        self.hetero_graph = self.hetero_graph.to(self.device_)
        self.hypergraph = self.hypergraph.to(self.device_)
        self.question_skill_matrix = self.question_skill_matrix.to(self.device_)

    def forward_pass(self, batch_data) -> dict[str, torch.Tensor]:
        sequence, response, mask = batch_data
        sequence, response, mask = self._move_tensor_to_device(sequence), self._move_tensor_to_device(response), self._move_tensor_to_device(mask)
        y_hat_full = self.model(sequence, response, mask, self.hetero_graph, self.hypergraph, self.question_skill_matrix)
        
        # HGIKT specific sampling/alignment
        y_hat = y_hat_full[:, :-1]
        target = response[:, 1:].float()
        mask_flat = mask[:, 1:]
        
        return {"loss": self.loss(y_hat[mask_flat], target[mask_flat]), "y_gold": target[mask_flat], "y_pred": torch.sigmoid(y_hat[mask_flat])}
