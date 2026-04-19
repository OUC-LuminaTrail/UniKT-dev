from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["ClusterKTTrainer", "ClusterKTModelParams"]


@register_model_params("ClusterKT")
class ClusterKTModelParams(BaseParamConfig):
    """ClusterKT model parameter configuration."""

    def define_params(self):
        group_name = "ClusterKT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 256,
                "short": "dm",
                "help": "Hidden dimension of the model (default: 256)",
            },
            "n_blocks": {
                "type": int,
                "default": 4,
                "short": "nb",
                "help": "Number of transformer blocks (default: 4)",
            },
            "n_heads": {
                "type": int,
                "default": 8,
                "short": "nh",
                "help": "Number of attention heads (default: 8)",
            },
            "dropout": {
                "type": float,
                "default": 0.05,
                "short": "dp",
                "help": "Dropout probability (default: 0.05)",
            },
            "d_ff": {
                "type": int,
                "default": 1024,
                "short": "df",
                "help": "Feed-forward network dimension (default: 1024)",
            },
            "cluster_size": {
                "type": int,
                "default": 10,
                "short": "cs",
                "help": "Number of cluster centers (default: 10)",
            },
            "final_fc_dim": {
                "type": int,
                "default": 512,
                "short": "fc",
                "help": "Final fully connected layer dimension (default: 512)",
            },
            "kq_same": {
                "type": int,
                "default": 1,
                "help": "Whether key and query use same linear transform (1=yes, 0=no)",
            },
            "separate_qa": {
                "type": int,
                "default": 0,
                "help": "Whether to use separate QA embeddings (1=yes, 0=no)",
            },
            "n_st": {
                "type": int,
                "default": 300,
                "help": "Spent time embedding vocabulary size (default: 300)",
            },
            "n_et": {
                "type": int,
                "default": 1440,
                "help": "Elapsed time embedding vocabulary size (default: 1440)",
            },
            "cluster_loss_weight": {
                "type": float,
                "default": 0.001,
                "help": "Weight for cluster regularization loss (default: 0.001)",
            },
            "epochs": {
                "type": int,
                "default": 100,
                "short": "ep",
                "help": "Number of training epochs (default: 100)",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-3,
                "short": "lr",
                "help": "Learning rate for optimizer (default: 0.001)",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch (default: None)",
            },
            "weight_decay": {
                "type": float,
                "default": 0.0,
                "short": "wd",
                "help": "Weight decay for optimizer (default: 0.0)",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size for training (default: 64)",
            },
        }
        return group_name, params


@TRAINERS.register("ClusterKT")
class ClusterKTTrainer(BaseTrainer):
    """ClusterKT model trainer."""

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        from model.ClusterKT.ClusterKT_data import ClusterKTModelData

        model_data = ClusterKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        from model.ClusterKT.ClusterKT_model import ClusterKT

        logger.info("Initializing ClusterKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)
        num_skill_groups = model_data.num_skill_groups

        logger.info(
            f"ClusterKT: n_question(skill_groups)={num_skill_groups}, "
            f"n_pid(questions)={n_pid}"
        )

        model = ClusterKT(
            n_question=num_skill_groups,
            n_pid=n_pid,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            kq_same=args.kq_same,
            dropout=args.dropout,
            cluster_size=args.cluster_size,
            final_fc_dim=args.final_fc_dim,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            n_st=args.n_st,
            n_et=args.n_et,
            separate_qa=bool(args.separate_qa),
        )

        loss_fn = torch.nn.BCEWithLogitsLoss()
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

        self.cluster_loss_weight = args.cluster_loss_weight

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
            model_name="ClusterKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

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
            y_hat_full, response, mask, skip_first=False
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
