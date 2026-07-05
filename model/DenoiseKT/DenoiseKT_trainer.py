"""DenoiseKT 模型训练器。"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("DenoiseKT")
class DenoiseKTModelParams(BaseParamConfig):
    """DenoiseKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "DenoiseKT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 256,
                "help": "Hidden dimension of the model",
            },
            "n_blocks": {
                "type": int,
                "default": 1,
                "help": "Number of transformer blocks",
            },
            "num_attn_heads": {
                "type": int,
                "default": 8,
                "help": "Number of attention heads",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability for the transformer",
            },
            "dropout1": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability for the GCN",
            },
            "d_ff": {
                "type": int,
                "default": 64,
                "help": "Feed-forward network dimension",
            },
            "final_fc_dim": {
                "type": int,
                "default": 256,
                "help": "First output MLP dimension",
            },
            "final_fc_dim2": {
                "type": int,
                "default": 256,
                "help": "Second output MLP dimension",
            },
            "bf": {
                "type": float,
                "default": 0.9,
                "help": "Distance-decay base for same-concept boost focus",
            },
            "kq_same": {
                "type": int,
                "default": 1,
                "help": "Whether key and query share the linear projection (1=yes, 0=no)",
            },
            "epochs": {
                "type": int,
                "default": 200,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-3,
                "short": "lr",
                "help": "Learning rate for optimizer",
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
                "help": "Weight decay for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size for training",
            },
        }
        return group_name, params


@register_trainer("DenoiseKT")
class DenoiseKTTrainer(BaseTrainer):
    """DenoiseKT 模型训练器。"""

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.DenoiseKT.DenoiseKT_data import DenoiseKTModelData

        model_data = DenoiseKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            question_concepts,
            question_graph,
        ) = model_data.prepare_data(args)

        from model.DenoiseKT.DenoiseKT_model import DenoiseKT

        metadata = data_src.get_metadata()
        num_q = metadata["num_questions"]
        num_c = metadata["num_skills"]
        logger.info("Initializing DenoiseKT model...")

        model = DenoiseKT(
            num_c=num_c,
            num_q=num_q,
            question_concepts=question_concepts,
            question_graph=question_graph,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            dropout1=args.dropout1,
            bf=args.bf,
            d_ff=args.d_ff,
            seq_len=args.max_seq_len,
            kq_same=args.kq_same,
            final_fc_dim=args.final_fc_dim,
            final_fc_dim2=args.final_fc_dim2,
            num_attn_heads=args.num_attn_heads,
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
                min_delta=getattr(args, "es_min_delta", 1e-3),
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
            model_name="DenoiseKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict:
        """前向传播。

        预测语义：
        ``preds[:, t]`` 使用 ``question[:, t]`` 与历史 ``qa[:, :t]`` 预测 ``response[:, t]``。

        batch_data: ``(question, response, mask)``
        """
        question, response, mask = batch_data
        question = self._move_tensor_to_device(question)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        preds = self.model(question, response)  # [B, S]

        y_hat, y_label, _ = self._extract_valid_predictions(
            preds, response, mask, same_position=True
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
