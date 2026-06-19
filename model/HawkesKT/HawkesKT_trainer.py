from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("HawkesKT")
class HawkesKTModelParams(BaseParamConfig):
    def define_params(self) -> tuple[str, dict]:
        group_name = "HawkesKT Parameters"
        params = {
            "emb_size": {
                "type": int,
                "default": 64,
                "help": "Size of embedding vectors",
            },
            "time_log": {
                "type": float,
                "default": 2.718281828459045,
                "help": "Log base of time intervals",
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
                "help": "Weight decay (L2 regularization) for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size for training",
            },
        }
        return group_name, params


@TRAINERS.register("HawkesKT")
class HawkesKTTrainer(BaseTrainer):
    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.HawkesKT.HawkesKT_data import HawkesKTModelData

        model_data = HawkesKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        from model.HawkesKT.HawkesKT_model import HawkesKT

        logger.info("Initializing HawkesKT model...")
        metadata = data_src.get_metadata()
        model = HawkesKT(
            num_c=metadata["num_skills"],
            num_q=metadata["num_questions"],
            emb_size=args.emb_size,
            time_log=args.time_log,
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
        es_patience = args.es_patience or None
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=args.es_monitor,
                mode=args.es_mode,
                patience=es_patience,
                min_delta=args.es_min_delta,
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
            model_name="HawkesKT",
            dataset_name=args.dataset,
        ).build()

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        skill, problem, time, label, mask = batch_data
        skill = self._move_tensor_to_device(skill)
        problem = self._move_tensor_to_device(problem)
        time = self._move_tensor_to_device(time)
        label = self._move_tensor_to_device(label)
        mask = self._move_tensor_to_device(mask)

        y_hat_full = self.model(skill, problem, time, label)

        # 归一化：y[t] 预测 label[t] → y[t] 预测 label[t+1]
        y_norm = torch.cat(
            [y_hat_full[:, 1:], torch.zeros_like(y_hat_full[:, :1])], dim=1
        )
        y_hat, y_label, _ = self._extract_valid_predictions(y_norm, label, mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }
