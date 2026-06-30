from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["LBKTTrainer", "LBKTModelParams"]


@register_model_params("LBKT")
class LBKTModelParams(BaseParamConfig):
    """LBKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "LBKT Parameters"
        params = {
            "dim_tp": {
                "type": int,
                "default": 128,
                "help": "Topic (question) embedding dimension (default: 128)",
            },
            "dim_hidden": {
                "type": int,
                "default": 50,
                "help": "Response embedding dimension (default: 50)",
            },
            "num_units": {
                "type": int,
                "default": 128,
                "help": "Hidden dimension (default: 128)",
            },
            "dropout": {
                "type": float,
                "default": 0.2,
                "help": "Dropout rate (default: 0.2)",
            },
            "q_gamma": {
                "type": float,
                "default": 0.1,
                "help": "Q-matrix smoothing factor (default: 0.1)",
            },
            "epochs": {
                "type": int,
                "default": 100,
                "short": "ep",
                "help": "Number of training epochs (default: 100)",
            },
            "learning_rate": {
                "type": float,
                "default": 0.001,
                "short": "lr",
                "help": "Learning rate for optimizer (default: 0.001)",
            },
            "lr_decay_step": {
                "type": int,
                "default": 1,
                "help": "Learning rate decay step size (default: 1)",
            },
            "lr_decay_rate": {
                "type": float,
                "default": 0.5,
                "help": "Learning rate decay factor (default: 0.5)",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-6,
                "help": "Weight decay for optimizer (default: 1e-6)",
            },
            "batch_size": {
                "type": int,
                "default": 16,
                "short": "bs",
                "help": "Batch size for training (default: 16)",
            },
        }
        return group_name, params


@register_trainer("LBKT")
class LBKTTrainer(BaseTrainer):
    """LBKT 模型训练器。"""

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        from model.LBKT.LBKT_data import LBKTModelData
        from model.LBKT.LBKT_model import LBKT

        model_data = LBKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.q_matrix,
        ) = model_data.prepare_data(args)

        # 初始化模型
        logger.info("Initializing LBKT model...")
        model = LBKT(args, data_src.get_metadata())

        # 优化器和损失函数
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8,
            betas=(0.1, 0.999),
        )

        # 学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate
        )

        # 初始化基类训练器
        super().__init__(model)

        # 构建早停配置
        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 0.0),
            )

        # 配置训练器
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
            model_name="LBKT",
            dataset_name=args.dataset,
        ).build()

        # 将 Q-matrix 移动到设备
        self.q_matrix = self.q_matrix.to(self.device_)

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """LBKT 前向传播。

        Args:
            batch_data: 包含 (sequence, response, mask, time_factor, attempt_factor, hint_factor) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, time_factor, attempt_factor, hint_factor = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        time_factor = self._move_tensor_to_device(time_factor)
        attempt_factor = self._move_tensor_to_device(attempt_factor)
        hint_factor = self._move_tensor_to_device(hint_factor)

        # 模型前向传播
        preds = self.model(
            sequence,
            response,
            time_factor,
            attempt_factor,
            hint_factor,
            self.q_matrix,
        )

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
