"""DKT2 模型训练器。"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["DKT2Trainer", "DKT2ModelParams"]


@register_model_params("DKT2")
class DKT2ModelParams(BaseParamConfig):
    """DKT2 参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        return "DKT2 Parameters", {
            # xLSTM 结构
            "factor": {
                "type": float,
                "default": 1.3,
                "help": "mLSTM/sLSTM feed-forward up-projection factor (default: 1.3)",
            },
            "num_blocks": {
                "type": int,
                "default": 1,
                "help": "Number of xLSTM blocks (default: 1)",
            },
            "num_heads": {
                "type": int,
                "default": 2,
                "help": "Number of attention heads per block (default: 2)",
            },
            "slstm_at": {
                "type": int,
                "default": [0],
                "nargs": "+",
                "help": "Block indices using sLSTM (rest use mLSTM) (default: [0])",
            },
            "slstm_backend": {
                "type": str,
                "default": "cuda",
                "help": "sLSTM backend: cuda or vanilla (default: cuda, auto-fallback to vanilla)",
            },
            "conv1d_kernel_size": {
                "type": int,
                "default": 4,
                "help": "Conv1d kernel size in xLSTM blocks (default: 4)",
            },
            "qkv_proj_blocksize": {
                "type": int,
                "default": 4,
                "help": "Block size of the mLSTM qkv projection (default: 4)",
            },
            "embedding_size": {
                "type": int,
                "default": 64,
                "short": "ed",
                "help": "Embedding / hidden dimension (default: 64)",
            },
            "dropout": {
                "type": float,
                "default": 0.2,
                "help": "Dropout probability (default: 0.2)",
            },
            "length": {
                "type": int,
                "default": 1,
                "help": "Prediction horizon (next-item when 1) (default: 1)",
            },
            # 训练
            "epochs": {
                "type": int,
                "default": 200,
                "short": "ep",
                "help": "Number of training epochs (default: 200)",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-3,
                "short": "lr",
                "help": "Learning rate (default: 1e-3)",
            },
            "weight_decay": {
                "type": float,
                "default": 0.0,
                "short": "wd",
                "help": "Weight decay (default: 0.0)",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size (default: 128)",
            },
        }


@register_trainer("DKT2")
class DKT2Trainer(BaseTrainer):
    """DKT2 训练器。"""

    def __init__(self, args: Any = None, data_src: Any = None, exp_manager: Any = None):
        from model.DKT2.DKT2_data import DKT2ModelData
        from model.DKT2.DKT2_model import DKT2

        model_data = DKT2ModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_skills,
            num_questions,
        ) = model_data.prepare_data(args)
        max_seq_len = data_src.get_metadata("max_seq_len")

        logger.info("Initializing DKT2 model...")
        model = DKT2(
            num_skills=num_skills,
            num_questions=num_questions,
            batch_size=args.batch_size,
            seq_len=max_seq_len,
            factor=args.factor,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            slstm_at=args.slstm_at,
            conv1d_kernel_size=args.conv1d_kernel_size,
            qkv_proj_blocksize=args.qkv_proj_blocksize,
            embedding_size=args.embedding_size,
            dropout=args.dropout,
            slstm_backend=args.slstm_backend,
            length=args.length,
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        early_stopping_cfg = EarlyStoppingConfig(
            monitor=args.es_monitor,
            mode=args.es_mode,
            patience=args.es_patience,
            min_delta=args.es_min_delta,
        )

        super().__init__(model)
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
            loss_fn=torch.nn.BCELoss(),
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="DKT2",
            dataset_name=args.dataset,
        ).build()

        logger.info(
            f"DKT2 Trainer: {num_skills} concepts (incl. padding), "
            f"{num_questions} questions, sLSTM backend = {model.slstm_backend}"
        )

    def forward_pass(self, batch_data: dict) -> dict[str, torch.Tensor]:
        """next-item 前向传播。

        模型输出已过 sigmoid 的概率，长度 S-1，output[t] 预测 response[t+1]。
        用 _pad_to_full_sequence 补一列后走基类的 next-item 对齐。
        """
        questions = self._move_tensor_to_device(batch_data["questions"])
        responses = self._move_tensor_to_device(batch_data["responses"])
        masks = self._move_tensor_to_device(batch_data["masks"])
        skills = self._move_tensor_to_device(batch_data["skills"])

        output, _ = self.model(questions, skills, responses, masks)  # [B, S-1] probs

        y_hat_full = self._pad_to_full_sequence(output)
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, responses, masks
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
        }
