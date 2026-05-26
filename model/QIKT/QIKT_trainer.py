"""QIKT 模型训练器模块"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("QIKT")
class QIKTModelParams(BaseParamConfig):
    """QIKT 模型参数配置"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "QIKT Parameters"
        params = {
            "emb_size": {
                "type": int,
                "default": 64,
                "help": "Embedding dimension",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability",
            },
            "mlp_layer_num": {
                "type": int,
                "default": 1,
                "help": "Number of MLP layers in prediction heads",
            },
            "output_mode": {
                "type": str,
                "default": "an",
                "help": "Output fusion mode: 'an' (additive normalization) or 'an_irt'",
            },
            "output_q_all_lambda": {
                "type": float,
                "default": 1.0,
                "help": "Output weight for question-all predictions",
            },
            "output_c_all_lambda": {
                "type": float,
                "default": 1.0,
                "help": "Output weight for concept-all predictions",
            },
            "output_c_next_lambda": {
                "type": float,
                "default": 1.0,
                "help": "Output weight for concept-next predictions",
            },
            "loss_q_all_lambda": {
                "type": float,
                "default": 1.0,
                "help": "Loss weight for question-all auxiliary loss",
            },
            "loss_c_all_lambda": {
                "type": float,
                "default": 1.0,
                "help": "Loss weight for concept-all auxiliary loss",
            },
            "loss_c_next_lambda": {
                "type": float,
                "default": 1.0,
                "help": "Loss weight for concept-next auxiliary loss",
            },
            "loss_q_next_lambda": {
                "type": float,
                "default": 0.0,
                "help": "Loss weight for question-next auxiliary loss",
            },
            "epochs": {
                "type": int,
                "default": 100,
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
                "default": 0,
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


@TRAINERS.register("QIKT")
class QIKTTrainer(BaseTrainer):
    """QIKT 模型训练器

    实现双路径预测的融合和多任务损失计算。
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.QIKT.QIKT_data import QIKTModelData
        from model.QIKT.QIKT_model import QIKT

        # 准备数据
        model_data = QIKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        # 初始化模型
        metadata = data_src.get_metadata()
        logger.info(
            f"Initializing QIKT model: {metadata['num_questions']} questions, "
            f"{metadata['num_skills']} skills, max_concepts={model_data.max_concepts}"
        )

        model = QIKT(
            num_questions=metadata["num_questions"],
            num_skills=metadata["num_skills"],
            emb_size=args.emb_size,
            max_concepts=model_data.max_concepts,
            dropout=args.dropout,
            mlp_layer_num=args.mlp_layer_num,
        )

        # 创建优化器和损失函数
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

        # 保存融合配置
        self.output_mode = args.output_mode
        self.output_q_all_lambda = args.output_q_all_lambda
        self.output_c_all_lambda = args.output_c_all_lambda
        self.output_c_next_lambda = args.output_c_next_lambda
        self.loss_q_all_lambda = args.loss_q_all_lambda
        self.loss_c_all_lambda = args.loss_c_all_lambda
        self.loss_c_next_lambda = args.loss_c_next_lambda
        self.loss_q_next_lambda = args.loss_q_next_lambda

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
            model_name="QIKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def _fuse_predictions(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """融合多预测头的结果

        Args:
            outputs: 模型输出的预测字典

        Returns:
            融合后的预测 [B, S]
        """
        y_q_all = outputs["y_question_all"]
        y_c_all = outputs["y_concept_all"]
        y_c_next = outputs["y_concept_next"]

        if self.output_mode == "an_irt":
            eps = 1e-8

            def sigmoid_inverse(x):
                return torch.log(x / (1 - x + eps) + eps)

            y = (
                sigmoid_inverse(y_q_all) * self.output_q_all_lambda
                + sigmoid_inverse(y_c_all) * self.output_c_all_lambda
                + sigmoid_inverse(y_c_next) * self.output_c_next_lambda
            )
            return torch.sigmoid(y)
        else:
            y = (
                y_q_all * self.output_q_all_lambda
                + y_c_all * self.output_c_all_lambda
                + y_c_next * self.output_c_next_lambda
            )
            total_w = (
                self.output_q_all_lambda
                + self.output_c_all_lambda
                + self.output_c_next_lambda
            )
            return y / total_w

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """QIKT 前向传播

        模型在时刻 t 的输出基于 question[0:t] 和 response[0:t-1]
        预测 response[t]，使用 skip_first=True 对齐。

        Args:
            batch_data: (sequence, response, mask, skills) 元组

        Returns:
            包含 y_hat, y_label, y_predict 及辅助预测的字典
        """
        sequence, response, mask, skills = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)
        skills = self._move_tensor_to_device(skills)

        # 模型前向传播
        outputs = self.model(sequence, response, mask, skills)

        # 融合预测
        y_fused = self._fuse_predictions(outputs)

        # 提取有效预测（跳过第一个位置）
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_fused, response, mask, skip_first=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # 提取辅助预测用于多任务损失（复用同一掩码逻辑）
        q_all, _, _ = self._extract_valid_predictions(
            outputs["y_question_all"], response, mask, skip_first=True
        )
        c_all, _, _ = self._extract_valid_predictions(
            outputs["y_concept_all"], response, mask, skip_first=True
        )
        q_next, _, _ = self._extract_valid_predictions(
            outputs["y_question_next"], response, mask, skip_first=True
        )
        c_next, _, _ = self._extract_valid_predictions(
            outputs["y_concept_next"], response, mask, skip_first=True
        )

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "_aux_q_all": q_all,
            "_aux_c_all": c_all,
            "_aux_q_next": q_next,
            "_aux_c_next": c_next,
        }

    def _compute_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """多任务损失计算

        总损失 = 主损失 + 辅助损失的加权和
        """
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        loss_fn = self.loss

        loss_kt = loss_fn(y_hat, y_label)
        loss_q_all = loss_fn(outputs["_aux_q_all"], y_label)
        loss_c_all = loss_fn(outputs["_aux_c_all"], y_label)
        loss_c_next = loss_fn(outputs["_aux_c_next"], y_label)

        if self.output_mode == "an_irt":
            total_loss = (
                loss_kt
                + self.loss_q_all_lambda * self.output_q_all_lambda * loss_q_all
                + self.loss_c_all_lambda * self.output_c_all_lambda * loss_c_all
                + self.loss_c_next_lambda * self.output_c_next_lambda * loss_c_next
            )
        else:
            loss_q_next = loss_fn(outputs["_aux_q_next"], y_label)
            total_loss = (
                loss_kt
                + self.loss_q_all_lambda * self.output_q_all_lambda * loss_q_all
                + self.loss_c_all_lambda * self.output_c_all_lambda * loss_c_all
                + self.loss_c_next_lambda * self.output_c_next_lambda * loss_c_next
                + self.loss_q_next_lambda * self.output_q_all_lambda * loss_q_next
            )

        return total_loss
