"""QIKT 模型训练器模块"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("QIKT")
class QIKTConfig(ModelConfig):
    """QIKT model configuration.

    Args:
        emb_size: Embedding dimension.
        dropout: Dropout probability.
        mlp_layer_num: Number of MLP layers in prediction heads.
        output_mode: Output fusion mode: 'an' (additive normalization) or 'an_irt'.
        output_q_all_lambda: Output weight for question-all predictions.
        output_c_all_lambda: Output weight for concept-all predictions.
        output_c_next_lambda: Output weight for concept-next predictions.
        loss_q_all_lambda: Loss weight for question-all auxiliary loss.
        loss_c_all_lambda: Loss weight for concept-all auxiliary loss.
        loss_c_next_lambda: Loss weight for concept-next auxiliary loss.
        loss_q_next_lambda: Loss weight for question-next auxiliary loss.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    emb_size: int = field(
        default=64,
        metadata={"optuna": {"type": "int", "low": 32, "high": 128}},
    )
    dropout: float = field(
        default=0.1,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    mlp_layer_num: int = field(
        default=1,
        metadata={"optuna": {"type": "int", "low": 1, "high": 3}},
    )
    output_mode: str = "an"
    output_q_all_lambda: float = 1.0
    output_c_all_lambda: float = 1.0
    output_c_next_lambda: float = 1.0
    loss_q_all_lambda: float = 1.0
    loss_c_all_lambda: float = 1.0
    loss_c_next_lambda: float = 1.0
    loss_q_next_lambda: float = 0.0
    epochs: int = 100
    learning_rate: float = field(
        default=1e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}},
    )
    lr_decay: float | None = None
    # log sampling cannot cover 0.0, so use choices instead
    weight_decay: float = field(
        default=0.0,
        metadata={
            "optuna": {"type": "categorical", "choices": [0.0, 1e-5, 1e-4, 1e-3]}
        },
    )
    batch_size: int = field(
        default=128,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )


@register_trainer("QIKT")
class QIKTTrainer(BaseTrainer):
    """QIKT 模型训练器

    实现双路径预测的融合和多任务损失计算。
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.QIKT.QIKT_data import QIKTModelData
        from model.QIKT.QIKT_model import QIKT

        model_data = QIKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        metadata = data_src.get_metadata()
        logger.info(
            f"Initializing QIKT model: {metadata['num_questions']} questions, "
            f"{metadata['num_skills']} skills, max_concepts={model_data.max_concepts}"
        )

        m = rc.model
        model = QIKT(
            num_questions=metadata["num_questions"],
            num_skills=metadata["num_skills"],
            emb_size=m.emb_size,
            max_concepts=model_data.max_concepts,
            dropout=m.dropout,
            mlp_layer_num=m.mlp_layer_num,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        # Fusion config consumed by forward_pass and _compute_loss
        self.output_mode = m.output_mode
        self.output_q_all_lambda = m.output_q_all_lambda
        self.output_c_all_lambda = m.output_c_all_lambda
        self.output_c_next_lambda = m.output_c_next_lambda
        self.loss_q_all_lambda = m.loss_q_all_lambda
        self.loss_c_all_lambda = m.loss_c_all_lambda
        self.loss_c_next_lambda = m.loss_c_next_lambda
        self.loss_q_next_lambda = m.loss_q_next_lambda

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

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
        预测 response[t+1]（next-item 对齐，output[t] 预测 response[t+1]）。

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

        outputs = self.model(sequence, response, mask, skills)

        y_fused = self._fuse_predictions(outputs)

        y_hat, y_label, _ = self._extract_valid_predictions(y_fused, response, mask)
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # Reuse mask logic so aux predictions align with the main head for multi-task loss
        q_all, _, _ = self._extract_valid_predictions(
            outputs["y_question_all"], response, mask
        )
        c_all, _, _ = self._extract_valid_predictions(
            outputs["y_concept_all"], response, mask
        )
        q_next, _, _ = self._extract_valid_predictions(
            outputs["y_question_next"], response, mask
        )
        c_next, _, _ = self._extract_valid_predictions(
            outputs["y_concept_next"], response, mask
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
