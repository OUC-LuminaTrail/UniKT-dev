from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)

__all__ = ["BDGKTTrainer", "BDGKTConfig"]


@register_model_config("BDGKT")
class BDGKTConfig(ModelConfig):
    """BDGKT model configuration."""

    hidden_dim: int = field(
        default=128,
        metadata={"help": "Hidden layer dimension (default: 128)", "short": "hd"},
    )
    layer_num: int = field(
        default=2, metadata={"help": "Number of GNN layers (default: 2)", "short": "ln"}
    )
    drop1: float = field(
        default=0.3, metadata={"help": "Feature dropout rate (default: 0.3)"}
    )
    drop2: float = field(
        default=0.3, metadata={"help": "Attention dropout rate (default: 0.3)"}
    )
    question_max_length: int = field(
        default=20,
        metadata={"help": "Student history length l_s (default: 20)", "short": "qml"},
    )
    student_max_length: int = field(
        default=5,
        metadata={
            "help": "Top similar students per question l_q (default: 5)",
            "short": "sml",
        },
    )
    learning_rate: float = field(
        default=0.001,
        metadata={"help": "Learning rate (default: 0.001)", "short": "lr"},
    )
    weight_decay: float = field(
        default=1e-4, metadata={"help": "Weight decay (default: 1e-4)", "short": "wd"}
    )
    batch_size: int = field(
        default=2000, metadata={"help": "Batch size (default: 2000)", "short": "bs"}
    )
    epochs: int = field(
        default=100,
        metadata={"help": "Number of training epochs (default: 100)", "short": "ep"},
    )


@register_trainer("BDGKT")
class BDGKTTrainer(BaseTrainer):
    """BDGKT trainer using precomputed fixed-shape context tensors."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.BDGKT.BDGKT_data import BDGKTModelData, _collate_fn
        from model.BDGKT.BDGKT_model import BDGKT

        logger.info("Initializing BDGKT model...")

        # 1. prepare data
        model_data = BDGKTModelData(data_src, cache=rc.general.cache)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_students,
            num_questions,
            q_kc,
        ) = model_data.prepare_data(rc)

        # 2. create model
        m = rc.model
        metadata = data_src.get_metadata()
        model = BDGKT(
            student_num=num_students,
            question_num=num_questions,
            skill_num=metadata["num_skills"],
            hidden_size=m.hidden_dim,
            student_max_length=m.student_max_length,
            question_max_length=m.question_max_length,
            drop1=m.drop1,
            drop2=m.drop2,
            layer_num=m.layer_num,
            Q_KC=q_kc,
        )

        # 3. optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=m.learning_rate,
            weight_decay=m.weight_decay,
        )
        loss_fn = torch.nn.BCELoss()

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
            collate_fn=_collate_fn,
        )

    def forward_pass(self, batch_data):
        """BDGKT forward pass.

        Args:
            batch_data: (target_students, target_questions, labels,
                        hist_q, hist_r, hist_mask,
                        q_ans_s, q_ans_r, q_ans_mask)
        """
        (
            target_students,
            target_questions,
            labels,
            hist_q,
            hist_r,
            hist_mask,
            q_ans_s,
            q_ans_r,
            q_ans_mask,
        ) = batch_data

        target_students = self._move_tensor_to_device(target_students)
        target_questions = self._move_tensor_to_device(target_questions)
        y_label = self._move_tensor_to_device(labels).float()

        hist_q = self._move_tensor_to_device(hist_q)
        hist_r = self._move_tensor_to_device(hist_r)
        hist_mask = self._move_tensor_to_device(hist_mask)
        q_ans_s = self._move_tensor_to_device(q_ans_s)
        q_ans_r = self._move_tensor_to_device(q_ans_r)
        q_ans_mask = self._move_tensor_to_device(q_ans_mask)

        pred = self.model(
            target_students,
            target_questions,
            hist_q,
            hist_r,
            hist_mask,
            q_ans_s,
            q_ans_r,
            q_ans_mask,
        )

        if y_label.numel() == 0:
            raise ValueError("Empty valid targets in current batch.")

        y_predict = (pred >= 0.5).long()

        return {
            "y_hat": pred,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": pred,
            "y_prob": pred,
        }
