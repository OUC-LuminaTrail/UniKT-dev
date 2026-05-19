import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["BDGKTTrainer", "BDGKTModelParams"]


@register_model_params("BDGKT")
class BDGKTModelParams(BaseParamConfig):
    """BDGKT model parameters."""

    def define_params(self) -> tuple[str, dict]:
        return "BDGKT Parameters", {
            "hidden_dim": {
                "type": int,
                "default": 128,
                "short": "hd",
                "help": "Hidden layer dimension (default: 128)",
            },
            "layer_num": {
                "type": int,
                "default": 2,
                "short": "ln",
                "help": "Number of GNN layers (default: 2)",
            },
            "drop1": {
                "type": float,
                "default": 0.3,
                "help": "Feature dropout rate (default: 0.3)",
            },
            "drop2": {
                "type": float,
                "default": 0.3,
                "help": "Attention dropout rate (default: 0.3)",
            },
            "question_max_length": {
                "type": int,
                "default": 20,
                "short": "qml",
                "help": "Student history length l_s (default: 20)",
            },
            "student_max_length": {
                "type": int,
                "default": 5,
                "short": "sml",
                "help": "Top similar students per question l_q (default: 5)",
            },
            "learning_rate": {
                "type": float,
                "default": 0.001,
                "short": "lr",
                "help": "Learning rate (default: 0.001)",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-4,
                "short": "wd",
                "help": "Weight decay (default: 1e-4)",
            },
            "batch_size": {
                "type": int,
                "default": 2000,
                "short": "bs",
                "help": "Batch size (default: 2000)",
            },
            "epochs": {
                "type": int,
                "default": 100,
                "short": "ep",
                "help": "Number of training epochs (default: 100)",
            },
        }


@TRAINERS.register("BDGKT")
class BDGKTTrainer(BaseTrainer):
    """BDGKT trainer using precomputed fixed-shape context tensors."""

    def __init__(self, args=None, data_src=None, exp_manager=None):
        from model.BDGKT.BDGKT_data import BDGKTModelData, _collate_fn
        from model.BDGKT.BDGKT_model import BDGKT

        logger.info("Initializing BDGKT model...")

        # 1. prepare data
        model_data = BDGKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_students,
            num_questions,
            q_kc,
        ) = model_data.prepare_data(args)

        # 2. create model
        l_s = getattr(args, "question_max_length", 20)
        l_q = getattr(args, "student_max_length", 5)
        metadata = data_src.get_metadata()

        model = BDGKT(
            student_num=num_students,
            question_num=num_questions,
            skill_num=metadata["num_skills"],
            hidden_size=args.hidden_dim,
            student_max_length=l_q,
            question_max_length=l_s,
            drop1=args.drop1,
            drop2=args.drop2,
            layer_num=args.layer_num,
            Q_KC=q_kc,
        )

        # 3. optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        loss_fn = torch.nn.BCELoss()

        # 4. early stopping
        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 0.0),
            )

        # 5. init base class
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
            collate_fn=_collate_fn,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="BDGKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

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
