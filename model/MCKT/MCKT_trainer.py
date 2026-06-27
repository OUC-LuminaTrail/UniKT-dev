"""MCKT trainer."""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("MCKT")
class MCKTModelParams(BaseParamConfig):
    def define_params(self) -> tuple[str, dict]:
        group_name = "MCKT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 128,
                "help": "Embedding and hidden dimension",
            },
            "n_heads": {
                "type": int,
                "default": 8,
                "help": "Number of attention heads",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability",
            },
            "temperature": {
                "type": float,
                "default": 0.8,
                "help": "Contrastive learning temperature",
            },
            "sim_threshold": {
                "type": float,
                "default": 0.8,
                "help": "Question cosine-similarity threshold for topic state attention",
            },
            "cl_batch_size": {
                "type": int,
                "default": 10000,
                "help": "Chunk size for question/interaction contrastive loss",
            },
            "cl_exp_mode": {
                "type": str,
                "default": "source",
                "choices": ["paper", "source"],
                "help": (
                    "Use 'source' for the released MCKT code path's double-exp "
                    "question/interaction CL, or 'paper' for exp(cos/tau)"
                ),
            },
            "pro_loss_weight": {
                "type": float,
                "default": 1.0,
                "help": "Question-level contrastive loss weight",
            },
            "react_loss_weight": {
                "type": float,
                "default": 1.0,
                "help": "Interaction-level contrastive loss weight",
            },
            "state_loss_weight": {
                "type": float,
                "default": 0.0001,
                "help": "Knowledge-state contrastive loss weight",
            },
            "pos_strategy": {
                "type": str,
                "default": "shared_kc",
                "choices": ["same_kc_set", "shared_kc"],
                "help": "How to build pos_matrix from question-KC relations",
            },
            "pos_include_self": {
                "type": bool,
                "default": True,
                "help": "Include diagonal self-positive pairs in pos_matrix",
            },
            "epochs": {
                "type": int,
                "default": 70,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 0.002,
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
                "default": 1e-5,
                "short": "wd",
                "help": "Weight decay for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 80,
                "short": "bs",
                "help": "Batch size for training",
            },
            "max_grad_norm": {
                "type": float,
                "default": 15.0,
                "help": "Max gradient norm for clipping",
            },
        }
        return group_name, params


@TRAINERS.register("MCKT")
class MCKTTrainer(BaseTrainer):
    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.MCKT.MCKT_data import MCKTModelData
        from model.MCKT.MCKT_model import MCKT

        model_data = MCKTModelData(data_src)
        train_dataset, val_dataset, test_dataset, pos_matrix = model_data.prepare_data(
            args
        )

        metadata = data_src.get_metadata()
        logger.info(
            f"Initializing MCKT model: {metadata['num_questions']} questions, "
            f"{metadata['num_skills']} skills"
        )
        model = MCKT(
            num_questions=metadata["num_questions"],
            num_skills=metadata["num_skills"],
            d_model=args.d_model,
            dropout=args.dropout,
            n_heads=args.n_heads,
            temperature=args.temperature,
            sim_threshold=args.sim_threshold,
            cl_batch_size=args.cl_batch_size,
            cl_exp_mode=args.cl_exp_mode,
            pro_loss_weight=args.pro_loss_weight,
            react_loss_weight=args.react_loss_weight,
            state_loss_weight=args.state_loss_weight,
            pos_matrix=pos_matrix,
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
                min_delta=getattr(args, "es_min_delta", 0.0),
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
            max_clip_grad_norm=args.max_grad_norm,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="MCKT",
            dataset_name=getattr(args, "dataset", ""),
            skip_test=getattr(args, "skip_test", False),
        ).build()

    def _build_mckt_inputs(
        self,
        question: torch.Tensor,
        response: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        last_problem = question[:, :-1]
        last_ans = response[:, :-1]
        next_problem = question[:, 1:]
        next_ans = response[:, 1:]
        return last_problem, last_ans, next_problem, next_ans

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        question, response, mask = batch_data
        question = self._move_tensor_to_device(question)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        last_problem, last_ans, next_problem, next_ans = self._build_mckt_inputs(
            question, response
        )
        y_hat_full, state_loss, pro_loss, react_loss = self.model(
            last_problem, last_ans, next_problem, next_ans
        )

        # y_hat_full is [B, S-1] next-item (out[t] predicts response[t+1]); pad to
        # [B, S] and reuse the unified extraction contract (adjacent-pair mask).
        y_hat_full = self._pad_to_full_sequence(y_hat_full)
        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "_state_loss": state_loss,
            "_pro_loss": pro_loss,
            "_react_loss": react_loss,
        }

    def _compute_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        bce_loss = self.loss(outputs["y_hat"], outputs["y_label"])
        if not self.model.training:
            return bce_loss
        return (
            bce_loss
            + outputs["_state_loss"]
            + outputs["_pro_loss"]
            + outputs["_react_loss"]
        )
