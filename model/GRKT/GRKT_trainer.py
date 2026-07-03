from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["GRKTTrainer", "GRKTModelParams"]


@register_model_params("GRKT")
class GRKTModelParams(BaseParamConfig):
    """GRKT model hyperparameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "GRKT Parameters"
        params = {
            "d_hidden": {
                "type": int,
                "default": 128,
                "short": "dh",
                "help": "Dimension of embedding and hidden states (default: 128)",
            },
            "k_hidden": {
                "type": int,
                "default": 16,
                "short": "kh",
                "help": "Dimension of hidden knowledge mastery (default: 16)",
            },
            "pos_mode": {
                "type": str,
                "default": "softmax",
                "short": "pm",
                "help": "Positive projection mode: sigmoid|softplus|relu|softmax|none (default: softmax)",
            },
            "k_hop": {
                "type": int,
                "default": 1,
                "short": "kp",
                "help": "Hops of graph operation (default: 1)",
            },
            "thresh": {
                "type": float,
                "default": 0.6,
                "short": "th",
                "help": "Threshold for relevance/prerequisite graph sparsity (default: 0.6)",
            },
            "tau": {
                "type": float,
                "default": 0.2,
                "short": "tau",
                "help": "Gumbel softmax temperature (default: 0.2)",
            },
            "alpha": {
                "type": float,
                "default": 0.01,
                "short": "alpha",
                "help": "Time interval scaling factor (default: 0.01)",
            },
            "epochs": {
                "type": int,
                "default": 200,
                "short": "ep",
                "help": "Number of training epochs (default: 200)",
            },
            "learning_rate": {
                "type": float,
                "default": 0.001,
                "short": "lr",
                "help": "Learning rate (default: 0.001)",
            },
            "weight_decay": {
                "type": float,
                "default": 0.0,
                "short": "wd",
                "help": "Weight decay (L2 regularization) (default: 0.0)",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size (default: 128)",
            },
        }
        return group_name, params


@register_trainer("GRKT")
class GRKTTrainer(BaseTrainer):
    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        # 1. Prepare data
        from model.GRKT.GRKT_data import GRKTModelData

        model_data = GRKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            num_questions,
            num_skills,
            max_k,
            self.rel_map,
            self.pre_map,
        ) = model_data.prepare_data(args)

        logger.info(
            f"GRKT data prepared: n_questions={num_questions}, "
            f"n_skills={num_skills}, max_skills_per_q={max_k}"
        )

        # 2. Build metadata dict for model construction
        metadata = data_src.get_metadata()
        metadata["num_questions"] = num_questions
        metadata["num_skills"] = num_skills

        # 3. Initialize model
        from model.GRKT.GRKT_model import GRKT

        logger.info("Initializing GRKT model...")
        model = GRKT(args, metadata, self.rel_map, self.pre_map)

        # 4. Create optimizer and loss function
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # 5. Initialize base trainer
        super().__init__(model)

        # 6. Build early stopping config
        early_stopping_cfg = None
        es_patience = args.es_patience or None
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=args.es_monitor or "auc",
                mode=args.es_mode or "max",
                patience=es_patience,
                min_delta=args.es_min_delta or 0.0,
            )

        # 7. Configure trainer
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
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="GRKT",
            dataset_name=args.dataset,
        ).build()

    def forward_pass(self, batch_data):
        questions, knows, responses, times, mask = batch_data
        questions = self._move_tensor_to_device(questions)
        knows = self._move_tensor_to_device(knows)
        responses = self._move_tensor_to_device(responses)
        times = self._move_tensor_to_device(times)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # Forward: scores[i] predicts response[i], shape [B, S]
        scores = self.model(questions, knows, responses, times)

        # Drop first prediction (no prior history), evaluate on 1..S-1
        y_hat = scores[:, 1:]
        y_label = responses.float()[:, 1:]
        valid_mask = mask[:, 1:]

        y_hat = torch.masked_select(y_hat, valid_mask)
        y_label = torch.masked_select(y_label, valid_mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }
