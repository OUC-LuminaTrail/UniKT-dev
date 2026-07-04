"""SQGKT 模型训练器。"""

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["SQGKTTrainer", "SQGKTModelParams"]


@register_model_params("SQGKT")
class SQGKTModelParams(BaseParamConfig):
    def define_params(self) -> tuple[str, dict]:
        return "SQGKT Parameters", {
            "n_hop": {
                "type": int,
                "default": 3,
                "short": "nh",
                "help": "Number of GNN aggregation hops (default: 3)",
            },
            "skill_neighbor_num": {
                "type": int,
                "default": 4,
                "help": "Number of skill neighbors sampled per hop (default: 4)",
            },
            "question_neighbor_num": {
                "type": int,
                "default": 4,
                "help": "Number of question neighbors sampled per hop (default: 4)",
            },
            "user_neighbor_num": {
                "type": int,
                "default": 5,
                "help": "Sampled students per question in the student-question graph (default: 5)",
            },
            "hist_neighbor_num": {
                "type": int,
                "default": 3,
                "short": "hn",
                "help": "Number of historical neighbor samples M (default: 3)",
            },
            "next_neighbor_num": {
                "type": int,
                "default": 4,
                "short": "nn",
                "help": "Number of next-question neighbor samples N (default: 4)",
            },
            "att_bound": {
                "type": float,
                "default": 0.7,
                "help": "Similarity threshold (default: 0.7)",
            },
            "aggregator": {
                "type": str,
                "default": "sum",
                "help": "Aggregator type: sum or concat (default: sum)",
            },
            "variant": {
                "type": str,
                "default": "hsei",
                "help": "History sampling variant: hssi/hsei (same skill) or ssei/dkt (similarity) (default: hsei)",
            },
            "sim_emb": {
                "type": str,
                "default": "question_emb",
                "help": "Similarity embedding: skill_emb/question_emb/feature (default: question_emb)",
            },
            "embedding_dim": {
                "type": int,
                "default": 100,
                "short": "ed",
                "help": "Embedding dimension (default: 100)",
            },
            "hidden_neurons": {
                "type": int,
                "default": [200, 100],
                "nargs": "+",
                "help": "Hidden sizes for each LSTM layer; last layer must equal embedding_dim (default: [200, 100])",
            },
            "dropout_probs": {
                "type": float,
                "default": [0.2, 0.2, 0],
                "nargs": "+",
                "help": "Dropout probabilities for [LSTM, GNN, eval] (default: [0.2, 0.2, 0])",
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
                "help": "Learning rate (default: 0.001)",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch (default: no decay)",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-8,
                "short": "wd",
                "help": "Weight decay (default: 1e-8)",
            },
            "batch_size": {
                "type": int,
                "default": 32,
                "short": "bs",
                "help": "Batch size (default: 32)",
            },
        }


@register_trainer("SQGKT")
class SQGKTTrainer(BaseTrainer):
    """SQGKT 模型训练器。"""

    def __init__(self, args=None, data_src=None, exp_manager=None):
        from model.SQGKT.SQGKT_data import SQGKTModelData
        from model.SQGKT.SQGKT_model import SQGKT

        model_data = SQGKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.graph_data,
            self.num_skills,
            self.num_questions,
            self.num_users,
            train_collate_fn,
            val_collate_fn,
        ) = model_data.prepare_data(args)

        logger.info("Initializing SQGKT model...")
        metadata = dict(data_src.get_metadata())
        metadata["num_users"] = self.num_users
        model = SQGKT(args=args, data_metadata=metadata)
        super().__init__(model)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        lr_scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
            if args.lr_decay
            else None
        )
        early_stopping_cfg = EarlyStoppingConfig(
            monitor=args.es_monitor,
            mode=args.es_mode,
            patience=args.es_patience,
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
            collate_fn=train_collate_fn,
            val_collate_fn=val_collate_fn,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="SQGKT",
            dataset_name=args.dataset,
        ).build()

        self.graph_data = {
            key: value.to(self.device_) if hasattr(value, "to") else value
            for key, value in self.graph_data.items()
        }
        self.graph_data["feature_embedding"] = self.model.feature_embedding.weight

        logger.info(
            f"SQGKT Trainer: {self.num_skills} skills, {self.num_questions} questions"
        )

    def forward_pass(self, batch_data):
        sequence = self._move_tensor_to_device(batch_data["sequence"])
        response = self._move_tensor_to_device(batch_data["response"])
        mask = self._move_tensor_to_device(batch_data["mask"])
        user_id = self._move_tensor_to_device(batch_data["user_id"])
        skills = self._move_tensor_to_device(batch_data["skills"])
        hist_neighbor_index = self._move_tensor_to_device(
            batch_data["hist_neighbor_index"]
        )

        y_hat_full = self._pad_to_full_sequence(
            self.model(
                user_sequence=sequence,
                user_response=response,
                user_mask=mask,
                user_ids=user_id,
                skills=skills,
                graph_data=self.graph_data,
                hist_neighbor_index=hist_neighbor_index,
            )
        )
        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.0),
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
        }
