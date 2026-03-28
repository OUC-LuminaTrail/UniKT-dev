"""
SGKT model trainer.

Defines training logic for Session Graph-based Knowledge Tracing model.
"""

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)

__all__ = ["SGKTTrainer", "SGKTModelParams"]


@register_model_params("SGKT")
class SGKTModelParams(BaseParamConfig):
    """SGKT model-specific parameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "SGKT Parameters"
        params = {
            # HRG (GCNConv) parameters
            "n_hop": {
                "type": int,
                "default": 3,
                "short": "nh",
                "help": "Number of GCN layers for HRG graph (multi-hop aggregation) (default: 3)",
            },
            # SG (GatedGraphConv) parameters
            "sg_layers": {
                "type": int,
                "default": 2,
                "help": "Number of GatedGraphConv layers for session graph (default: 2)",
            },
            # Neighbor sampling parameters
            "hist_neighbor_num": {
                "type": int,
                "default": 3,
                "short": "hn",
                "help": "Number of historical neighbors to sample (default: 3)",
            },
            "next_neighbor_num": {
                "type": int,
                "default": 4,
                "short": "nn",
                "help": "Number of next question neighbors to sample (default: 4)",
            },
            "att_bound": {
                "type": float,
                "default": 0.7,
                "help": "Similarity threshold for historical neighbor sampling (default: 0.7)",
            },
            "cooc_neighbor_num": {
                "type": int,
                "default": 0,
                "help": "Max number of co-occurrence neighbors per question in HRG graph (default: 0)",
            },
            "skill_neighbor_num": {
                "type": int,
                "default": 4,
                "help": "Number of skill neighbors to sample per hop (default: 4)",
            },
            "question_neighbor_num": {
                "type": int,
                "default": 4,
                "help": "Number of question neighbors to sample per hop (default: 4)",
            },
            "aggregator": {
                "type": str,
                "default": "sum",
                "help": "Aggregator type: sum or concat (default: sum)",
            },
            "select_index": {
                "type": list,
                "default": [0, 1, 2],
                "nargs": "?",
                "help": "Feature indices used for model inputs (default: [0, 1, 2])",
            },
            "sim_emb": {
                "type": str,
                "default": "question_emb",
                "help": "Embedding type for similarity (default: question_emb)",
            },
            # Standard parameters
            "embedding_dim": {
                "type": int,
                "default": 100,
                "short": "ed",
                "help": "Embedding dimension (default: 100)",
            },
            "hidden_dim": {
                "type": int,
                "default": 100,
                "short": "hd",
                "help": "Hidden layer dimension (default: 100)",
            },
            "dropout_keep_probs": {
                "type": list,
                "default": [0.8, 0.8, 1],
                "nargs": "?",
                "help": "Dropout keep probabilities for each GCN layer in HRG (default: [0.8, 0.8, 1])",
            },
            "epochs": {
                "type": int,
                "default": 100,
                "short": "ep",
                "help": "Number of training epochs (default: 100)",
            },
            "learning_rate": {
                "type": float,
                "default": 0.00025,
                "short": "lr",
                "help": "Learning rate for optimizer (default: 0.00025)",
            },
            "lr_decay": {
                "type": float,
                "default": 0.92,
                "help": "Learning rate decay factor per epoch (default: 0.92)",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-8,
                "short": "wd",
                "help": "Weight decay (L2 regularization) for optimizer (default: 1e-8)",
            },
            "batch_size": {
                "type": int,
                "default": 6,
                "short": "bs",
                "help": "Batch size for training (default: 6)",
            },
        }

        return group_name, params


@TRAINERS.register("SGKT")
class SGKTTrainer(BaseTrainer):
    """
    SGKT model trainer
    """

    def __init__(
        self,
        args=None,
        data_src=None,
        exp_manager=None,
    ):
        # 1. 准备数据
        from model.SGKT import SGKTModelData

        model_data = SGKTModelData(data_src)
        (
            train_dataset,
            val_dataset,
            test_dataset,
            self.hrg_data,
            self.num_skills,
            self.num_questions,
            train_collate_fn,
            val_collate_fn,
        ) = model_data.prepare_data(args)

        # 2. 初始化模型
        from model.SGKT.SGKT_model import SGKT

        logger.info("Initializing SGKT model...")
        model = SGKT(args=args, data_metadata=data_src.get_metadata())

        # 3. 调用父类构造函数
        super().__init__(model)

        # 4. 创建优化器和损失函数
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # 5. 创建学习率调度器
        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        # 6. 构建早停配置
        early_stopping_cfg = None
        es_patience = getattr(args, "es_patience", None)
        if es_patience is not None:
            early_stopping_cfg = EarlyStoppingConfig(
                monitor=getattr(args, "es_monitor", "auc"),
                mode=getattr(args, "es_mode", "max"),
                patience=es_patience,
                min_delta=getattr(args, "es_min_delta", 0.0),
            )

        # 7. 配置训练器
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
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="SGKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

        # 8. Move HRG context to device and bind feature embedding table
        self.hrg_data = {
            key: value.to(self.device_) if hasattr(value, "to") else value
            for key, value in self.hrg_data.items()
        }
        self.hrg_data["feature_embedding"] = self.model.feature_embedding.weight

        logger.info(
            f"SGKT Trainer initialized with {self.num_skills} skills and {self.num_questions} questions"
        )

    def forward_pass(self, batch_data):
        """
        SGKT forward pass.

        Args:
            batch_data: Dictionary with keys 'sequence', 'response', 'mask', 'hist_neighbor_index'

        Returns:
            Dictionary with 'y_hat', 'y_label', 'y_predict'
        """
        # Unpack batch data
        batch_dict = batch_data
        sequence = batch_dict["sequence"]
        response = batch_dict["response"]
        mask = batch_dict["mask"]
        hist_neighbor_index = batch_dict.get("hist_neighbor_index", None)

        # Move to device
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # Move hist_neighbor_index to device if provided
        if hist_neighbor_index is not None:
            hist_neighbor_index = self._move_tensor_to_device(hist_neighbor_index)

        # Model forward pass
        # Output at time t predicts label at time t+1
        y_hat_full = self.model(
            user_sequence=sequence,
            user_response=response,
            user_mask=mask,
            hrg_data=self.hrg_data,
            hist_neighbor_index=hist_neighbor_index,  # Pass pre-computed fallback indices
        )  # [B, S-1]

        # Extract valid predictions
        # Model already returns [B, S-1] (shifted predictions)
        # So we shift response and mask to match: [B, S] -> [B, S-1]
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full,
            response[:, 1:],  # Shift to match predictions
            mask[:, 1:],  # Shift to match predictions
            skip_first=False,
        )

        # Handle empty batch
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # Generate binary predictions
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
        }
