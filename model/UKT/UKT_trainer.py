"""UKT 模型训练器"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("UKT")
class UKTModelParams(BaseParamConfig):
    """UKT 模型参数配置"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "UKT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 256,
                "help": "Hidden dimension of the model",
            },
            "n_blocks": {
                "type": int,
                "default": 4,
                "help": "Number of transformer blocks",
            },
            "num_attn_heads": {
                "type": int,
                "default": 8,
                "help": "Number of attention heads",
            },
            "dropout": {
                "type": float,
                "default": 0.2,
                "help": "Dropout probability",
            },
            "d_ff": {
                "type": int,
                "default": 512,
                "help": "Feed-forward network dimension",
            },
            "final_fc_dim": {
                "type": int,
                "default": 512,
                "help": "First fully connected layer dimension",
            },
            "final_fc_dim2": {
                "type": int,
                "default": 256,
                "help": "Second fully connected layer dimension",
            },
            "kq_same": {
                "type": int,
                "default": 1,
                "help": "Whether key and query use the same linear transformation (1=yes, 0=no)",
            },
            "separate_qa": {
                "type": int,
                "default": 0,
                "help": "Whether to use separate QA embeddings (1=yes, 0=no)",
            },
            "use_CL": {
                "type": int,
                "default": 1,
                "help": "Enable contrastive learning (1=yes, 0=no)",
            },
            "cl_weight": {
                "type": float,
                "default": 0.02,
                "help": "Weight for contrastive learning loss",
            },
            "l2": {
                "type": float,
                "default": 1e-5,
                "help": "L2 regularization coefficient for Rasch difficulty",
            },
            "no_uncertainty_aug": {
                "type": bool,
                "default": False,
                "help": "Disable uncertainty augmentation for contrastive learning",
            },
            "atten_type": {
                "type": str,
                "default": "w2",
                "help": "Attention type: w2 (Wasserstein) or dp (dot product)",
            },
            "epochs": {
                "type": int,
                "default": 200,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-4,
                "short": "lr",
                "help": "Learning rate for optimizer",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-5,
                "short": "wd",
                "help": "Weight decay for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size for training",
            },
        }
        return group_name, params


@TRAINERS.register("UKT")
class UKTTrainer(BaseTrainer):
    """UKT 模型训练器

    负责初始化UKT模型、优化器和训练数据，并实现前向传播逻辑。
    支持对比学习损失的组合训练。
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        from model.UKT.UKT_data import UKTModelData

        model_data = UKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        from model.UKT.UKT_model import UKT

        logger.info("Initializing UKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)
        if n_pid > 0:
            logger.info(f"UKT: Using Problem ID (Rasch model) with {n_pid} questions")
        else:
            logger.warning("UKT: Problem ID not available, using skill-only model")

        model = UKT(
            num_c=metadata["num_skills"],
            n_pid=n_pid,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            d_ff=args.d_ff,
            final_fc_dim=args.final_fc_dim,
            final_fc_dim2=args.final_fc_dim2,
            num_attn_heads=args.num_attn_heads,
            kq_same=args.kq_same,
            separate_qa=bool(args.separate_qa),
            use_CL=bool(args.use_CL),
            cl_weight=args.cl_weight,
            use_uncertainty_aug=not args.no_uncertainty_aug,
            l2=args.l2,
            atten_type=args.atten_type,
            seq_len=args.max_seq_len,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="UKT",
            dataset_name=args.dataset,
        ).build()

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build UKT Rasch pid data with 0 reserved for padding."""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(self, batch_data):
        """UKT 前向传播

        UKT 使用因果注意力，preds[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]。
        """
        if len(batch_data) == 5:
            sequence, response, mask, question, response_aug = batch_data
        else:
            sequence, response, mask, question = batch_data
            response_aug = None

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)
        if response_aug is not None:
            response_aug = self._move_tensor_to_device(response_aug)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else None

        preds, cl_loss, _, c_reg_loss = self.model(
            sequence, response, mask, pid_data, response_aug
        )

        y_norm = torch.cat([preds[:, 1:], torch.zeros_like(preds[:, :1])], dim=1)
        y_hat, y_label, _ = self._extract_valid_predictions(y_norm, response, mask)
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
            "cl_loss": cl_loss,
        }
        if use_pid:
            result["c_reg_loss"] = c_reg_loss
        return result

    def test_forward_pass(self, batch_data):
        sequence, response, mask, late_group_id, true_labels, question = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else None

        preds, _, _, _ = self.model(sequence, response, mask, pid_data)

        y_hat = torch.masked_select(preds, mask)
        y_label = torch.masked_select(true_labels, mask).float()
        group_ids = torch.masked_select(late_group_id, mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs):
        """计算损失：BCE + 对比学习损失"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if self.model.use_CL and "cl_loss" in outputs:
            bce_loss = bce_loss + self.model.cl_weight * outputs["cl_loss"]

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            bce_loss = bce_loss + outputs["c_reg_loss"]

        return bce_loss
