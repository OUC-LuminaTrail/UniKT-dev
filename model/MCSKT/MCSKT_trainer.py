"""MCSKT 模型训练器"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import get_logger, register_trainer
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("MCSKT")
class MCSKTModelParams(BaseParamConfig):
    """MCSKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "MCSKT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 256,
                "help": "Hidden dimension (256 with n_blocks=5 reproduces the paper's "
                "~5.6M param count)",
            },
            "n_blocks": {
                "type": int,
                "default": 5,
                "help": "Number of Mamba blocks per encoder (Q/K)",
            },
            "num_heads": {
                "type": int,
                "default": 8,
                "help": "Number of dynamic k-sparse attention heads",
            },
            "d_state": {
                "type": int,
                "default": 16,
                "help": "SSM latent state dimension in Mamba",
            },
            "d_conv": {
                "type": int,
                "default": 4,
                "help": "Conv1D kernel width in Mamba block",
            },
            "expand": {
                "type": int,
                "default": 2,
                "help": "Mamba internal expansion factor",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability (applied to embeddings, Mamba blocks, "
                "prediction head)",
            },
            "l2": {
                "type": float,
                "default": 1e-5,
                "help": "L2 reg coefficient for Rasch difficulty parameter",
            },
            "num_rgap": {
                "type": int,
                "default": 100,
                "help": "Number of review-gap (repeated time gap) buckets",
            },
            "num_sgap": {
                "type": int,
                "default": 100,
                "help": "Number of sequence-gap buckets",
            },
            "num_pcount": {
                "type": int,
                "default": 15,
                "help": "Number of past-trial-count buckets",
            },
            "delta1": {
                "type": float,
                "default": 0.25,
                "help": "Lower bound of dynamic sparsity interval [d1, d2] (paper: 1/4)",
            },
            "delta2": {
                "type": float,
                "default": 0.667,
                "help": "Upper bound of dynamic sparsity interval [d1, d2] (paper: 2/3)",
            },
            "epochs": {
                "type": int,
                "default": 200,
                "short": "ep",
                "help": "Number of training epochs (paper: 200)",
            },
            "learning_rate": {
                "type": float,
                "default": 1e-4,
                "short": "lr",
                "help": "Learning rate (paper: 1e-5; raised for stable Adam)",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-4,
                "short": "wd",
                "help": "Weight decay for optimizer",
            },
            "max_clip_grad_norm": {
                "type": float,
                "default": 1.0,
                "short": "clip",
                "help": "Max gradient norm for clipping (None to disable)",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size (paper: 64)",
            },
        }
        return group_name, params


@register_trainer("MCSKT")
class MCSKTTrainer(BaseTrainer):
    """MCSKT 模型训练器。

    预测语义（same_position）：
        - preds[:, t] 使用历史 0..t-1 与当前题目 x_t 预测 response[t]
        - trainer 用 ``same_position=True`` 由内置函数归一化为 next-item 视图
    """

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        from model.MCSKT.MCSKT_data import MCSKTModelData
        from model.MCSKT.MCSKT_model import MCSKT

        model_data = MCSKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        logger.info("Initializing MCSKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)

        if n_pid > 0:
            logger.info(f"MCSKT: Using Rasch embeddings with {n_pid} questions")
        else:
            logger.warning("MCSKT: Problem ID not available, using skill-only model")

        model = MCSKT(
            num_c=metadata["num_skills"],
            n_pid=n_pid,
            num_rgap=args.num_rgap,
            num_sgap=args.num_sgap,
            num_pcount=args.num_pcount,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            num_heads=args.num_heads,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            dropout=args.dropout,
            l2=args.l2,
            delta1=args.delta1,
            delta2=args.delta2,
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
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
            max_clip_grad_norm=getattr(args, "max_clip_grad_norm", None),
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="MCSKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def _build_pid_data(
        self, question: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """构建 Rasch pid 数据，0 保留给填充位置。"""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """MCSKT 前向传播（same_position 约定）。

        batch_data: (sequence, response, mask, question, rgap, sgap, pcount)
        """
        sequence, response, mask, question, rgap, sgap, pcount = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)
        rgap = self._move_tensor_to_device(rgap)
        sgap = self._move_tensor_to_device(sgap)
        pcount = self._move_tensor_to_device(pcount)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else question

        y_hat_full, c_reg_loss = self.model(
            sequence, response, mask, pid_data, rgap, sgap, pcount
        )

        # same_position：out[t] 预测 response[t]，归一化为 next-item 视图
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }
        if use_pid:
            result["c_reg_loss"] = c_reg_loss
        return result

    def test_forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """测试前向传播，支持 windowlateauc_mean 评估。

        batch_data: (sequence, response, mask, late_group_id, true_labels,
                     question, rgap, sgap, pcount)  9-元组
        遗忘特征由窗口内时间戳实时计算（非零填充）。
        """
        (
            sequence,
            response,
            mask,
            late_group_id,
            true_labels,
            question,
            rgap,
            sgap,
            pcount,
        ) = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)
        rgap = self._move_tensor_to_device(rgap)
        sgap = self._move_tensor_to_device(sgap)
        pcount = self._move_tensor_to_device(pcount)

        use_pid = self.model.n_pid > 0
        # windowlate 的 mask 仅在目标位为 1；模型注意力需要"所有有效位置"作为 key 有效掩码，
        # 故用 late_group_id>=0 构造完整有效掩码，目标位掩码仅用于选取评估预测。
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else question

        y_hat_full, _ = self.model(
            sequence, response, valid_mask, pid_data, rgap, sgap, pcount
        )

        # windowlate：每个窗口仅在目标位（mask=1）评估；
        # same_position 下 out[target] 用历史 0..target-1 预测 response[target]
        y_hat = torch.masked_select(y_hat_full, mask)
        y_label = torch.masked_select(true_labels.float(), mask)
        group_ids = torch.masked_select(late_group_id, mask)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失：BCE + Rasch 正则化（论文 Eq.9 / Eq.11 风格）。"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            return bce_loss + outputs["c_reg_loss"]
        return bce_loss
