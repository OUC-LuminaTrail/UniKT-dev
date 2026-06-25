from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("FAKT")
class FAKTModelParams(BaseParamConfig):
    """FAKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "FAKT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 256,
                "help": "Hidden dimension of the model",
            },
            "n_blocks": {
                "type": int,
                "default": 2,
                "help": "Number of transformer blocks",
            },
            "num_attn_heads": {
                "type": int,
                "default": 4,
                "help": "Number of attention heads",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability",
            },
            "d_ff": {
                "type": int,
                "default": 256,
                "help": "Feed-forward network dimension",
            },
            "final_fc_dim": {
                "type": int,
                "default": 256,
                "help": "Final fully connected layer dimension",
            },
            "final_fc_dim2": {
                "type": int,
                "default": 256,
                "help": "Second final fully connected layer dimension",
            },
            "kernel_size1": {
                "type": int,
                "default": 5,
                "help": "Kernel size for the first frequency-band causal conv",
            },
            "kernel_size2": {
                "type": int,
                "default": 5,
                "help": "Kernel size for the second frequency-band causal conv",
            },
            "kq_same": {
                "type": bool,
                "default": True,
                "help": "Whether key and query share the linear transform",
            },
            "separate_qa": {
                "type": bool,
                "default": False,
                "help": "Whether to use separate QA embeddings",
            },
            "emb_type": {
                "type": str,
                "default": "qidband",
                "help": "Embedding type (default enables frequency-band enhancement)",
            },
            "use_moe": {
                "type": bool,
                "default": True,
                "help": "Whether to enable Mixture-of-Experts",
            },
            "num_experts": {
                "type": int,
                "default": 4,
                "help": "Number of experts in MoE",
            },
            "confidence_thresholds": {
                "type": str,
                "default": "[0.8, 0.6, 0.4]",
                "help": "Confidence thresholds for adaptive expert selection",
            },
            "mamba_d_state": {
                "type": int,
                "default": 16,
                "help": "Mamba SSM state dimension",
            },
            "mamba_d_conv": {
                "type": int,
                "default": 4,
                "help": "Mamba local convolution width",
            },
            "mamba_expand": {
                "type": int,
                "default": 2,
                "help": "Mamba expansion factor",
            },
            "min_experts": {
                "type": int,
                "default": 1,
                "help": "Minimum number of experts selected adaptively",
            },
            "max_experts": {
                "type": int,
                "default": None,
                "help": "Maximum number of experts selected adaptively (None=num_experts)",
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
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch",
            },
            "weight_decay": {
                "type": float,
                "default": 0.0,
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


@TRAINERS.register("FAKT")
class FAKTTrainer(BaseTrainer):
    """FAKT 模型训练器

    负责初始化 FAKT 模型、优化器和训练数据，并实现前向传播逻辑。
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        # 准备数据
        from model.FAKT.FAKT_data import FAKTModelData

        model_data = FAKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        # 初始化模型
        from model.FAKT.FAKT_model import FAKT

        logger.info("Initializing FAKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)
        if n_pid > 0:
            logger.info(f"FAKT: Using Problem ID (Rasch model) with {n_pid} questions")
        else:
            logger.info("FAKT: Problem ID not available, using skill-only model")

        max_experts = args.max_experts
        model = FAKT(
            n_question=metadata["num_skills"],
            n_pid=n_pid,
            num_rgap=model_data.num_rgap,
            num_sgap=model_data.num_sgap,
            num_pcount=model_data.num_pcount,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            d_ff=args.d_ff,
            seq_len=metadata["max_seq_len"],
            kernel_size1=args.kernel_size1,
            kernel_size2=args.kernel_size2,
            freq=True,
            kq_same=args.kq_same,
            final_fc_dim=args.final_fc_dim,
            final_fc_dim2=args.final_fc_dim2,
            num_attn_heads=args.num_attn_heads,
            separate_qa=args.separate_qa,
            emb_type=args.emb_type,
            use_moe=args.use_moe,
            num_experts=args.num_experts,
            confidence_thresholds=args.confidence_thresholds,
            mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
            min_experts=args.min_experts,
            max_experts=max_experts,
        )

        # 创建优化器和损失函数
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        # 创建学习率调度器
        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        # 初始化基类训练器
        super().__init__(model)

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
            model_name="FAKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def _build_pid_data(
        self, question: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """构造 Rasch pid 数据：question + 1，padding 位置置 0。"""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """FAKT 前向传播"""
        sequence, response, mask, rgaps, sgaps, pcounts, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        rgaps = self._move_tensor_to_device(rgaps)
        sgaps = self._move_tensor_to_device(sgaps)
        pcounts = self._move_tensor_to_device(pcounts)
        question = self._move_tensor_to_device(question)

        # Rasch pid：question + 1 偏移，padding(mask=0) 位置置 0
        pid_data = (
            self._build_pid_data(question, mask) if self.model.n_pid > 0 else None
        )

        # 模型前向传播
        preds = self.model(sequence, response, rgaps, sgaps, pcounts, pid_data)

        # 提取有效位置的预测和标签
        y_hat, y_label, _ = self._extract_valid_predictions(
            preds, response, mask, same_position=True
        )

        # 处理空批次
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # 生成二分类预测
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

    def test_forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """测试前向传播，支持 windowlateauc_mean 评估。

        batch_data: (sequence, response, mask, late_group_id, true_labels, question,
                     rgaps, sgaps, pcounts)
        """
        (
            sequence,
            response,
            mask,
            late_group_id,
            true_labels,
            question,
            rgaps,
            sgaps,
            pcounts,
        ) = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)
        rgaps = self._move_tensor_to_device(rgaps)
        sgaps = self._move_tensor_to_device(sgaps)
        pcounts = self._move_tensor_to_device(pcounts)

        # Rasch pid：question + 1 偏移，padding(late_group_id<0) 位置置 0
        valid_mask = late_group_id >= 0
        pid_data = (
            self._build_pid_data(question, valid_mask) if self.model.n_pid > 0 else None
        )

        # 模型前向传播
        preds = self.model(sequence, response, rgaps, sgaps, pcounts, pid_data)

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
