"""MTKT 模型训练器"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("MTKT")
class MTKTModelParams(BaseParamConfig):
    """MTKT 模型参数配置

    Args:
        d_model: 模型隐藏维度
        n_blocks: Transformer 块数量
        num_attn_heads: 注意力头数量
        dropout: Dropout 概率
        d_ff: CIC 隐藏维度
        final_fc_dim: 输出全连接层维度 1
        final_fc_dim2: 输出全连接层维度 2
        kq_same: Key 和 Query 是否使用相同的线性变换
        separate_qa: 是否使用独立的 QA 嵌入
        l2: L2 正则化系数 (Rasch 模型)
        k1: CIC 卷积核大小 1
        k2: CIC 卷积核大小 2
        num_rgap: 复习间隔桶数量
        num_sgap: 连续间隔桶数量
        num_pcount: 练习次数桶数量
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        batch_size: 批次大小
    """

    def define_params(self) -> tuple[str, dict]:
        """定义模型参数"""
        group_name = "MTKT Parameters"
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
                "default": 256,
                "help": "CIC hidden dimension",
            },
            "final_fc_dim": {
                "type": int,
                "default": 512,
                "help": "Output FC layer dimension 1",
            },
            "final_fc_dim2": {
                "type": int,
                "default": 256,
                "help": "Output FC layer dimension 2",
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
            "l2": {
                "type": float,
                "default": 1e-5,
                "help": "L2 regularization coefficient for Rasch model",
            },
            "k1": {
                "type": int,
                "default": 1,
                "help": "CIC convolution kernel size 1",
            },
            "k2": {
                "type": int,
                "default": 3,
                "help": "CIC convolution kernel size 2",
            },
            "num_rgap": {
                "type": int,
                "default": 100,
                "help": "Number of review gap buckets",
            },
            "num_sgap": {
                "type": int,
                "default": 100,
                "help": "Number of sequential gap buckets",
            },
            "num_pcount": {
                "type": int,
                "default": 15,
                "help": "Number of practice count buckets",
            },
            "epochs": {
                "type": int,
                "default": 150,
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


@TRAINERS.register("MTKT")
class MTKTTrainer(BaseTrainer):
    """MTKT 模型训练器

    负责初始化 MTKT 模型、优化器和训练数据，并实现前向传播逻辑。
    """

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        from model.MTKT.MTKT_data import MTKTModelData
        from model.MTKT.MTKT_model import MTKT

        model_data = MTKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        logger.info("Initializing MTKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)

        if n_pid > 0:
            logger.info(f"MTKT: Using Problem ID (Rasch model) with {n_pid} questions")
        else:
            logger.info("MTKT: Problem ID not available, using skill-only model")

        model = MTKT(
            num_skills=metadata["num_skills"],
            n_pid=n_pid,
            num_rgap=args.num_rgap,
            num_sgap=args.num_sgap,
            num_pcount=args.num_pcount,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            d_ff=args.d_ff,
            kq_same=args.kq_same,
            separate_qa=bool(args.separate_qa),
            l2=args.l2,
            k1=args.k1,
            k2=args.k2,
            num_attn_heads=args.num_attn_heads,
            final_fc_dim=args.final_fc_dim,
            final_fc_dim2=args.final_fc_dim2,
            seq_len=args.max_seq_len,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=args.lr_decay,
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
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="MTKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """构建 Rasch pid 数据, 0 保留给 padding"""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(self, batch_data: tuple) -> dict[str, torch.Tensor]:
        """MTKT 前向传播

        MTKT 预测语义:
        - preds[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]
        - 使用 skip_first=False 直接对齐

        Args:
            batch_data: (sequence, response, mask, question, rgap, sgap, pcount)

        Returns:
            包含 y_hat, y_label, y_predict 的字典
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
            sequence,
            response,
            pid_data,
            rgap,
            sgap,
            pcount,
        )

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full,
            response,
            mask,
            skip_first=False,
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
        """测试前向传播, 支持 windowlateauc_mean 评估

        Windowlate 数据为 6-元组:
            (sequence, response, mask, late_group_id, true_labels, question)
        时间间隔使用零填充。
        """
        sequence, response, mask, late_group_id, true_labels, question = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        # 时间间隔零填充 (windowlate 数据无时间戳)
        rgap = torch.zeros_like(sequence)
        sgap = torch.zeros_like(sequence)
        pcount = torch.zeros_like(sequence)

        use_pid = self.model.n_pid > 0
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else question

        y_hat_full, _ = self.model(
            sequence,
            response,
            pid_data,
            rgap,
            sgap,
            pcount,
        )

        y_hat = torch.masked_select(y_hat_full, mask)
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

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失, 包含 BCE 损失和 Rasch 正则化损失"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            return bce_loss + outputs["c_reg_loss"]

        return bce_loss
