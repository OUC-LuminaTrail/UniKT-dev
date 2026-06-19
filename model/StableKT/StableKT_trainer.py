"""StableKT 模型训练器模块"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("StableKT")
class StableKTModelParams(BaseParamConfig):
    """StableKT 模型参数配置

    Args:
        d_model: 模型维度
        n_blocks: Transformer 块数量
        n_heads: 注意力头数量（必须为偶数）
        dropout: Dropout 概率
        d_ff: 前馈网络维度
        kq_same: 是否共享 key 和 query 的权重
        separate_qa: 是否使用独立的交互嵌入
        final_fc_dim: 第一层全连接层维度
        final_fc_dim2: 第二层全连接层维度
        emb_type: 嵌入类型
        r: 半影锥半径
        gamma: 半影锥温度参数
        num_buckets: T5 相对位置偏置分桶数
        max_distance: T5 相对位置偏置最大距离
    """

    def define_params(self) -> tuple[str, dict]:
        """定义模型参数"""
        group_name = "StableKT Parameters"
        params = {
            "d_model": {
                "type": int,
                "default": 256,
                "help": "Dimension of the model",
            },
            "n_blocks": {
                "type": int,
                "default": 2,
                "help": "Number of transformer blocks",
            },
            "n_heads": {
                "type": int,
                "default": 4,
                "help": "Number of attention heads (must be even for HAKT)",
            },
            "dropout": {
                "type": float,
                "default": 0.1,
                "help": "Dropout probability",
            },
            "d_ff": {
                "type": int,
                "default": 256,
                "help": "Dimension of feed-forward network",
            },
            "kq_same": {
                "type": int,
                "default": 1,
                "help": "Whether to share key and query weights (1 for yes, 0 for no)",
            },
            "separate_qa": {
                "type": int,
                "default": 0,
                "help": "Whether to use separate interaction embedding (1 for yes, 0 for no)",
            },
            "final_fc_dim": {
                "type": int,
                "default": 512,
                "help": "First fully connected layer dimension in output",
            },
            "final_fc_dim2": {
                "type": int,
                "default": 256,
                "help": "Second fully connected layer dimension in output",
            },
            "emb_type": {
                "type": str,
                "default": "qid",
                "help": "Embedding type: qid, qid_woha, qid_sin, qid_t5, qid_rotary, qid_wha, etc.",
            },
            "r": {
                "type": float,
                "default": 1.0,
                "help": "Penumbral cone radius for HAKT attention",
            },
            "gamma": {
                "type": float,
                "default": 1.0,
                "help": "Penumbral cone temperature parameter for HAKT attention",
            },
            "num_buckets": {
                "type": int,
                "default": 32,
                "help": "Number of buckets for T5 relative position bias",
            },
            "max_distance": {
                "type": int,
                "default": 100,
                "help": "Maximum distance for T5 relative position bias",
            },
            "epochs": {
                "type": int,
                "default": 100,
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
                "default": 0,
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


@TRAINERS.register("StableKT")
class StableKTTrainer(BaseTrainer):
    """StableKT 模型训练器

    负责初始化 StableKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        args: 模型参数配置
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        # 准备数据
        from model.StableKT.StableKT_data import StableKTModelData

        model_data = StableKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        # 初始化模型
        from model.StableKT.StableKT_model import StableKT

        logger.info("Initializing StableKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata["num_questions"]
        logger.info(f"StableKT: Using Problem ID (Rasch model) with {n_pid} questions")

        model = StableKT(
            num_skills=metadata["num_skills"],
            n_pid=n_pid,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            seq_len=args.max_seq_len,
            kq_same=args.kq_same,
            separate_qa=bool(args.separate_qa),
            final_fc_dim=args.final_fc_dim,
            final_fc_dim2=args.final_fc_dim2,
            emb_type=args.emb_type,
            r=args.r,
            gamma=args.gamma,
            num_buckets=args.num_buckets,
            max_distance=args.max_distance,
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
            model_name="StableKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """StableKT 前向传播

        StableKT 预测语义：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - 第一个位置：y_hat[:, 0] 基于 sequence[0:1] 和空历史预测 response[0]
        - y_hat[:, t] 直接对应 response[t]，需要对齐

        Args:
            batch_data: 包含 (sequence, response, mask, question) 的元组（question 必选，用于Rasch pid）

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        # 解包数据并移动到设备
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        pid_data = self._build_pid_data(question, mask)

        # 模型前向传播
        y_hat_full = self.model(sequence, response, mask, pid_data)

        # 归一化：y[t] 预测 response[t] → y[t] 预测 response[t+1]
        y_norm = torch.cat(
            [y_hat_full[:, 1:], torch.zeros_like(y_hat_full[:, :1])], dim=1
        )

        # 提取有效位置的预测和标签
        y_hat, y_label, _ = self._extract_valid_predictions(y_norm, response, mask)

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

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估

        数据格式说明：
        - sequence: [技能历史, 目标技能]
        - response: [历史标签, 0]  # 目标位置 response=0 用于避免数据泄露
        - mask: [0, ..., 0, 1]  # 只有最后一个位置需要预测
        - late_group_id: [g1, ..., gN]  # 最后一个位置是当前题目的 group_id
        - true_labels: [历史标签, 真实标签]  # 用于评估
        - question: [题目历史, 目标题目]  # 用于Rasch pid

        StableKT 预测语义：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - y_hat[:, t] 直接对应 response[t]
        """
        sequence, response, mask, late_group_id, true_labels, question = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask)

        # 模型前向传播
        y_hat_full = self.model(sequence, response, mask, pid_data)

        # StableKT 预测对齐
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
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        return self.loss(y_hat, y_label)
