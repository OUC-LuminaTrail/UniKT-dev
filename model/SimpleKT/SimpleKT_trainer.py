"""SimpleKT 模型训练器模块"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("SimpleKT")
class SimpleKTModelParams(BaseParamConfig):
    """SimpleKT 模型参数配置

    Args:
        d_model: 模型维度
        n_blocks: Transformer 块数量
        n_heads: 注意力头数量
        dropout: Dropout 概率
        d_ff: 前馈网络维度
        kq_same: 是否共享 key 和 query 的权重
        separate_qa: 是否使用独立的交互嵌入
        final_fc_dim: 第一层全连接层维度
        final_fc_dim2: 第二层全连接层维度
    """

    def define_params(self) -> tuple[str, dict]:
        """定义模型参数"""
        group_name = "SimpleKT Parameters"
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
                "default": 256,
                "help": "First fully connected layer dimension in output",
            },
            "final_fc_dim2": {
                "type": int,
                "default": 256,
                "help": "Second fully connected layer dimension in output",
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
                "default": 1e-5,
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


@TRAINERS.register("SimpleKT")
class SimpleKTTrainer(BaseTrainer):
    """SimpleKT 模型训练器

    负责初始化 SimpleKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        args: 模型参数配置
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        # 准备数据
        from model.SimpleKT.SimpleKT_data import SimpleKTModelData

        model_data = SimpleKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        # 初始化模型
        from model.SimpleKT.SimpleKT_model import SimpleKT

        logger.info("Initializing SimpleKT model...")
        metadata = data_src.get_metadata()
        model = SimpleKT(
            num_skills=metadata["num_skills"],
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
            model_name="SimpleKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """SimpleKT 前向传播

        SimpleKT 预测语义：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - 第一个位置：y_hat[:, 0] 基于 sequence[0:1] 和空历史预测 response[0]
        - y_hat[:, t] 直接对应 response[t]，需要对齐

        Args:
            batch_data: 包含 (sequence, response, mask) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        # 解包数据并移动到设备
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # 模型前向传播
        y_hat_full = self.model(sequence, response, mask)  # [B, S]

        # 提取有效位置的预测和标签
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=False
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

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估

        数据格式说明：
        - sequence: [技能历史, 目标技能]
        - response: [历史标签, 0]  # 目标位置 response=0 用于避免数据泄露
        - mask: [0, ..., 0, 1]  # 只有最后一个位置需要预测
        - late_group_id: [g1, ..., gN]  # 最后一个位置是当前题目的 group_id
        - true_labels: [历史标签, 真实标签]  # 用于评估

        SimpleKT 预测语义：
        - y_hat[:, t] 基于 sequence[0:t+1] 和 response[0:t] 预测 response[t]
        - 模型内部已经使用移位后的 response 作为 target，所以 y_hat[:, t] 直接对应 response[t]
        - 测试时 response[:, -1] = 0（占位），模型会忽略（使用移位后的目标）
        """
        sequence, response, mask, late_group_id, true_labels = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        # 模型前向传播
        y_hat_full = self.model(sequence, response, mask)  # [B, S]

        # SimpleKT 预测对齐
        # y_hat[:, t] 预测的是 response[t]
        # 使用 mask 筛选需要预测的位置（只有 mask=1 的位置需要预测）
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
