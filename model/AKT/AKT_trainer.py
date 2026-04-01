"""AKT 模型训练器"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("AKT")
class AKTModelParams(BaseParamConfig):
    """AKT 模型参数配置

    Args:
        d_model: 模型隐藏维度
        n_blocks: Transformer块数量
        num_attn_heads: 注意力头数量
        dropout: Dropout概率
        d_ff: 前馈网络维度
        final_fc_dim: 最终全连接层维度
        kq_same: Key和Query是否使用相同的线性变换
        separate_qa: 是否使用独立的QA嵌入
        l2: L2正则化系数
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        batch_size: 批次大小
    """

    def define_params(self) -> tuple[str, dict]:
        """定义模型参数"""
        group_name = "AKT Parameters"
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
                "help": "Final fully connected layer dimension",
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
                "help": "L2 regularization coefficient",
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
                "help": "Weight decay (L2 regularization) for optimizer",
            },
            "batch_size": {
                "type": int,
                "default": 64,
                "short": "bs",
                "help": "Batch size for training",
            },
        }
        return group_name, params


@TRAINERS.register("AKT")
class AKTTrainer(BaseTrainer):
    """AKT 模型训练器

    负责初始化AKT模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        args: 模型参数配置
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        # 准备数据
        from model.AKT.AKT_data import AKTModelData

        model_data = AKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        # 初始化模型
        from model.AKT.AKT_model import AKT

        logger.info("Initializing AKT model...")
        metadata = data_src.get_metadata()
        n_pid = metadata.get("num_questions", 0)  # 获取题目数量作为n_pid

        if n_pid > 0:
            logger.info(f"AKT: Using Problem ID (Rasch model) with {n_pid} questions")
        else:
            logger.info("AKT: Problem ID not available, using skill-only model")

        model = AKT(
            num_c=metadata["num_skills"],
            n_pid=n_pid,
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            dropout=args.dropout,
            d_ff=args.d_ff,
            final_fc_dim=args.final_fc_dim,
            num_attn_heads=args.num_attn_heads,
            kq_same=args.kq_same,
            separate_qa=bool(args.separate_qa),
            l2=args.l2,
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
            model_name="AKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """AKT 前向传播。

        AKT预测语义：
        - y_hat[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]
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

        # 检查模型是否使用Problem ID
        use_pid = self.model.n_pid > 0

        # 如果使用Problem ID，需要从sequence中获取problem_id
        # 注意：当前数据集中sequence是skill_id，需要额外传递problem_id
        # 如果数据集没有problem_id，传入None
        pid_data = sequence if use_pid else None

        # 模型前向传播
        y_hat_full, c_reg_loss = self.model(
            sequence, response, mask, pid_data
        )  # [B, S]

        # 提取有效位置的预测和标签
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=False
        )

        # 处理空批次
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # 生成二分类预测
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        result = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

        # 如果使用Rasch模型，添加正则化损失
        if use_pid:
            result["c_reg_loss"] = c_reg_loss

        return result

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估。

        数据格式说明：
        - sequence: [技能历史, 目标技能]
        - response: [历史标签, 0]  # 目标位置 response=0 占位
        - mask: [0, ..., 0, 1]  # 只有最后一个位置需要预测
        - late_group_id: [g1, ..., gN]  # 最后一个位置是当前题目的 group_id
        - true_labels: [历史标签, 真实标签]  # 用于评估

        AKT 预测语义：
        - y_hat[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]
        - 模型直接输出对齐的预测，无需额外对齐
        """
        sequence, response, mask, late_group_id, true_labels = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        # 检查模型是否使用Problem ID
        use_pid = self.model.n_pid > 0
        pid_data = sequence if use_pid else None

        # 模型前向传播
        y_hat_full, _ = self.model(sequence, response, mask, pid_data)  # [B, S]

        # AKT 预测对齐
        # y_hat[:, t] 直接预测 response[t]，无需额外对齐
        # 使用 mask 筛选需要预测的位置（只有 mask=1 的位置）
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
        """计算损失，包含BCE损失和Rasch正则化损失（如果使用Problem ID）"""
        y_hat = outputs["y_hat"]
        y_label = outputs["y_label"]
        bce_loss = self.loss(y_hat, y_label)

        # 添加Rasch模型正则化损失
        if "c_reg_loss" in outputs and self.model.n_pid > 0:
            return bce_loss + outputs["c_reg_loss"]

        return bce_loss
