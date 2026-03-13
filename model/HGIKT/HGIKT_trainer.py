"""HGIKT 模型训练器。

定义 HGIKT 模型特定的训练逻辑。
"""

from typing import Any

import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


@register_model_params("HGIKT")
class HGIKTModelParams(BaseParamConfig):
    """HGIKT 模型参数配置。"""

    def define_params(self) -> tuple[str, dict]:
        group_name = "HGIKT Parameters"
        params = {
            "hidden_dim": {
                "type": int,
                "default": 250,
                "short": "hd",
                "help": "Hidden layer dimension",
            },
            "n_hop": {
                "type": int,
                "default": 4,
                "short": "nh",
                "help": "Number of GNN hops",
            },
            "heads": {
                "type": int,
                "default": 1,
                "short": "hs",
                "help": "Number of attention heads",
            },
            "lstm_layers": {
                "type": int,
                "default": 1,
                "short": "ll",
                "help": "Number of LSTM layers",
            },
            "history_neighbour": {
                "type": int,
                "default": 5,
                "short": "hn",
                "help": "History neighbor count",
            },
            "att_bound": {
                "type": float,
                "default": 0.1,
                "short": "ab",
                "help": "Attention bound",
            },
            "num_difficulty_clusters": {
                "type": int,
                "default": 5,
                "help": "Number of difficulty clusters for weighted hypergraph",
            },
            "epochs": {
                "type": int,
                "default": 120,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 0.0003,
                "short": "lr",
                "help": "Learning rate for optimizer",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch",
            },
            "dropout": {
                "type": float,
                "default": 0.25,
                "help": "Dropout rate",
            },
            "weight_decay": {
                "type": float,
                "default": 0.00001,
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


@TRAINERS.register("HGIKT")
class HGIKTTrainer(BaseTrainer):
    """HGIKT 模型训练器 - 使用 Fluent API。

    负责初始化 HGIKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        args: 模型参数配置
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def __init__(
        self,
        args: Any = None,
        data_src: Any = None,
        exp_manager: Any = None,
    ) -> None:
        # 1. 准备数据
        from model.HGIKT import HGIKTModelData

        model_data = HGIKTModelData(data_src)
        data_dict = model_data.prepare_data(args)

        # 解包数据
        train_dataset = data_dict["train_dataset"]
        val_dataset = data_dict["val_dataset"]
        self.hypergraph = data_dict["skill_hypergraph"]
        self.hetero_graph = data_dict["hetero_graph"]
        self.question_skill_matrix = data_dict["question_skill_matrix"]

        # 2. 初始化模型
        from model.HGIKT.HGIKT_model import HGIKT

        logger.info("Initializing HGIKT model...")
        model = HGIKT(args, data_src.get_metadata(), self.hetero_graph.metadata())

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
            batch_size=args.batch_size,
        ).with_optimization(
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            early_stopping=early_stopping_cfg,
        ).with_experiment(
            exp_manager=exp_manager,
            hyperparams=args,
            model_name="HGIKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

        # 8. 将静态数据移动到设备
        self.hetero_graph = self.hetero_graph.to(self.device_)
        self.hypergraph = self.hypergraph.to(self.device_)
        self.question_skill_matrix = self.question_skill_matrix.to(self.device_)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """HGIKT 前向传播，使用基类辅助方法统一处理数据移动和预测生成。

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
        # 模型在时刻 t 的输出预测的是 t+1 的标签
        y_hat_full = self.model(
            sequence,
            response,
            mask,
            self.hetero_graph,
            self.hypergraph,
            self.question_skill_matrix,
        )  # [B, S]

        # 提取有效位置的预测和标签
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=True
        )

        # 处理空批次
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # 生成二分类预测
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
        }
