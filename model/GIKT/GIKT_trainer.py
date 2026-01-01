"""
GIKT 模型训练器
定义 GIKT 模型特定的训练逻辑
"""

import torch
from utils.training import BaseTrainer
from utils.core import TRAINERS, get_logger
from utils.config import register_model_params, BaseParamConfig

logger = get_logger(__name__)

__all__ = ["GIKTTrainer", "GIKTModelParams"]


@register_model_params("GIKT")
class GIKTModelParams(BaseParamConfig):
    """GIKT model-specific parameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "GIKT Parameters"
        params = {
            "hidden_dim": {
                "type": int,
                "default": 100,
                "short": "hd",
                "help": "Hidden layer dimension (default: 100)",
            },
            "embedding_dim": {
                "type": int,
                "default": 100,
                "short": "ed",
                "help": "Embedding dimension (default: 100)",
            },
            "lstm_layers": {
                "type": int,
                "default": 2,
                "short": "ll",
                "help": "Number of LSTM layers (default: 2)",
            },
            "n_hop": {
                "type": int,
                "default": 3,
                "short": "nh",
                "help": "Number of GNN hops (default: 3)",
            },
            "heads": {
                "type": int,
                "default": 2,
                "short": "hs",
                "help": "Number of attention heads (default: 2)",
            },
            "history_neighbour": {
                "type": int,
                "default": 5,
                "short": "hn",
                "help": "History neighbor count (default: 5)",
            },
            "att_bound": {
                "type": float,
                "default": 0.2,
                "short": "ab",
                "help": "Attention bound (default: 0.2)",
            },
            "dropout": {
                "type": float,
                "default": 0.4,
                "short": "dp",
                "help": "Dropout rate (default: 0.4)",
            },
            "epochs": {
                "type": int,
                "default": 150,
                "short": "ep",
                "help": "Number of training epochs (default: 150)",
            },
            "learning_rate": {
                "type": float,
                "default": 0.001,
                "short": "lr",
                "help": "Learning rate for optimizer (default: 0.001)",
            },
            "lr_decay": {
                "type": float,
                "default": None,
                "help": "Learning rate decay factor per epoch (default: None)",
            },
            "weight_decay": {
                "type": float,
                "default": 1e-4,
                "short": "wd",
                "help": "Weight decay (L2 regularization) for optimizer (default: 0.0001)",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size for training (default: 128)",
            },
        }

        return group_name, params


@TRAINERS.register("GIKT")
class GIKTTrainer(BaseTrainer):
    """
    GIKT模型训练器
    """

    def __init__(
        self,
        args=None,
        data_src=None,
        exp_manager=None,
    ):
        # 构建数据
        from model.GIKT import GIKTModelData

        model_data = GIKTModelData(data_src)
        train_data, val_data, self.graph, self.question_skill_matrix = (
            model_data.prepare_data(args)
        )
        model, opt, loss, lr_scheduler = self.init_model(args, data_src)
        super().__init__(
            model=model,
            epochs=args.epochs,
            opt=opt,
            loss=loss,
            train_data=train_data,
            val_data=val_data,
            lr_scheduler=lr_scheduler,
            hyperparams=args,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            seed=args.seed,
            exp_manager=exp_manager,
        )

        # 将静态数据移动到设备中
        self.graph = self.graph.to(self.device_)  # 图
        self.question_skill_matrix = self.question_skill_matrix.to(
            self.device_
        )  # 问题-技能矩阵

    def init_model(self, args, data_src):
        from model.GIKT.GIKT_model import GIKT

        logger.info("Initializing GIKT model...")
        model = GIKT(args, data_src.get_metadata())

        # 二分类交叉熵损失
        loss_fn = torch.nn.BCEWithLogitsLoss()
        # 优化器
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        # 学习率调度器
        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        return model, optimizer, loss_fn, lr_scheduler

    def forward_pass(self, batch_data):
        """GIKT 前向传播，使用基类辅助方法统一处理数据移动和预测生成"""
        # 解包数据并移动到设备
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        # 模型前向传播
        # 模型在时刻 t 的输出预测的是 t+1 的标签
        y_hat_full = self.model(
            sequence, response, mask, self.graph, self.question_skill_matrix
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
        }
