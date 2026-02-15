"""
GIKT 模型训练器
定义 GIKT 模型特定的训练逻辑
"""

import torch
from utils.training import BaseTrainer
from utils.core import TRAINERS, get_logger
from utils.config import register_model_params, BaseParamConfig

logger = get_logger(__name__)


@register_model_params("HGIKT")
class HGIKTModelParams(BaseParamConfig):
    """HGIKT model-specific parameters."""

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
    """
    HGIKT模型训练器
    """

    def __init__(
        self,
        args=None,
        data_src=None,
        exp_manager=None,
    ):
        # 构建数据
        from model.HGIKT import HGIKTModelData

        model_data = HGIKTModelData(data_src)
        data_dict = model_data.prepare_data(args)

        # 解包数据
        train_data = data_dict["train_dataloader"]
        val_data = data_dict["val_dataloader"]
        self.hypergraph = data_dict["skill_hypergraph"]
        self.hetero_graph = data_dict["hetero_graph"]
        self.question_skill_matrix = data_dict["question_skill_matrix"]

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

        # 将静态数据移动到设备
        self.hetero_graph = self.hetero_graph.to(self.device_)
        self.hypergraph = self.hypergraph.to(self.device_)
        self.question_skill_matrix = self.question_skill_matrix.to(self.device_)

    def init_model(self, args, data_src):
        from model.HGIKT.HGIKT_model import HGIKT

        logger.info("Initializing HGIKT model...")
        model = HGIKT(args, data_src.get_metadata(), self.hetero_graph.metadata())

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
        """HGIKT 前向传播，使用基类辅助方法统一处理数据移动和预测生成"""
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
        }
