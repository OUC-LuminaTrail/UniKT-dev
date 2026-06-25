"""GKT (Graph-based Knowledge Tracing) 训练器模块"""

from typing import Any

import numpy as np
import torch

from utils.config import BaseParamConfig, EarlyStoppingConfig, register_model_params
from utils.core import TRAINERS, get_logger
from utils.training import BaseTrainer

logger = get_logger(__name__)


def build_dense_graph(num_c: int) -> torch.Tensor:
    """构建全连接图

    Args:
        num_c: 概念数量

    Returns:
        图邻接矩阵，形状为 [num_c, num_c]
    """
    graph = 1.0 / (num_c - 1) * np.ones((num_c, num_c))
    np.fill_diagonal(graph, 0)
    return torch.from_numpy(graph).float()


def build_transition_graph(sequences: list, num_c: int) -> torch.Tensor:
    """构建转移图

    基于概念序列的转移模式构建邻接矩阵

    Args:
        sequences: 概念序列列表
        num_c: 概念数量

    Returns:
        图邻接矩阵，形状为 [num_c, num_c]
    """
    graph = np.zeros((num_c, num_c))

    for seq in sequences:
        # 过滤掉填充值
        valid_seq = [s for s in seq if s >= 0 and s < num_c]
        for i in range(len(valid_seq) - 1):
            pre = valid_seq[i]
            next_c = valid_seq[i + 1]
            graph[pre, next_c] += 1

    # 对角线置零
    np.fill_diagonal(graph, 0)

    # 行归一化
    rowsum = np.array(graph.sum(1))

    def inv(x):
        return 1.0 / x if x != 0 else 0.0

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)

    return torch.from_numpy(graph).float()


@register_model_params("GKT")
class GKTModelParams(BaseParamConfig):
    """GKT 模型参数配置

    Args:
        hidden_dim: 隐藏层维度
        embedding_dim: 嵌入维度
        dropout: Dropout概率
        graph_type: 图类型 ("dense" 或 "transition")
    """

    def define_params(self) -> tuple[str, dict]:
        """定义模型参数"""
        group_name = "GKT Parameters"
        params = {
            "hidden_dim": {
                "type": int,
                "default": 32,
                "help": "Hidden dimension of the model",
            },
            "embedding_dim": {
                "type": int,
                "default": 32,
                "help": "Embedding dimension of the model",
            },
            "dropout": {
                "type": float,
                "default": 0.5,
                "help": "Dropout probability",
            },
            "graph_type": {
                "type": str,
                "default": "dense",
                "choices": ["dense", "transition"],
                "help": "Graph type for GKT model",
            },
            "epochs": {
                "type": int,
                "default": 150,
                "short": "ep",
                "help": "Number of training epochs",
            },
            "learning_rate": {
                "type": float,
                "default": 0.001,
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
                "default": 0.0001,
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


@TRAINERS.register("GKT")
class GKTTrainer(BaseTrainer):
    """GKT 模型训练器

    负责初始化GKT模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        args: 模型参数配置
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def __init__(
        self, args: Any = None, data_src: Any = None, exp_manager: Any = None
    ) -> None:
        # 准备数据
        from model.GKT.GKT_data import GKTModelData

        model_data = GKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(args)

        # 获取元数据
        metadata = data_src.get_metadata()
        num_skills = metadata["num_skills"]

        # 构建图
        graph = self._build_graph(args, train_dataset, num_skills)

        # 初始化模型
        from model.GKT.GKT_model import GKT

        logger.info("Initializing GKT model...")
        model = GKT(
            num_c=num_skills,
            hidden_dim=args.hidden_dim,
            emb_size=args.embedding_dim,
            graph_type=args.graph_type,
            graph=graph,
            dropout=args.dropout,
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
            model_name="GKT",
            dataset_name=getattr(args, "dataset", ""),
        ).build()

    def _build_graph(self, args, train_dataset, num_skills: int) -> torch.Tensor:
        """构建图

        Args:
            args: 模型参数
            train_dataset: 训练数据集
            num_skills: 技能数量

        Returns:
            图邻接矩阵
        """
        if args.graph_type == "dense":
            logger.info("Building dense graph...")
            return build_dense_graph(num_skills)
        elif args.graph_type == "transition":
            logger.info("Building transition graph from training data...")
            # 从训练数据构建转移图
            sequences = train_dataset.sequences.tolist()
            return build_transition_graph(sequences, num_skills)
        else:
            logger.warning(f"Unknown graph type: {args.graph_type}, using dense graph")
            return build_dense_graph(num_skills)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """GKT 前向传播

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

        # 模型前向传播：输出 [B, S-1]，y[:, t] 预测 response[:, t+1]（next-item）
        y_hat_full = self._pad_to_full_sequence(self.model(sequence, response, mask))

        # 提取有效位置的预测和标签（pad 到 [B, S] 后用内置 next-item 对齐）
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full,
            response,
            mask,
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
        - sequence: [技能历史, 目标技能]，形状 [B, S]
        - response: [历史标签, 0]  # 目标位置 response=0 避免数据泄露，形状 [B, S]
        - mask: [0, ..., 0, 1]  # 只有最后一个位置需要预测，形状 [B, S]
        - late_group_id: [g1, ..., gN]  # 最后一个位置是当前题目的 group_id，形状 [B, S]
        - true_labels: [历史标签, 真实标签]  # 用于评估，形状 [B, S]

        GKT 预测语义（更新后）：
        - 模型输出 [B, S-1]，其中 y[:, t] 预测 response[:, t+1]
        - 即 y[:, 0] 预测 response[:, 1], ..., y[:, S-2] 预测 response[:, S-1]
        """
        sequence, response, mask, late_group_id, true_labels, _ = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)

        # 模型前向传播
        # 输出形状: [B, S-1]，其中 y[:, t] 预测 response[:, t+1]
        y_hat_full = self.model(sequence, response, mask)

        # ==================== 关键：GKT 预测对齐 ====================
        # 模型输出 [B, S-1]：y[:, t] 预测 response[:, t+1]
        # 所以 y_hat_full[:, t] 对应 true_labels[:, t+1]
        # 需要对齐：y_hat_full 对应 true_labels[:, 1:]

        # y_hat_full 已经是 [B, S-1]，对应 true_labels[:, 1:]
        true_labels_aligned = true_labels[:, 1:]  # [B, S-1]
        mask_aligned = mask[:, 1:]  # [B, S-1]
        group_id_aligned = late_group_id[:, 1:]  # [B, S-1]

        # 使用 mask 筛选需要预测的位置
        y_hat = torch.masked_select(y_hat_full, mask_aligned)
        y_label = torch.masked_select(true_labels_aligned, mask_aligned).float()
        group_ids = torch.masked_select(group_id_aligned, mask_aligned)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
            "group_id": group_ids,
        }
