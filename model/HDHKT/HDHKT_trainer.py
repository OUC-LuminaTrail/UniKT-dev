"""HDHKT 模型训练器。

定义 HDHKT 模型特定的训练逻辑。
"""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("HDHKT")
class HDHKTConfig(ModelConfig):
    """HDHKT 模型配置。"""

    hidden_dim: int = field(
        default=250, metadata={"help": "Hidden layer dimension", "short": "hd"}
    )
    n_hop: int = field(
        default=4, metadata={"help": "Number of GNN hops", "short": "nh"}
    )
    heads: int = field(
        default=1, metadata={"help": "Number of attention heads", "short": "hs"}
    )
    lstm_layers: int = field(
        default=1, metadata={"help": "Number of LSTM layers", "short": "ll"}
    )
    history_neighbour: int = field(
        default=5, metadata={"help": "History neighbor count", "short": "hn"}
    )
    att_bound: float = field(
        default=0.1, metadata={"help": "Attention bound", "short": "ab"}
    )
    num_difficulty_clusters: int = field(
        default=5,
        metadata={"help": "Number of difficulty clusters for weighted hypergraph"},
    )
    epochs: int = field(
        default=120, metadata={"help": "Number of training epochs", "short": "ep"}
    )
    learning_rate: float = field(
        default=0.0003, metadata={"help": "Learning rate for optimizer", "short": "lr"}
    )
    lr_decay: float | None = field(
        default=None, metadata={"help": "Learning rate decay factor per epoch"}
    )
    dropout: float = field(default=0.25, metadata={"help": "Dropout rate"})
    weight_decay: float = field(
        default=0.00001,
        metadata={
            "help": "Weight decay (L2 regularization) for optimizer",
            "short": "wd",
        },
    )
    batch_size: int = field(
        default=64, metadata={"help": "Batch size for training", "short": "bs"}
    )


@register_trainer("HDHKT")
class HDHKTTrainer(BaseTrainer):
    """HDHKT 模型训练器。

    负责初始化 HDHKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.HDHKT.HDHKT_data import HDHKTModelData

        model_data = HDHKTModelData(data_src)
        data_dict = model_data.prepare_data(rc)

        train_dataset = data_dict["train_dataset"]
        val_dataset = data_dict["val_dataset"]
        test_dataset = data_dict.get("test_dataset")
        self.hypergraph = data_dict["skill_hypergraph"]
        self.hetero_graph = data_dict["hetero_graph"]
        self.question_skill_matrix = data_dict["question_skill_matrix"]

        from model.HDHKT.HDHKT_model import HDHKT

        logger.info("Initializing HDHKT model...")
        m = rc.model
        model = HDHKT(
            data_metadata=data_src.get_metadata(),
            hetero_metadata=self.hetero_graph.metadata(),
            hidden_dim=m.hidden_dim,
            n_hop=m.n_hop,
            heads=m.heads,
            lstm_layers=m.lstm_layers,
            dropout=m.dropout,
            history_neighbour=m.history_neighbour,
            att_bound=m.att_bound,
        )

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        device = (
            torch.device(rc.general.device) if rc.general.device else self._try_gpu()
        )
        self.hetero_graph = self.hetero_graph.to(device)
        self.hypergraph = self.hypergraph.to(device)
        self.question_skill_matrix = self.question_skill_matrix.to(device)

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """HDHKT 前向传播，使用基类辅助方法统一处理数据移动和预测生成。

        Args:
            batch_data: 包含 (sequence, response, mask) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # Model output at step t predicts the label at step t+1
        y_hat_full = self.model(
            sequence,
            response,
            mask,
            self.hetero_graph,
            self.hypergraph,
            self.question_skill_matrix,
        )  # [B, S]

        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": torch.sigmoid(y_hat),
        }
