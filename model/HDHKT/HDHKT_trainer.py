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
    """HDHKT 模型配置。

    Args:
        hidden_dim: Hidden layer dimension.
        n_hop: Number of GNN hops.
        heads: Number of attention heads.
        lstm_layers: Number of LSTM layers.
        history_neighbour: History neighbor count.
        att_bound: Attention bound.
        num_difficulty_clusters: Number of difficulty clusters for weighted hypergraph.
        use_hetero_graph: Ablation: use the heterogeneous graph.
        use_sa_relation: Ablation: use the skill-assignment relation.
        use_qt_relation: Ablation: use the question-template relation.
        use_hypergraph: Ablation: use the difficulty-aware hypergraph.
        use_edge_weights: Ablation: weight hyperedges by difficulty.
        use_difficulty_clustering: Ablation: cluster questions by difficulty per skill.
        fusion_mode: Ablation: view fusion strategy, one of moe/sum/cat/wgt.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        dropout: Dropout rate.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    hidden_dim: int = field(
        default=250,
        metadata={"optuna": {"type": "int", "low": 128, "high": 512}},
    )
    n_hop: int = field(
        default=4,
        metadata={"optuna": {"type": "int", "low": 2, "high": 6}},
    )
    heads: int = 1
    lstm_layers: int = 1
    history_neighbour: int = 5
    att_bound: float = 0.1
    num_difficulty_clusters: int = 5
    # --- Ablation switches (defaults = the full model; excluded from Optuna) ---
    use_hetero_graph: bool = True
    use_sa_relation: bool = True
    use_qt_relation: bool = True
    use_hypergraph: bool = True
    use_edge_weights: bool = True
    use_difficulty_clustering: bool = True
    fusion_mode: str = "moe"
    epochs: int = 120
    learning_rate: float = field(
        default=0.0003,
        metadata={
            "optuna": {"type": "float", "low": 0.00001, "high": 0.001, "log": True}
        },
    )
    lr_decay: float | None = None
    dropout: float = field(
        default=0.25,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    weight_decay: float = field(
        default=0.00001,
        metadata={
            "optuna": {"type": "float", "low": 1e-06, "high": 0.0001, "log": True}
        },
    )
    batch_size: int = field(
        default=64,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
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
        self.skill_ids_per_question = data_dict["skill_ids_per_question"]

        from model.HDHKT.HDHKT_model import HDHKT

        logger.info("Initializing HDHKT model...")
        m = rc.model
        model = HDHKT(
            data_metadata=data_src.get_metadata(),
            hetero_metadata=(
                self.hetero_graph.metadata() if m.use_hetero_graph else None
            ),
            hidden_dim=m.hidden_dim,
            n_hop=m.n_hop,
            heads=m.heads,
            lstm_layers=m.lstm_layers,
            dropout=m.dropout,
            history_neighbour=m.history_neighbour,
            att_bound=m.att_bound,
            use_hetero_graph=m.use_hetero_graph,
            use_sa_relation=m.use_sa_relation,
            use_qt_relation=m.use_qt_relation,
            use_hypergraph=m.use_hypergraph,
            fusion_mode=m.fusion_mode,
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
        if self.hetero_graph is not None:
            self.hetero_graph = self.hetero_graph.to(device)
        if self.hypergraph is not None:
            self.hypergraph = self.hypergraph.to(device)
            _ = self.hypergraph.L_HGNN
        self.question_skill_matrix = self.question_skill_matrix.to(device)
        self.skill_ids_per_question = self.skill_ids_per_question.to(device)

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
            self.skill_ids_per_question,
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
