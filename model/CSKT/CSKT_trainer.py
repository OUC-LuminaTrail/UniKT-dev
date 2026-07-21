"""CSKT 模型训练器"""

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("CSKT")
class CSKTConfig(ModelConfig):
    """CSKT model configuration.

    Args:
        d_model: Hidden dimension of the model.
        num_blocks: Number of transformer blocks.
        num_attn_heads: Number of attention heads.
        dropout: Dropout probability.
        d_ff: Feed-forward network dimension.
        r: Cone attention radius parameter.
        gamma: Cone attention temperature parameter.
        kq_same: Whether key and query share the linear projection (1=yes, 0=no).
        separate_qa: Whether to use a separate interaction embedding (1=yes, 0=no).
        final_fc_dim: First fully connected layer dimension in output head.
        final_fc_dim2: Second fully connected layer dimension in output head.
        emb_type: Embedding type ('qid' vector difficulty, 'qid_scalar' scalar).
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        lr_decay: Learning rate decay factor per epoch.
        weight_decay: Weight decay (L2 regularization) for optimizer.
        batch_size: Batch size for training.
    """

    d_model: int = 128
    num_blocks: int = 2
    num_attn_heads: int = 4
    dropout: float = 0.1
    d_ff: int = 256
    r: float = 0.6
    gamma: float = 1.0
    kq_same: int = 1
    separate_qa: int = 0
    final_fc_dim: int = 512
    final_fc_dim2: int = 256
    emb_type: str = "qid"
    epochs: int = 100
    learning_rate: float = 1e-4
    lr_decay: float | None = None
    weight_decay: float = 0.0
    batch_size: int = 64


@register_trainer("CSKT")
class CSKTTrainer(BaseTrainer):
    """CSKT 模型训练器

    负责初始化 CSKT 模型、优化器和训练数据，并实现前向传播逻辑。

    Args:
        rc: RunConfig (OmegaConf DictConfig)
        data_src: 数据源实例
        exp_manager: 实验管理器（可选）
    """

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.CSKT.CSKT_data import CSKTModelData

        model_data = CSKTModelData(data_src)
        train_dataset, val_dataset, test_dataset = model_data.prepare_data(rc)

        from model.CSKT.CSKT_model import CSKT

        logger.info("Initializing CSKT model...")
        metadata = data_src.get_metadata()
        # Question count feeds the Rasch question-difficulty term (n_pid)
        n_pid = metadata.get("num_questions", 0)

        if n_pid > 0:
            logger.info(f"CSKT: Using Rasch question difficulty with {n_pid} questions")
        else:
            logger.info("CSKT: Question ID not available, using skill-only model")

        m = rc.model
        model = CSKT(
            num_c=metadata["num_skills"],
            max_seq_len=rc.data.max_seq_len,
            n_pid=n_pid,
            d_model=m.d_model,
            num_blocks=m.num_blocks,
            dropout=m.dropout,
            d_ff=m.d_ff,
            num_attn_heads=m.num_attn_heads,
            r=m.r,
            gamma=m.gamma,
            kq_same=bool(m.kq_same),
            final_fc_dim=m.final_fc_dim,
            final_fc_dim2=m.final_fc_dim2,
            separate_qa=bool(m.separate_qa),
            emb_type=m.emb_type,
        )

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )

        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )

        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=test_dataset,
        )

    def _build_pid_data(
        self,
        question: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """构建 Rasch pid 数据（0 保留给 padding）。"""
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """CSKT 前向传播。

        Args:
            batch_data: 包含 (sequence, response, mask, question) 的元组

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        sequence, response, mask, question = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        pid_data = self._build_pid_data(question, mask) if use_pid else None

        y_hat_full = self.model(sequence, response, pid_data)  # [B, S]

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, same_position=True
        )

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.5)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_score": y_hat,
            "y_prob": y_hat,
        }

    def test_forward_pass(self, batch_data):
        """测试前向传播，支持 windowlateauc_mean 评估。

        batch_data: (sequence, response, mask, late_group_id, true_labels, question)
        """
        sequence, response, mask, late_group_id, true_labels, question = batch_data

        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)
        late_group_id = self._move_tensor_to_device(late_group_id)
        true_labels = self._move_tensor_to_device(true_labels)
        question = self._move_tensor_to_device(question)

        use_pid = self.model.n_pid > 0
        valid_mask = late_group_id >= 0
        pid_data = self._build_pid_data(question, valid_mask) if use_pid else None

        y_hat_full = self.model(sequence, response, pid_data)  # [B, S]

        # Same-position alignment: preds[:, t] predicts response[t]
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
