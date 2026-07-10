from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm

from utils.config import (
    BaseParamConfig,
    EarlyStoppingConfig,
    create_optimized_dataloader,
    register_model_params,
)
from utils.core import get_logger, register_trainer
from utils.data_process import DataSource
from utils.training import MultiTrainer, StageComponents, StageConfig

from .ABKT_data import ABKTModelData
from .ABKT_model import GMF, IRT_2, K_CMF

logger = get_logger(__name__)

__all__ = ["ABKTTrainer", "ABKTModelParams"]


@register_model_params("ABKT")
class ABKTModelParams(BaseParamConfig):
    """ABKT model-specific parameters."""

    def define_params(self) -> tuple[str, dict]:
        group_name = "ABKT Parameters"
        params = {
            # Knowledge Module (KM) parameters
            "km_hidden_dim": {
                "type": int,
                "default": 5,
                "short": "kmh",
                "help": "Hidden dimension for knowledge growth in K_CMF (default: 5)",
            },
            "km_guess": {
                "type": float,
                "default": 0.25,
                "short": "kmg",
                "help": "Guess parameter for IRT response function (default: 0.25)",
            },
            "km_lr": {
                "type": float,
                "default": 0.001,
                "short": "kmlr",
                "help": "Learning rate for KM stage (default: 0.001)",
            },
            "km_epochs": {
                "type": int,
                "default": 100,
                "short": "kme",
                "help": "Number of epochs for KM stage (default: 100)",
            },
            "km_patience": {
                "type": int,
                "default": 10,
                "short": "kmp",
                "help": "Early stopping patience for KM stage (default: 10)",
            },
            # Ability Module (AM) parameters
            "am_embedding_dim": {
                "type": int,
                "default": 50,
                "short": "amed",
                "help": "Embedding dimension for GMF (default: 50)",
            },
            "am_lambda": {
                "type": float,
                "default": 0.1,
                "short": "aml",
                "help": "L2 regularization weight for AM (default: 0.1)",
            },
            "am_layer": {
                "type": int,
                "default": 1,
                "short": "amly",
                "help": "Number of GNN layers in GMF (0-3) (default: 1)",
            },
            "am_lr": {
                "type": float,
                "default": 0.0001,
                "short": "amlr",
                "help": "Learning rate for AM stage (default: 0.0001)",
            },
            "am_epochs": {
                "type": int,
                "default": 500,
                "short": "ame",
                "help": "Number of epochs for AM stage (default: 500)",
            },
            "am_patience": {
                "type": int,
                "default": 10,
                "short": "amp",
                "help": "Early stopping patience for AM stage (default: 10)",
            },
            # Common parameters
            "pretrain_clip": {
                "type": float,
                "default": 0.4,
                "short": "pc",
                "help": "Clip value for pretrained predictions (default: 0.4)",
            },
            "combine_mode": {
                "type": str,
                "default": "add",
                "short": "cm",
                "help": "Combine mode for KM and AM predictions: 'add' or 'mul' (default: add)",
            },
            "batch_size": {
                "type": int,
                "default": 128,
                "short": "bs",
                "help": "Batch size for AM training (default: 128)",
            },
            "use_adj": {
                "type": bool,
                "default": True,
                "help": "Whether to use learnable adjacency weights in GMF (default: True)",
            },
        }
        return group_name, params


# ==================== KM 阶段的数据集 ====================


class KMUserDataset(Data.Dataset):
    """KM 阶段的用户序列数据集（每个样本是一个用户的完整答题序列，batch_size=1）。"""

    def __init__(self, train_users: list, train_sequences: dict):
        self.train_users = train_users
        self.train_sequences = train_sequences

    def __len__(self):
        return len(self.train_users)

    def __getitem__(self, idx: int):
        user_id = self.train_users[idx]
        seq_data = self.train_sequences[user_id]
        items = torch.tensor(seq_data[0][0], dtype=torch.long)
        corrects = torch.tensor(seq_data[1][0], dtype=torch.long)
        return user_id, items, corrects


class KMValidationDataset(Data.Dataset):
    """KM 阶段的验证数据集（每个样本是一个测试三元组）。"""

    def __init__(self, test_triplets: list):
        self.test_triplets = test_triplets

    def __len__(self):
        return len(self.test_triplets)

    def __getitem__(self, idx: int):
        user_id, item_id, correct = self.test_triplets[idx]
        return user_id, item_id, correct


# ==================== AM 阶段的数据集 ====================


class AMTripletDataset(Data.Dataset):
    """AM 阶段数据集（每个样本为 [user, item, correct, km_pred, g, w]）。"""

    def __init__(self, triplets: torch.Tensor):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int):
        return self.triplets[idx]


# ==================== 自定义损失函数 ====================


class KMLoss(nn.Module):
    """KM 阶段损失函数：BCE Loss。"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_hat: torch.Tensor, y_label: torch.Tensor) -> torch.Tensor:
        return self.bce(y_hat, y_label)


class AMLoss(nn.Module):
    """AM 阶段损失函数：Boosting Loss + L2 正则化。"""

    def __init__(self, am_lambda: float, combine_mode: str):
        super().__init__()
        self.am_lambda = am_lambda
        self.combine_mode = combine_mode

    def forward(
        self,
        pred: torch.Tensor,
        k_batch: torch.Tensor,
        g_batch: torch.Tensor,
        w_batch: torch.Tensor,
        u_norm: torch.Tensor,
        i_norm: torch.Tensor,
    ) -> torch.Tensor:
        if self.combine_mode == "add":
            loss = -torch.mean(w_batch * torch.pow(pred + g_batch / w_batch, 2))
        elif self.combine_mode == "mul":
            loss = -torch.mean(
                w_batch
                * k_batch
                * torch.pow(pred + g_batch / (w_batch * k_batch) - 1, 2)
            )
        else:
            raise ValueError(f"Unknown combine mode: {self.combine_mode}")

        return loss + self.am_lambda * (u_norm + i_norm)


@register_trainer("ABKT")
class ABKTTrainer(MultiTrainer):
    """ABKT 两阶段训练器。

    - Stage "km": 训练知识模块 K_CMF（按用户序列建模知识状态 + IRT 预测）。
    - Stage "am": 训练能力模块 GMF（图矩阵分解 + boosting 残差修正）。
    """

    def __init__(
        self,
        args=None,
        data_src: DataSource | None = None,
        exp_manager=None,
    ):
        """初始化 ABKT 训练器。

        Args:
            args: 命令行参数。
            data_src: 数据源。
            exp_manager: 实验管理器（可选；提供时会在构造末尾自动 build）。
        """
        device = getattr(args, "device", None) if args is not None else None
        super().__init__(device=device)

        self.args = args

        logger.info("Preparing ABKT data...")
        self.model_data = ABKTModelData(data_src)
        self.data = self.model_data.prepare_data(args)

        # 模型引用与跨阶段数据，在阶段构建器 / on_stage_complete 中填充
        self.km_model: K_CMF | None = None
        self.am_model: GMF | None = None
        self.Q_matrix: torch.Tensor | None = None
        self.am_train_triplets: torch.Tensor | None = None
        self.am_test_triplets: torch.Tensor | None = None
        self.adj_norm: torch.Tensor | None = None

        # 提供实验管理器时立即完成基础设施构建（与 train.py 的调用约定一致）
        if exp_manager is not None:
            self.with_experiment(
                exp_manager=exp_manager,
                hyperparams=args,
                model_name="ABKT",
                dataset_name=getattr(args, "dataset", "") if args else "",
            ).build()

    def build_stages(self) -> list[StageConfig]:
        """声明两个训练阶段。构建器延迟执行，am 可依赖 km 的输出。"""
        return [
            StageConfig(name="km", build=self._build_km),
            StageConfig(name="am", build=self._build_am),
        ]

    def _build_km(self) -> StageComponents:
        """构建 KM 阶段：K_CMF 模型 + 单用户序列数据 + BCE 损失。"""
        Q_matrix = self.data["Q_matrix"].to(self.device_)
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]
        num_skills = self.data["num_skills"]

        self.km_model = K_CMF(
            k_hidden_size=self.args.km_hidden_dim,
            skill_num=num_skills,
            user_num=num_users,
            item_num=num_items,
            q_matrix=Q_matrix,
        )
        self.Q_matrix = Q_matrix

        train_dataset = KMUserDataset(
            train_users=self.data["train_users"],
            train_sequences=self.data["train_sequences"],
        )
        train_loader = create_optimized_dataloader(
            train_dataset, batch_size=1, shuffle=True, device=self.device_
        )

        val_dataset = KMValidationDataset(self.data["test_triplets"])
        val_loader = create_optimized_dataloader(
            val_dataset, batch_size=len(val_dataset), shuffle=False, device=self.device_
        )

        optimizer = optim.Adam(self.km_model.parameters(), lr=self.args.km_lr)

        logger.info(
            f"KM Stage: users={num_users}, items={num_items}, skills={num_skills}"
        )

        return StageComponents(
            model=self.km_model,
            optimizer=optimizer,
            loss_fn=KMLoss(),
            train_data=train_loader,
            val_data=val_loader,
            epochs=self.args.km_epochs,
            early_stopping=EarlyStoppingConfig(
                monitor="acc", mode="max", patience=self.args.km_patience
            ),
            # 早停看 ACC，保存 / 回传给 AM 的是最佳 AUC 模型
            checkpoint_monitor="auc",
            checkpoint_mode="max",
        )

    def _build_am(self) -> StageComponents:
        """构建 AM 阶段：GMF 模型 + boosting 残差三元组 + boosting 损失。"""
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]

        if self.am_train_triplets is None:
            raise RuntimeError(
                "AM stage requires boosting residuals. "
                "Ensure the KM stage has completed (on_stage_complete)."
            )

        self.am_model = GMF(
            n_users=num_users,
            n_items=num_items,
            embedding_k=self.args.am_embedding_dim,
            aj_norm=self.adj_norm,
            adj=self.args.use_adj,
            layer=self.args.am_layer,
        )

        train_dataset = AMTripletDataset(self.am_train_triplets)
        train_loader = create_optimized_dataloader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            device=self.device_,
        )

        val_dataset = AMTripletDataset(self.am_test_triplets)
        val_loader = create_optimized_dataloader(
            val_dataset, batch_size=len(val_dataset), shuffle=False, device=self.device_
        )

        optimizer = optim.Adam(self.am_model.parameters(), lr=self.args.am_lr)
        loss_fn = AMLoss(
            am_lambda=self.args.am_lambda, combine_mode=self.args.combine_mode
        )

        logger.info(
            f"AM Stage: train_triplets={len(self.am_train_triplets)}, "
            f"test_triplets={len(self.am_test_triplets)}"
        )

        return StageComponents(
            model=self.am_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_loader,
            val_data=val_loader,
            epochs=self.args.am_epochs,
            early_stopping=EarlyStoppingConfig(
                monitor="acc", mode="max", patience=self.args.am_patience
            ),
            # 早停看 ACC，报告 / 保存的是最佳 AUC
            checkpoint_monitor="auc",
            checkpoint_mode="max",
        )

    def forward_pass(self, batch_data: tuple[Any, ...]) -> dict:
        """模型前向传播，按当前阶段分发。"""
        if self._current_stage == "km":
            return self._forward_km(batch_data)
        if self._current_stage == "am":
            return self._forward_am(batch_data)
        raise ValueError(f"Unknown stage: {self._current_stage!r}")

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        """计算损失，按当前阶段分发。"""
        if self._current_stage == "km":
            return self.loss(outputs["y_hat"], outputs["y_label"])

        # 验证模式：无 g/w，使用组合预测的 BCE 损失
        if "g" not in outputs or "w" not in outputs:
            y_hat = outputs["y_hat"].clamp(1e-6, 1 - 1e-6)
            return F.binary_cross_entropy(y_hat, outputs["y_label"])
        # 训练模式：完整 boosting 损失
        return self.loss(
            outputs["am_pred"],
            outputs["km_pred"],
            outputs["g"],
            outputs["w"],
            outputs["u_norm"],
            outputs["i_norm"],
        )

    def on_stage_complete(self, name: str, result) -> None:
        """KM 阶段结束后计算 boosting 残差，供 AM 阶段使用。"""
        if name == "km":
            logger.info("Computing boosting residuals for AM stage...")
            self._compute_boosting_residuals()

    def _forward_km(self, batch_data: tuple) -> dict:
        """KM 阶段的前向传播。

        训练时 batch_size=1，每次处理一个用户的完整序列；
        验证时 batch 包含所有测试三元组 (user_id, item_id, correct)。
        """
        user_ids, items, corrects = batch_data

        # 训练时 items 为 2D [1, seq_len]，验证时为 1D [batch_size]
        if items.dim() == 2:
            return self._forward_km_train(user_ids, items, corrects)
        return self._forward_km_val(user_ids, items, corrects)

    def _forward_km_train(
        self, user_ids: torch.Tensor, items: torch.Tensor, corrects: torch.Tensor
    ) -> dict:
        """KM 阶段训练时的前向传播。"""
        # batch_size=1，移除 batch 维
        user_id = user_ids.item()
        items = items.squeeze(0).to(self.device_)
        corrects = corrects.squeeze(0).to(self.device_)

        user_k, _, _ = self.km_model(user_id, items)
        item_q = self.Q_matrix[items, :]
        item_k = self.km_model.item_k[items, :]

        pred = IRT_2(user_k[:-1, :], item_k, item_q, self.args.km_guess).clamp(
            1e-6, 1 - 1e-6
        )

        return {
            "y_hat": pred,
            "y_label": corrects.float(),
            "y_predict": (pred >= 0.5).int(),
            "y_score": pred,
            "y_prob": pred,
        }

    def _forward_km_val(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor, corrects: torch.Tensor
    ) -> dict:
        """KM 阶段验证时的前向传播（批量处理模式）。

        先批量计算所有测试用户的最终知识状态，再批量预测所有测试样本。

        Args:
            user_ids: 用户 ID [batch_size]
            item_ids: 测试题目 ID [batch_size]
            corrects: 实际正确率 [batch_size]
        """
        user_ids = user_ids.to(self.device_)
        item_ids = item_ids.to(self.device_)
        corrects = corrects.to(self.device_)

        # Step 1: 批量计算所有测试用户的最终知识状态
        test_user_state_dict = {}
        with torch.no_grad():
            for test_user_id in self.data["test_users"]:
                if test_user_id in self.data["train_sequences"]:
                    train_items = self.data["train_sequences"][test_user_id][0][0]
                    train_items_tensor = torch.tensor(
                        train_items, dtype=torch.long, device=self.device_
                    )
                    user_k, _, _ = self.km_model(test_user_id, train_items_tensor)
                    test_user_state_dict[test_user_id] = user_k[-1, :]
                else:
                    # 无训练数据的用户使用初始知识状态
                    test_user_state_dict[test_user_id] = torch.sigmoid(
                        self.km_model.user_initial_k[test_user_id, :]
                    )

        # Step 2: 按测试样本的用户 ID 批量取知识状态
        user_states_k = torch.cat(
            [
                test_user_state_dict[user_id].unsqueeze(0)
                for user_id in user_ids.tolist()
            ],
            dim=0,
        )  # [batch_size, skill_num]

        # Step 3: 批量取题目参数
        item_states_q = self.Q_matrix[item_ids, :]  # [batch_size, skill_num]
        item_state_k = self.km_model.item_k[item_ids, :]  # [batch_size, skill_num]

        # Step 4: 批量 IRT 预测
        pred = IRT_2(
            user_states_k, item_state_k, item_states_q, self.args.km_guess
        ).clamp(1e-6, 1 - 1e-6)

        return {
            "y_hat": pred,
            "y_label": corrects.float(),
            "y_predict": (pred >= 0.5).int(),
            "y_score": pred,
            "y_prob": pred,
        }

    def _forward_am(self, batch_data: tuple) -> dict:
        """AM 阶段的前向传播。

        训练数据: [user, item, correct, km_pred, g, w] - 6 列
        验证数据: [user, item, correct, km_pred] - 4 列
        """
        # batch: [batch, 6] 或 [batch, 4]
        batch = batch_data[0] if isinstance(batch_data, tuple) else batch_data
        batch = batch.to(self.device_)

        u_idx = batch[:, 0].long()
        i_idx = batch[:, 1].long()
        correct = batch[:, 2]
        km_pred = batch[:, 3]

        am_pred, u_norm, i_norm = self.am_model(u_idx, i_idx)

        if self.args.combine_mode == "mul":
            final_pred = am_pred * km_pred
        else:
            final_pred = am_pred + km_pred

        output = {
            "y_hat": final_pred,
            "y_label": correct,
            "y_predict": (final_pred >= 0.5).int(),
            "y_score": final_pred,
            "y_prob": final_pred.clamp(0.0, 1.0),
            "am_pred": am_pred,
            "km_pred": km_pred,
            "u_norm": u_norm,
            "i_norm": i_norm,
        }

        # 训练模式下有 g、w 列，用于 boosting 损失
        if batch.shape[1] >= 6:
            output["g"] = batch[:, 4]
            output["w"] = batch[:, 5]

        return output

    def _compute_boosting_residuals(self):
        """基于训练好的 KM 模型计算 boosting 残差，并构建 AM 阶段所需数据。"""
        Q_matrix = self.Q_matrix
        train_sequences = self.data["train_sequences"]
        test_triplets = self.data["test_triplets"]
        train_users = self.data["train_users"]
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]

        self.km_model.eval()

        train_itemsq = []
        train_correctsq = []
        for user_id in train_users:
            seq_data = train_sequences[user_id]
            train_itemsq.append(torch.tensor(seq_data[0][0], dtype=torch.long))
            train_correctsq.append(torch.tensor(seq_data[1][0], dtype=torch.long))

        train_triplet_list = []
        user_final_states = {}

        with torch.no_grad():
            for idx, user_id in enumerate(
                tqdm(train_users, desc="Computing KM predictions")
            ):
                itemsq = train_itemsq[idx].to(self.device_)
                correctsq = train_correctsq[idx].to(self.device_)

                user_k, _, _ = self.km_model(user_id, itemsq)
                item_q = Q_matrix[itemsq, :]
                item_k = self.km_model.item_k[itemsq, :]

                clip_pred = IRT_2(
                    user_k[:-1, :], item_k, item_q, self.args.km_guess
                ).clamp(self.args.pretrain_clip, 1 - self.args.pretrain_clip)

                user_final_states[user_id] = user_k[-1, :]

                for i in range(len(itemsq)):
                    train_triplet_list.append(
                        [
                            user_id,
                            itemsq[i].item(),
                            correctsq[i].item(),
                            clip_pred[i].item(),
                        ]
                    )

        test_triplet_list = []
        with torch.no_grad():
            for user_id, item_id, correct in tqdm(
                test_triplets, desc="Computing test KM predictions"
            ):
                if user_id in user_final_states:
                    user_k = user_final_states[user_id]
                    item_q = Q_matrix[item_id, :]
                    item_k = self.km_model.item_k[item_id, :]
                    clip_pred = IRT_2(
                        user_k.unsqueeze(0),
                        item_k.unsqueeze(0),
                        item_q.unsqueeze(0),
                        self.args.km_guess,
                    ).clamp(self.args.pretrain_clip, 1 - self.args.pretrain_clip)
                    test_triplet_list.append(
                        [user_id, item_id, correct, clip_pred.item()]
                    )

        train_triplets = torch.tensor(train_triplet_list, dtype=torch.float32)
        self.am_test_triplets = torch.tensor(test_triplet_list, dtype=torch.float32)

        # boosting 残差: g = y/k - (1-y)/(1-k), w = -(y/k^2 + (1-y)/(1-k)^2)
        y = train_triplets[:, 2]
        k = train_triplets[:, 3]
        g = y / k - (1 - y) / (1 - k)
        w = -(y / (k**2) + (1 - y) / ((1 - k) ** 2))

        self.am_train_triplets = torch.cat(
            [train_triplets, g.unsqueeze(1), w.unsqueeze(1)], dim=1
        )

        logger.info(f"Train triplets shape: {self.am_train_triplets.shape}")
        logger.info(f"Test triplets shape: {self.am_test_triplets.shape}")

        self.adj_norm = self.model_data.build_normalized_adj_matrix(
            train_sequences=train_sequences,
            num_users=num_users,
            num_items=num_items,
            symmetric=True,
        ).to(self.device_)
