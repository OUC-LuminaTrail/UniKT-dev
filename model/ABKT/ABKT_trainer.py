"""
ABKT 模型训练器

使用 MultiTrainer 框架实现两阶段 Boosting 训练:
- Stage 1 (KM): 训练知识模块 K_CMF
- Stage 2 (AM): 训练能力模块 GMF，使用 boosting 残差
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm

from utils.config import (
    BaseParamConfig,
    EarlyStopping,
    EarlyStoppingConfig,
    create_optimized_dataloader,
    register_model_params,
)
from utils.core import TRAINERS, get_logger
from utils.data_process import DataSource
from utils.training import MultiTrainer, StageConfig

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
    """KM 阶段的用户序列数据集

    每个样本是一个用户的完整答题序列，batch_size=1
    """

    def __init__(
        self,
        train_users: list,
        train_sequences: dict,
    ):
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
    """KM 阶段的验证数据集

    每个样本是一个测试三元组
    """

    def __init__(self, test_triplets: list):
        self.test_triplets = test_triplets

    def __len__(self):
        return len(self.test_triplets)

    def __getitem__(self, idx: int):
        user_id, item_id, correct = self.test_triplets[idx]
        return user_id, item_id, correct


# ==================== AM 阶段的数据集 ====================


class AMTripletDataset(Data.Dataset):
    """AM 阶段的训练数据集

    每个样本包含 [user, item, correct, km_pred, g, w]
    """

    def __init__(self, triplets: torch.Tensor):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx: int):
        return self.triplets[idx]


# ==================== 自定义损失函数 ====================


class KMLoss(nn.Module):
    """KM 阶段损失函数：BCE Loss"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_hat: torch.Tensor, y_label: torch.Tensor) -> torch.Tensor:
        # y_hat 已经是概率，直接用 BCE
        return self.bce(y_hat, y_label)


class AMLoss(nn.Module):
    """AM 阶段损失函数：Boosting Loss + L2 正则化"""

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
        # Boosting 损失
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

        # L2 正则化
        l2_loss = u_norm + i_norm
        total_loss = loss + self.am_lambda * l2_loss

        return total_loss


# ==================== ABKT 训练器 ====================


@TRAINERS.register("ABKT")
class ABKTTrainer(MultiTrainer):
    """
    ABKT 两阶段训练器

    继承 MultiTrainer，实现：
    - Stage 1 (km): 训练 K_CMF (Knowledge Module)
    - Stage 2 (am): 训练 GMF (Ability Module) with boosting residuals
    """

    def __init__(
        self,
        args=None,
        data_src: DataSource | None = None,
        exp_manager=None,
    ):
        """
        初始化 ABKT 训练器

        Args:
            args: 命令行参数
            data_src: 数据源
            exp_manager: 实验管理器（可选，可后续通过 with_experiment 配置）
        """
        # 调用父类初始化（无参数）
        super().__init__()

        # 准备数据
        logger.info("Preparing ABKT data...")
        self.model_data = ABKTModelData(data_src)
        self.data = self.model_data.prepare_data(args)

        # 保存 args 供后续使用
        self.args = args

        # 设备
        device = None
        if args is not None and hasattr(args, "device") and args.device:
            device = args.device

        # 注册阶段构建器（必须在 build() 之前）
        self.with_stage_builder("km", self._build_km_stage)
        self.with_stage_builder("am", self._build_am_stage)

        # 如果提供了 exp_manager，则直接配置
        if exp_manager is not None:
            self.with_experiment(
                exp_manager=exp_manager,
                hyperparams=args,
                use_swanlab=True,
                model_name="ABKT",
                dataset_name=getattr(args, "dataset", "") if args else "",
                seed=getattr(args, "seed", None) if args else None,
                device=device,
            )
            self.build()

        # 模型引用
        self.km_model: K_CMF | None = None
        self.am_model: GMF | None = None

        # AM 阶段的额外数据（在 prepare_next_stage 中计算）
        self.am_train_triplets: torch.Tensor | None = None
        self.am_test_triplets: torch.Tensor | None = None
        self.adj_norm: torch.Tensor | None = None

        # 最佳指标
        self.best_km_auc = 0.0
        self.best_am_auc = 0.0

    def _build_km_stage(self) -> StageConfig:
        """构建 KM 阶段配置"""
        Q_matrix = self.data["Q_matrix"].to(self.device_)
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]
        num_skills = self.data["num_skills"]

        # 初始化 K_CMF 模型
        self.km_model = K_CMF(
            k_hidden_size=self.args.km_hidden_dim,
            skill_num=num_skills,
            user_num=num_users,
            item_num=num_items,
            q_matrix=Q_matrix,
        )

        # 保存 Q_matrix 引用
        self.Q_matrix = Q_matrix

        # 创建数据集和加载器（batch_size=1，单用户序列）
        train_dataset = KMUserDataset(
            train_users=self.data["train_users"],
            train_sequences=self.data["train_sequences"],
        )
        train_loader = create_optimized_dataloader(
            train_dataset, batch_size=1, shuffle=True, device=self.device_
        )

        # 验证数据：使用测试三元组
        val_dataset = KMValidationDataset(self.data["test_triplets"])
        val_loader = create_optimized_dataloader(
            val_dataset, batch_size=len(val_dataset), shuffle=False, device=self.device_
        )

        # 优化器
        optimizer = optim.Adam(self.km_model.parameters(), lr=self.args.km_lr)

        # 早停
        early_stopping = EarlyStopping(
            EarlyStoppingConfig(
                monitor="acc",
                mode="max",
                patience=self.args.km_patience,
            )
        )

        logger.info(
            f"KM Stage: users={num_users}, items={num_items}, skills={num_skills}"
        )

        return StageConfig(
            name="km",
            model=self.km_model,
            optimizer=optimizer,
            loss_fn=KMLoss(),
            train_data=train_loader,
            val_data=val_loader,
            epochs=self.args.km_epochs,
            early_stopping=early_stopping,
        )

    def _build_am_stage(self) -> StageConfig:
        """初始化 AM 阶段"""
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]

        # 确保 boosting 残差已计算
        if self.am_train_triplets is None:
            raise RuntimeError(
                "AM stage requires boosting residuals. "
                "Make sure prepare_next_stage was called after KM stage."
            )

        # 初始化 GMF 模型
        self.am_model = GMF(
            n_users=num_users,
            n_items=num_items,
            embedding_k=self.args.am_embedding_dim,
            aj_norm=self.adj_norm,
            adj=self.args.use_adj,
            layer=self.args.am_layer,
        )

        # 创建数据集和加载器
        train_dataset = AMTripletDataset(self.am_train_triplets)
        train_loader = create_optimized_dataloader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            device=self.device_,
        )

        # 验证数据
        val_dataset = AMTripletDataset(self.am_test_triplets)
        val_loader = create_optimized_dataloader(
            val_dataset, batch_size=len(val_dataset), shuffle=False, device=self.device_
        )

        # 优化器
        optimizer = optim.Adam(self.am_model.parameters(), lr=self.args.am_lr)

        # 早停
        early_stopping = EarlyStopping(
            EarlyStoppingConfig(
                monitor="auc",
                mode="max",
                patience=self.args.am_patience,
            )
        )

        # 损失函数
        loss_fn = AMLoss(
            am_lambda=self.args.am_lambda,
            combine_mode=self.args.combine_mode,
        )

        logger.info(
            f"AM Stage: train_triplets={len(self.am_train_triplets)}, "
            f"test_triplets={len(self.am_test_triplets)}"
        )

        return StageConfig(
            name="am",
            model=self.am_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_data=train_loader,
            val_data=val_loader,
            epochs=self.args.am_epochs,
            early_stopping=early_stopping,
        )

    def forward_pass(self, batch_data: tuple[Any, ...], stage_name: str) -> dict:
        """模型前向传播

        Args:
            batch_data: 批次数据
            stage_name: 阶段名称

        Returns:
            包含 y_hat, y_label, y_predict 的字典
        """
        if stage_name == "km":
            return self._forward_km(batch_data)
        elif stage_name == "am":
            return self._forward_am(batch_data)
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

    def _forward_km(self, batch_data: tuple) -> dict:
        """KM 阶段的前向传播

        训练时：batch_size=1，每次处理一个用户的完整序列
        验证时：batch 包含所有测试三元组 (user_id, item_id, correct)
        """
        user_ids, items, corrects = batch_data

        # 检测是训练模式还是验证模式
        # 训练模式：items 是 2D [1, seq_len]
        # 验证模式：items 是 1D [batch_size] (每个元素是单个 item_id)
        is_training = items.dim() == 2

        if is_training:
            return self._forward_km_train(user_ids, items, corrects)
        else:
            return self._forward_km_val(user_ids, items, corrects)

    def _forward_km_train(
        self, user_ids: torch.Tensor, items: torch.Tensor, corrects: torch.Tensor
    ) -> dict:
        """KM 阶段训练时的前向传播

        Args:
            user_ids: 用户 ID [1]
            items: 题目序列 [1, seq_len]
            corrects: 正确率序列 [1, seq_len]
        """
        # 移除 batch 维度（因为 batch_size=1）
        user_id = user_ids.item()
        items = items.squeeze(0).to(self.device_)
        corrects = corrects.squeeze(0).to(self.device_)

        # 前向传播
        user_k, _, _ = self.km_model(user_id, items)

        # 获取题目参数
        item_q = self.Q_matrix[items, :]
        item_k = self.km_model.item_k[items, :]

        # IRT 预测
        pred = IRT_2(user_k[:-1, :], item_k, item_q, self.args.km_guess)
        pred = pred.clamp(1e-6, 1 - 1e-6)

        # 生成二元预测
        y_predict = (pred >= 0.5).int()

        return {
            "y_hat": pred,
            "y_label": corrects.float(),
            "y_predict": y_predict,
        }

    def _forward_km_val(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor, corrects: torch.Tensor
    ) -> dict:
        """KM 阶段验证时的前向传播 (批量处理模式)

        采用原始ABKT的批量处理策略：
        1. 先批量计算所有测试用户的最终知识状态
        2. 然后批量预测所有测试样本

        Args:
            user_ids: 用户 ID [batch_size]
            item_ids: 测试题目 ID [batch_size]
            corrects: 实际正确率 [batch_size]
        """
        user_ids = user_ids.to(self.device_)
        item_ids = item_ids.to(self.device_)
        corrects = corrects.to(self.device_)

        # 获取唯一的测试用户
        test_users = self.data["test_users"]

        # Step 1: 批量计算所有测试用户的最终知识状态
        test_user_state_dict = {}
        with torch.no_grad():
            for test_user_id in test_users:
                if test_user_id in self.data["train_sequences"]:
                    # 获取用户的训练序列
                    train_items = self.data["train_sequences"][test_user_id][0][0]
                    train_items_tensor = torch.tensor(
                        train_items, dtype=torch.long, device=self.device_
                    )

                    # 计算知识状态演变
                    user_k, _, _ = self.km_model(test_user_id, train_items_tensor)
                    # 保存最后的知识状态
                    test_user_state_dict[test_user_id] = user_k[-1, :]
                else:
                    # 如果用户没有训练数据，使用初始知识状态
                    test_user_state_dict[test_user_id] = torch.sigmoid(
                        self.km_model.user_initial_k[test_user_id, :]
                    )

        # Step 2: 根据测试样本的用户ID，批量获取知识状态
        user_states_k_list = []
        for user_id in user_ids.tolist():
            user_states_k_list.append(test_user_state_dict[user_id].unsqueeze(0))
        user_states_k = torch.cat(user_states_k_list, dim=0)  # [batch_size, skill_num]

        # Step 3: 批量获取题目参数
        item_states_q = self.Q_matrix[item_ids, :]  # [batch_size, skill_num]
        item_state_k = self.km_model.item_k[item_ids, :]  # [batch_size, skill_num]

        # Step 4: 批量IRT预测
        pred = IRT_2(
            user_states_k, item_state_k, item_states_q, self.args.km_guess
        ).clamp(1e-6, 1 - 1e-6)

        # 生成二元预测
        y_predict = (pred >= 0.5).int()

        return {
            "y_hat": pred,
            "y_label": corrects.float(),
            "y_predict": y_predict,
        }

    def _forward_am(self, batch_data: tuple) -> dict:
        """AM 阶段的前向传播

        训练数据: [user, item, correct, km_pred, g, w] - 6 列
        验证数据: [user, item, correct, km_pred] - 4 列
        """
        # batch_data: [batch, 6] 或 [batch, 4]
        batch = batch_data[0] if isinstance(batch_data, tuple) else batch_data
        batch = batch.to(self.device_)

        u_idx = batch[:, 0].long()
        i_idx = batch[:, 1].long()
        correct = batch[:, 2]
        km_pred = batch[:, 3]

        # GMF 前向传播
        am_pred, u_norm, i_norm = self.am_model(u_idx, i_idx)

        # 组合预测
        if self.args.combine_mode == "add":
            final_pred = am_pred + km_pred
        elif self.args.combine_mode == "mul":
            final_pred = am_pred * km_pred
        else:
            final_pred = am_pred + km_pred

        # 生成二元预测
        y_predict = (final_pred >= 0.5).int()

        # 构建输出字典
        output = {
            "y_hat": final_pred.clamp(0.0, 1.0),
            "y_label": correct,
            "y_predict": y_predict,
            "am_pred": am_pred,
            "km_pred": km_pred,
            "u_norm": u_norm,
            "i_norm": i_norm,
        }

        # 训练模式下有 g 和 w 列（用于 boosting 损失计算）
        if batch.shape[1] >= 6:
            output["g"] = batch[:, 4]
            output["w"] = batch[:, 5]

        return output

    def compute_loss(self, outputs: dict, stage_name: str) -> torch.Tensor:
        """计算损失

        Args:
            outputs: forward_pass 的输出
            stage_name: 阶段名称

        Returns:
            损失张量
        """
        if stage_name == "km":
            return self.loss(outputs["y_hat"], outputs["y_label"])
        elif stage_name == "am":
            # 验证模式下没有 g 和 w，使用简单的 BCE 损失
            if "g" not in outputs or "w" not in outputs:
                # 使用组合预测的 BCE 损失作为验证损失
                y_hat = outputs["y_hat"].clamp(1e-6, 1 - 1e-6)
                return F.binary_cross_entropy(y_hat, outputs["y_label"])
            # 训练模式下使用完整的 boosting 损失
            return self.loss(
                outputs["am_pred"],
                outputs["km_pred"],
                outputs["g"],
                outputs["w"],
                outputs["u_norm"],
                outputs["i_norm"],
            )
        else:
            raise ValueError(f"Unknown stage: {stage_name}")

    def prepare_next_stage(self, stage_name: str, stage_outputs: dict) -> None:
        """阶段间数据准备

        在 KM 阶段完成后，计算 boosting 残差用于 AM 阶段
        """
        if stage_name == "km":
            logger.info("Computing boosting residuals for AM stage...")
            self._compute_boosting_residuals()
            self.best_km_auc = stage_outputs.get("best_metric", 0.0)

    def _compute_boosting_residuals(self):
        """计算 boosting 残差"""
        Q_matrix = self.Q_matrix
        train_sequences = self.data["train_sequences"]
        test_triplets = self.data["test_triplets"]
        train_users = self.data["train_users"]
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]

        self.km_model.eval()

        # 准备训练数据
        train_itemsq = []
        train_correctsq = []
        for user_id in train_users:
            seq_data = train_sequences[user_id]
            items = torch.tensor(seq_data[0][0], dtype=torch.long)
            corrects = torch.tensor(seq_data[1][0], dtype=torch.long)
            train_itemsq.append(items)
            train_correctsq.append(corrects)

        # 计算训练集的 KM 预测
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

                # KM 预测并裁剪
                pred = IRT_2(user_k[:-1, :], item_k, item_q, self.args.km_guess)
                clip_pred = pred.clamp(
                    self.args.pretrain_clip, 1 - self.args.pretrain_clip
                )

                # 保存最终状态
                user_final_states[user_id] = user_k[-1, :]

                # 构建训练三元组
                for i in range(len(itemsq)):
                    train_triplet_list.append(
                        [
                            user_id,
                            itemsq[i].item(),
                            correctsq[i].item(),
                            clip_pred[i].item(),
                        ]
                    )

        # 计算测试集的 KM 预测
        test_triplet_list = []
        with torch.no_grad():
            for user_id, item_id, correct in tqdm(
                test_triplets, desc="Computing test KM predictions"
            ):
                if user_id in user_final_states:
                    user_k = user_final_states[user_id]
                    item_q = Q_matrix[item_id, :]
                    item_k = self.km_model.item_k[item_id, :]
                    pred = IRT_2(
                        user_k.unsqueeze(0),
                        item_k.unsqueeze(0),
                        item_q.unsqueeze(0),
                        self.args.km_guess,
                    )
                    clip_pred = pred.clamp(
                        self.args.pretrain_clip, 1 - self.args.pretrain_clip
                    )
                    test_triplet_list.append(
                        [user_id, item_id, correct, clip_pred.item()]
                    )

        # 转换为张量
        train_triplets = torch.tensor(train_triplet_list, dtype=torch.float32)
        test_triplets_tensor = torch.tensor(test_triplet_list, dtype=torch.float32)

        # 计算 boosting 残差: g = y/k - (1-y)/(1-k), w = -(y/k^2 + (1-y)/(1-k)^2)
        y = train_triplets[:, 2]
        k = train_triplets[:, 3]
        g = y / k - (1 - y) / (1 - k)
        w = -(y / (k**2) + (1 - y) / ((1 - k) ** 2))

        # 添加 g 和 w 到训练三元组
        self.am_train_triplets = torch.cat(
            [train_triplets, g.unsqueeze(1), w.unsqueeze(1)], dim=1
        )

        self.am_test_triplets = test_triplets_tensor

        logger.info(f"Train triplets shape: {self.am_train_triplets.shape}")
        logger.info(f"Test triplets shape: {self.am_test_triplets.shape}")

        # 构建邻接矩阵
        self.adj_norm = self.model_data.build_normalized_adj_matrix(
            train_sequences=train_sequences,
            num_users=num_users,
            num_items=num_items,
            symmetric=True,
        ).to(self.device_)

    def _finish(self):
        """训练完成，记录最终指标"""
        # 获取 AM 阶段的最佳指标
        if "am" in self._stage_outputs:
            self.best_am_auc = self._stage_outputs["am"].get("best_metric", 0.0)

        logger.info("=" * 60)
        logger.info("ABKT Training Complete")
        logger.info(f"Best KM AUC: {self.best_km_auc:.4f}")
        logger.info(f"Best AM (Combined) AUC: {self.best_am_auc:.4f}")
        logger.info("=" * 60)

        # 调用父类的 _finish
        super()._finish()
