"""
ABKT 模型训练器

实现两阶段 Boosting 训练:
- Stage 1 (KM): 训练知识模块 K_CMF
- Stage 2 (AM): 训练能力模块 GMF，使用 boosting 残差
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from utils.config import BaseParamConfig, register_model_params
from utils.core import TRAINERS, get_logger, seed_everything
from utils.data_process import DataSource

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
                "default": 50,
                "short": "kme",
                "help": "Number of epochs for KM stage (default: 50)",
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
                "default": 50,
                "short": "amp",
                "help": "Early stopping patience for AM stage (default: 50)",
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


@TRAINERS.register("ABKT")
class ABKTTrainer:
    """
    ABKT 两阶段训练器

    Stage 1: 训练 K_CMF (Knowledge Module)
    Stage 2: 训练 GMF (Ability Module) with boosting residuals
    """

    def __init__(
        self,
        args=None,
        data_src: DataSource = None,
        exp_manager=None,
    ):
        """
        初始化 ABKT 训练器

        Args:
            args: 命令行参数
            data_src: 数据源
            exp_manager: 实验管理器
        """
        self.args = args
        self.data_src = data_src
        self.exp_manager = exp_manager

        # 设备设置
        self.device = torch.device(
            args.device
            if hasattr(args, "device") and args.device
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # 设置随机种子
        self.seed = seed_everything(args.seed if hasattr(args, "seed") else None)

        # 日志目录
        if exp_manager is not None:
            self.log_dir = exp_manager.get_log_dir()
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        else:
            self.log_dir = "./runs/abkt_default"
            os.makedirs(self.log_dir, exist_ok=True)

        # 准备数据
        logger.info("Preparing ABKT data...")
        self.model_data = ABKTModelData(data_src)
        self.data = self.model_data.prepare_data(args)

        # 初始化模型（将在 run() 中创建）
        self.km_model: Optional[K_CMF] = None
        self.am_model: Optional[GMF] = None

        # 最佳指标
        self.best_km_auc = 0.0
        self.best_am_auc = 0.0

        # SwanLab 初始化
        self.use_swanlab = True
        if self.use_swanlab:
            self._init_swanlab()

    def _init_swanlab(self):
        """初始化 SwanLab 实验追踪"""
        import swanlab
        from dotenv import load_dotenv

        load_dotenv()

        from swanlab.plugin.notification import LarkCallback

        callbacks = []
        lark_webhook = os.getenv("LARK_WEBHOOK_URL")
        lark_secret = os.getenv("LARK_SECRET")
        if lark_webhook:
            callbacks.append(LarkCallback(webhook_url=lark_webhook, secret=lark_secret))

        # 构建超参数字典
        config = {
            "model": "ABKT",
            "dataset": self.args.dataset
            if hasattr(self.args, "dataset")
            else "unknown",
            "fold": self.args.fold if hasattr(self.args, "fold") else -1,
            "km_hidden_dim": self.args.km_hidden_dim,
            "km_guess": self.args.km_guess,
            "km_lr": self.args.km_lr,
            "km_epochs": self.args.km_epochs,
            "km_patience": self.args.km_patience,
            "am_embedding_dim": self.args.am_embedding_dim,
            "am_lambda": self.args.am_lambda,
            "am_layer": self.args.am_layer,
            "am_lr": self.args.am_lr,
            "am_epochs": self.args.am_epochs,
            "am_patience": self.args.am_patience,
            "pretrain_clip": self.args.pretrain_clip,
            "combine_mode": self.args.combine_mode,
            "batch_size": self.args.batch_size,
            "seed": self.seed,
        }

        experiment_name = os.path.basename(self.log_dir) if self.log_dir else "ABKT_run"
        swanlab.init(
            workspace=os.getenv("SWANLAB_WORKSPACE", None),
            project_name="kt-exp-graph",
            experiment_name=f"Run_{experiment_name}",
            config=config,
            callbacks=callbacks,
            group="ABKT",
            tags=["cuda" if torch.cuda.is_available() else "cpu", "two-stage"],
        )
        logger.info(f"SwanLab initialized for ABKT experiment: {experiment_name}")

    def run(self):
        """运行两阶段训练"""
        logger.info("=" * 50)
        logger.info("Starting ABKT Two-Stage Training")
        logger.info("=" * 50)

        # Stage 1: Train Knowledge Module
        logger.info("\n" + "=" * 50)
        logger.info("Stage 1: Training Knowledge Module (K_CMF)")
        logger.info("=" * 50)
        self._train_km_stage()

        # Compute boosting residuals
        logger.info("\n" + "=" * 50)
        logger.info("Computing Boosting Residuals")
        logger.info("=" * 50)
        train_triplets, test_triplets, adj_norm = self._compute_boosting_residuals()

        # Stage 2: Train Ability Module
        logger.info("\n" + "=" * 50)
        logger.info("Stage 2: Training Ability Module (GMF)")
        logger.info("=" * 50)
        self._train_am_stage(train_triplets, test_triplets, adj_norm)

        # 完成训练
        self._finish()

    def _train_km_stage(self):
        """Stage 1: 训练知识模块 K_CMF"""
        Q_matrix = self.data["Q_matrix"].to(self.device)
        train_sequences = self.data["train_sequences"]
        test_triplets = self.data["test_triplets"]
        train_users = self.data["train_users"]
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]
        num_skills = self.data["num_skills"]

        # 初始化 K_CMF 模型
        self.km_model = K_CMF(
            k_hidden_size=self.args.km_hidden_dim,
            skill_num=num_skills,
            user_num=num_users,
            item_num=num_items,
            Q_matrix=Q_matrix,
        ).to(self.device)

        # 优化器和损失函数
        optimizer = optim.Adam(self.km_model.parameters(), lr=self.args.km_lr)
        loss_fn = nn.BCELoss()

        # 准备训练数据
        train_itemsq = []
        train_correctsq = []
        for user_id in train_users:
            seq_data = train_sequences[user_id]
            items = torch.tensor(seq_data[0][0], dtype=torch.long)
            corrects = torch.tensor(seq_data[1][0], dtype=torch.long)
            train_itemsq.append(items)
            train_correctsq.append(corrects)

        # 准备测试数据
        test_user_set = set([t[0] for t in test_triplets])

        # 训练索引
        train_index = torch.arange(len(train_users))
        train_loader = Data.DataLoader(train_index, batch_size=1, shuffle=True)

        # Early stopping
        best_auc = 0.0
        patience_counter = 0

        # 创建进度条
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(table_column=Column(justify="right")),
            TimeRemainingColumn(),
            expand=True,
        )

        with progress:
            epoch_task = progress.add_task(
                "[bold red]KM Epochs", total=self.args.km_epochs
            )

            for epoch in range(self.args.km_epochs):
                self.km_model.train()

                train_preds = []
                train_labels = []
                total_loss = 0.0

                for idx in train_loader:
                    idx = idx.item()
                    user_id = train_users[idx]
                    itemsq = train_itemsq[idx].to(self.device)
                    correctsq = train_correctsq[idx].to(self.device)

                    # 前向传播
                    user_k, _, _ = self.km_model(user_id, itemsq)
                    item_q = Q_matrix[itemsq, :]
                    item_k = self.km_model.item_k[itemsq, :]

                    # IRT 预测
                    pred = IRT_2(user_k[:-1, :], item_k, item_q, self.args.km_guess)
                    pred = pred.clamp(1e-6, 1 - 1e-6)

                    # 计算损失
                    loss = loss_fn(pred, correctsq.float())
                    total_loss += loss.item()

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 记录预测
                    train_preds.extend(pred.detach().cpu().numpy())
                    train_labels.extend(correctsq.cpu().numpy())

                # 计算训练指标
                train_auc = roc_auc_score(train_labels, train_preds)
                train_acc = accuracy_score(
                    train_labels, [1 if p >= 0.5 else 0 for p in train_preds]
                )
                avg_loss = total_loss / len(train_users)

                # 验证阶段
                self.km_model.eval()
                with torch.no_grad():
                    # 获取测试用户的最终知识状态
                    user_final_states = {}
                    for idx, user_id in enumerate(train_users):
                        if user_id in test_user_set:
                            itemsq = train_itemsq[idx].to(self.device)
                            user_k, _, _ = self.km_model(user_id, itemsq)
                            user_final_states[user_id] = user_k[-1, :]

                    # 测试预测
                    test_preds = []
                    test_labels = []
                    for user_id, item_id, correct in test_triplets:
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
                            test_preds.append(pred.item())
                            test_labels.append(correct)

                    test_auc = roc_auc_score(test_labels, test_preds)
                    test_acc = accuracy_score(
                        test_labels, [1 if p >= 0.5 else 0 for p in test_preds]
                    )

                # SwanLab 日志
                if self.use_swanlab:
                    import swanlab

                    swanlab.log(
                        {
                            "Stage1/train_loss": avg_loss,
                            "Stage1/train_auc": train_auc,
                            "Stage1/train_acc": train_acc,
                            "Stage1/test_auc": test_auc,
                            "Stage1/test_acc": test_acc,
                        },
                        step=epoch,
                    )

                logger.info(
                    f"KM Epoch {epoch + 1}/{self.args.km_epochs}: "
                    f"Loss={avg_loss:.4f}, Train AUC={train_auc:.4f}, "
                    f"Test AUC={test_auc:.4f}, Test ACC={test_acc:.4f}"
                )

                # Early stopping
                if test_auc > best_auc:
                    best_auc = test_auc
                    self.best_km_auc = best_auc
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(
                        self.km_model.state_dict(),
                        os.path.join(self.log_dir, "best_km_model.pth"),
                    )
                else:
                    patience_counter += 1

                if patience_counter >= self.args.km_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best AUC: {best_auc:.4f}"
                    )
                    break

                progress.advance(epoch_task)

        # 加载最佳模型
        self.km_model.load_state_dict(
            torch.load(os.path.join(self.log_dir, "best_km_model.pth"))
        )
        logger.info(f"KM Stage completed. Best Test AUC: {best_auc:.4f}")

    def _compute_boosting_residuals(self) -> tuple:
        """
        计算 boosting 残差

        Returns:
            train_triplets: 训练三元组 [user, item, correct, km_pred, g, w]
            test_triplets: 测试三元组 [user, item, correct, km_pred]
            adj_norm: 归一化邻接矩阵
        """
        Q_matrix = self.data["Q_matrix"].to(self.device)
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
                itemsq = train_itemsq[idx].to(self.device)
                correctsq = train_correctsq[idx].to(self.device)

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
        train_triplets = torch.cat(
            [train_triplets, g.unsqueeze(1), w.unsqueeze(1)], dim=1
        )

        logger.info(f"Train triplets shape: {train_triplets.shape}")
        logger.info(f"Test triplets shape: {test_triplets_tensor.shape}")

        # 构建邻接矩阵
        adj_norm = self.model_data.build_normalized_adj_matrix(
            train_sequences=self.data["train_sequences"],
            num_users=num_users,
            num_items=num_items,
            symmetric=True,
        )

        return (
            train_triplets.to(self.device),
            test_triplets_tensor.to(self.device),
            adj_norm.to(self.device),
        )

    def _train_am_stage(
        self,
        train_triplets: torch.Tensor,
        test_triplets: torch.Tensor,
        adj_norm: torch.sparse.Tensor,
    ):
        """
        Stage 2: 训练能力模块 GMF

        Args:
            train_triplets: [user, item, correct, km_pred, g, w]
            test_triplets: [user, item, correct, km_pred]
            adj_norm: 归一化邻接矩阵
        """
        num_users = self.data["num_users"]
        num_items = self.data["num_items"]

        # 初始化 GMF 模型
        self.am_model = GMF(
            n_users=num_users,
            n_items=num_items,
            embedding_k=self.args.am_embedding_dim,
            aj_norm=adj_norm,
            adj=self.args.use_adj,
            layer=self.args.am_layer,
        ).to(self.device)

        # 优化器
        optimizer = optim.Adam(self.am_model.parameters(), lr=self.args.am_lr)

        # 数据加载器
        train_loader = Data.DataLoader(
            train_triplets, batch_size=self.args.batch_size, shuffle=True
        )

        # Early stopping
        best_auc = 0.0
        patience_counter = 0

        # 创建进度条
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(table_column=Column(justify="right")),
            TimeRemainingColumn(),
            expand=True,
        )

        with progress:
            epoch_task = progress.add_task(
                "[bold blue]AM Epochs", total=self.args.am_epochs
            )

            for epoch in range(self.args.am_epochs):
                self.am_model.train()
                total_loss = 0.0
                total_l2_loss = 0.0

                for batch in train_loader:
                    u_idx = batch[:, 0].long()
                    i_idx = batch[:, 1].long()
                    k_batch = batch[:, 3]
                    g_batch = batch[:, 4]
                    w_batch = batch[:, 5]

                    # 前向传播
                    pred, u_norm, i_norm = self.am_model(u_idx, i_idx)

                    # Boosting 损失
                    if self.args.combine_mode == "add":
                        loss = -torch.mean(
                            w_batch * torch.pow(pred + g_batch / w_batch, 2)
                        )
                    elif self.args.combine_mode == "mul":
                        loss = -torch.mean(
                            w_batch
                            * k_batch
                            * torch.pow(pred + g_batch / (w_batch * k_batch) - 1, 2)
                        )
                    else:
                        raise ValueError(
                            f"Unknown combine mode: {self.args.combine_mode}"
                        )

                    # L2 正则化
                    l2_loss = u_norm + i_norm
                    total_batch_loss = loss + self.args.am_lambda * l2_loss

                    # 反向传播
                    optimizer.zero_grad()
                    total_batch_loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * batch.shape[0]
                    total_l2_loss += l2_loss.item() * batch.shape[0]

                avg_loss = total_loss / train_triplets.shape[0]
                avg_l2_loss = total_l2_loss / train_triplets.shape[0]

                # 验证阶段
                self.am_model.eval()
                with torch.no_grad():
                    test_u_idx = test_triplets[:, 0].long()
                    test_i_idx = test_triplets[:, 1].long()
                    test_correct = test_triplets[:, 2]
                    test_km_pred = test_triplets[:, 3]

                    am_pred, _, _ = self.am_model(test_u_idx, test_i_idx)

                    # 组合预测
                    if self.args.combine_mode == "add":
                        final_pred = am_pred + test_km_pred
                    elif self.args.combine_mode == "mul":
                        final_pred = am_pred * test_km_pred
                    else:
                        final_pred = am_pred + test_km_pred

                    # 计算指标
                    final_pred_np = final_pred.cpu().numpy()
                    test_correct_np = test_correct.cpu().numpy()

                    test_auc = roc_auc_score(test_correct_np, final_pred_np)
                    test_acc = accuracy_score(
                        test_correct_np, [1 if p >= 0.5 else 0 for p in final_pred_np]
                    )

                # SwanLab 日志
                if self.use_swanlab:
                    import swanlab

                    swanlab.log(
                        {
                            "Stage2/train_loss": avg_loss,
                            "Stage2/l2_loss": avg_l2_loss,
                            "Stage2/test_auc": test_auc,
                            "Stage2/test_acc": test_acc,
                        },
                        step=self.args.km_epochs + epoch,
                    )

                logger.info(
                    f"AM Epoch {epoch + 1}/{self.args.am_epochs}: "
                    f"Loss={avg_loss:.4f}, L2={avg_l2_loss:.4f}, "
                    f"Test AUC={test_auc:.4f}, Test ACC={test_acc:.4f}"
                )

                # Early stopping
                if test_auc > best_auc:
                    best_auc = test_auc
                    self.best_am_auc = best_auc
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(
                        self.am_model.state_dict(),
                        os.path.join(self.log_dir, "best_am_model.pth"),
                    )
                else:
                    patience_counter += 1

                if patience_counter >= self.args.am_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best AUC: {best_auc:.4f}"
                    )
                    break

                progress.advance(epoch_task)

        logger.info(f"AM Stage completed. Best Test AUC: {best_auc:.4f}")

    def _finish(self):
        """训练完成，清理资源"""
        logger.info("=" * 50)
        logger.info("ABKT Training Complete")
        logger.info(f"Best KM AUC: {self.best_km_auc:.4f}")
        logger.info(f"Best AM (Combined) AUC: {self.best_am_auc:.4f}")
        logger.info("=" * 50)

        if self.use_swanlab:
            import swanlab

            swanlab.log(
                {
                    "Final/best_km_auc": self.best_km_auc,
                    "Final/best_am_auc": self.best_am_auc,
                }
            )
            swanlab.finish()
            logger.info("SwanLab run finished")
