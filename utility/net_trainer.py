import torch
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import time
import os
from tqdm import tqdm
from typing import Tuple, Any
from torch_geometric.profile import count_parameters


class Trainer(ABC):
    """
    模型训练器

    子类需要实现：
    1. init_model: 模型初始化
    2. forward_pass: 模型前向传播逻辑

    指标计算：
    - 训练器会在每个 epoch 结束时自动聚合所有 batch 的预测结果
    - 自动计算并记录 ACC 和 AUC 指标（*/ACC-epoch, */AUC-epoch）
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epochs: int,
        opt: torch.optim.Optimizer,
        loss,
        train_data,
        val_data=None,
        lr_scheduler=None,
        hyperparams=None,
        log_dir: str = None,
        device: torch.device = None,
    ):
        self.device_: torch.device = device
        self.model: torch.nn.Module = model
        self.epochs: int = epochs
        self.opt = opt
        self.loss = loss
        self.train_data = train_data
        self.val_data = val_data
        self.lr_scheduler = lr_scheduler

        # 以当前时间戳命名日志文件夹
        if log_dir is None:
            log_dir = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join("runs", log_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # TensorBoard 日志记录器
        self.logger = SummaryWriter(self.log_dir)

        # 初始化超参数管理器
        self.hyperparam_manager = None
        if hyperparams is not None:
            self.setup_hyperparameters(hyperparams)

    def setup_hyperparameters(self, hyperparams, model_name=None, dataset_name=None):
        """
        设置并保存超参数

        Args:
            hyperparams: 超参数（字典或Namespace对象）
            model_name: 模型名称（可选）
            dataset_name: 数据集名称（可选）
        """
        from utility.hyperparam_manager import create_hyperparameter_manager

        # 创建超参数管理器
        self.hyperparam_manager = create_hyperparameter_manager(
            args=hyperparams,
            save_dir=self.log_dir,
            model_name=model_name,
            dataset_name=dataset_name,
        )

        # 添加训练器相关元数据
        self.hyperparam_manager.add_metadata(
            "total_params", count_parameters(self.model)
        )
        self.hyperparam_manager.add_metadata("optimizer", type(self.opt).__name__)
        self.hyperparam_manager.add_metadata("loss_function", type(self.loss).__name__)
        if self.lr_scheduler is not None:
            self.hyperparam_manager.add_metadata(
                "lr_scheduler", type(self.lr_scheduler).__name__
            )
        if hasattr(self.opt, "defaults") and "weight_decay" in self.opt.defaults:
            self.hyperparam_manager.add_metadata(
                "weight_decay", self.opt.defaults["weight_decay"]
            )

        # 添加设备信息（包括CUDA设备型号）
        if self.device_ is not None:
            device_info = self.get_device_info()
            for key, value in device_info.items():
                self.hyperparam_manager.add_metadata(key, value)

        # 保存超参数
        self.hyperparam_manager.save()

        # 打印摘要
        print(self.hyperparam_manager.get_summary())

        # 将超参数摘要记录到TensorBoard
        self.logger.add_text(
            "Hyperparameters",
            self.hyperparam_manager.get_summary().replace("\n", "  \n"),
        )

    @abstractmethod
    def init_model(self):
        """
        初始化模型、优化器、损失函数和学习率调度器

        返回:
            model: torch.nn.Module
            optimizer: torch.optim.Optimizer
            loss_function: 损失函数
            lr_scheduler: 学习率调度器（可选）
        """
        raise NotImplementedError(
            "Subclasses of Trainer must implement init_model method"
        )

    def try_gpu(self, device=None):
        if device is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # 确保 device 是 torch.device 对象
            if isinstance(device, str):
                self.device_ = torch.device(device)
            else:
                self.device_ = device
        return self.device_

    def get_device_info(self):
        """
        获取设备信息，包括CUDA设备型号

        Returns:
            dict: 包含设备类型和设备名称的字典
        """
        device_info = {
            "device_type": str(self.device_),
        }

        if self.device_.type == "cuda":
            device_info["cuda_available"] = True
            device_info["cuda_device_count"] = torch.cuda.device_count()
            # 获取当前设备的索引
            device_index = self.device_.index if self.device_.index is not None else 0
            device_info["cuda_device_name"] = torch.cuda.get_device_name(device_index)
            device_info["cuda_device_capability"] = torch.cuda.get_device_capability(
                device_index
            )
        else:
            device_info["cuda_available"] = False
            device_info["device_name"] = "CPU"

        return device_info

    def run(self):
        self.model.to(self.device_)  # 将模型移动到设备中
        self.loss = self.loss.to(self.device_)  # 将损失函数移动到设备中

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}")
            # 训练
            train_total_loss = self.process_data(self.train_data, epoch, is_train=True)
            self.logger.add_scalar("Train/Loss-epoch", train_total_loss, epoch)
            # 验证
            if self.val_data is not None:
                val_total_loss = self.process_data(self.val_data, epoch, is_train=False)
                self.logger.add_scalar("Val/Loss-epoch", val_total_loss, epoch)

            # 学习率调度器更新
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.logger.add_scalar(
                    "Learning Rate", self.lr_scheduler.get_last_lr()[0], epoch
                )
            # 刷新日志
            self.logger.flush()
        self.logger.close()
        print("Training complete")

    @abstractmethod
    def forward_pass(
        self, batch_data: Tuple[Any, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        模型前向传播

        参数:
            batch_data: 从DataLoader获取的一个批次数据

        返回:
            Tuple[y_hat, y_label, y_predict]:
                - y_hat: 模型输出的预测概率/logits
                - y_label: 真实标签
                - y_predict: 预测结果
        """
        raise NotImplementedError(
            "Subclasses of Trainer must implement forward_pass method"
        )

    def log_metric(self, name: str, value: float, step: int):
        """
        记录标量指标到 TensorBoard

        参数:
            name: 指标名称（如 "Train-ACC/batch"）
            value: 指标值
            step: 当前步数（通常是 epoch 或 batch 数）
        """
        self.logger.add_scalar(name, value, step)

    def process_data(self, data_loader, epoch, is_train=True):
        """
        处理数据加载器中的所有批次

        参数:
            data_loader: 数据加载器
            epoch: 当前轮数
            is_train: 是否为训练模式

        返回:
            总损失值
        """
        total_loss = 0.0
        if is_train:
            self._train_accum = {"y_hat": [], "y_label": [], "y_pred": []}
            for batch_data in tqdm(data_loader, desc="Training"):
                loss = self.run_train_batch(batch_data)
                total_loss += loss
            # 训练阶段聚合指标
            self._aggregate_and_log(epoch, phase="train")
        else:
            self._val_accum = {"y_hat": [], "y_label": [], "y_pred": []}
            for batch_data in tqdm(data_loader, desc="Validation"):
                loss = self.run_eval_batch(batch_data)
                total_loss += loss
            # 验证阶段聚合指标
            self._aggregate_and_log(epoch, phase="val")
        return total_loss

    def run_train_batch(self, batch_data: Tuple[Any, ...]) -> float:
        """
        执行一个训练批次

        参数:
            batch_data: 从DataLoader获取的一个批次数据

        返回:
            该批次的损失值
        """
        self.model.train()
        # 清零梯度
        self.opt.zero_grad()
        # 前向传播
        y_hat, y_label, y_predict = self.forward_pass(batch_data)
        # 计算损失
        loss = self.loss(y_hat, y_label)
        if hasattr(self, "_train_accum"):
            self._train_accum["y_hat"].append(y_hat.detach().cpu())
            self._train_accum["y_label"].append(y_label.detach().cpu())
            self._train_accum["y_pred"].append(y_predict.detach().cpu())
        # 反向传播和优化
        loss.backward()
        self.opt.step()
        return loss.item()

    def run_eval_batch(self, batch_data: Tuple[Any, ...]) -> float:
        """
        执行一个验证批次

        参数:
            batch_data: 从DataLoader获取的一个批次数据
            epoch: 当前轮数

        返回:
            该批次的损失值
        """
        self.model.eval()
        with torch.no_grad():
            # 前向传播
            y_hat, y_label, y_predict = self.forward_pass(batch_data)
            # 计算损失
            loss = self.loss(y_hat, y_label)
            # 累积到 epoch 容器（用于 epoch 级别的指标计算）
            if hasattr(self, "_val_accum"):
                self._val_accum["y_hat"].append(y_hat.detach().cpu())
                self._val_accum["y_label"].append(y_label.detach().cpu())
                self._val_accum["y_pred"].append(y_predict.detach().cpu())
        return loss.item()

    def _aggregate_and_log(self, epoch: int, phase: str):
        r"""
        将本轮所有 batch 的预测与标签拼接，计算并记录按 epoch 聚合的 AUC 与 ACC。

        参数:
            epoch: 当前轮数
            phase: "train" 或 "val"
        - Train/ACC-epoch, Train/AUC-epoch
        - Val/ACC-epoch, Val/AUC-epoch
        """
        from sklearn.metrics import roc_auc_score, accuracy_score, root_mean_squared_error

        accum = self._train_accum if phase == "train" else self._val_accum
        # 若没有数据，直接返回
        if not accum or len(accum["y_label"]) == 0:
            return

        y_label = torch.cat(accum["y_label"]).numpy()
        y_pred = torch.cat(accum["y_pred"]).numpy()
        y_hat = torch.cat(accum["y_hat"]).numpy()

        prefix = "Train/" if phase == "train" else "Val/"
        # ACC
        acc = accuracy_score(y_label, y_pred)
        self.log_metric(f"{prefix}ACC-epoch", acc, epoch)
        # AUC
        try:
            auc = roc_auc_score(y_label, y_hat)
            self.log_metric(f"{prefix}AUC-epoch", auc, epoch)
        except ValueError:
            pass
        # RMSE
        rmse = root_mean_squared_error(y_label, y_hat)
        self.log_metric(f"{prefix}RMSE-epoch", rmse, epoch)
        # 如果是验证阶段，保存最佳模型
        if phase == "val":
            self._save_best_model_checkpoint(auc, epoch)

    def _save_best_model_checkpoint(self, metric: float, epoch: int):
        r"""
        保存模型检查点

        参数:
            - metric: 用于判断最佳模型的指标值
            - epoch: 当前轮数
        """
        checkpoint_path = os.path.join(self.log_dir, "best_model.pth")
        if not hasattr(self, "_best_metric") or metric > self._best_metric:
            self._best_metric = metric
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Best model saved at epoch {epoch+1} with metric {metric:.4f}")

    def __del__(self):
        if hasattr(self, "logger"):
            self.logger.close()


__all__ = ["Trainer"]
