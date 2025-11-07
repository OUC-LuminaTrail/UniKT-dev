import torch
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import time
import os
from tqdm import tqdm
from typing import Tuple, Any


class Trainer(ABC):
    """
    模型训练器

    子类需要实现：
    1. forward_pass: 模型前向传播逻辑
    2. compute_metrics: 指标计算和记录
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
    ):
        self.device_: torch.device | None = None
        self.model: torch.nn.Module = model
        self.epochs: int = epochs
        self.opt = opt
        self.loss = loss
        self.train_data = train_data
        self.val_data = val_data
        self.lr_scheduler = lr_scheduler
        self.loss_history = []

        # 以当前时间戳命名日志文件夹
        log_dir = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(os.path.join("runs", log_dir)):
            os.makedirs(os.path.join("runs", log_dir))
        # TensorBoard 日志记录器
        self.logger = SummaryWriter("runs/" + log_dir)
        # 记录模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.add_text("Model-Parameters", f"Total parameters: {total_params}")
        print(f"Model Parameters: {total_params}")

    def try_gpu(self, device=None):
        if device is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = device
        print(f"Using Device: {self.device_}")
        return self.device_

    def run(self):
        if self.device_ is None:
            self.try_gpu()  # 获取设备信息
        self.model.to(self.device_)  # 将模型移动到GPU中
        self.loss = self.loss.to(self.device_)  # 将损失函数移动到GPU中

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

    @abstractmethod
    def compute_metrics(
        self,
        loss: torch.Tensor,
        y_label: torch.Tensor,
        y_hat: torch.Tensor,
        y_predict: torch.Tensor,
        epoch: int,
        phase: str,
    ):
        """
        计算并记录模型的各项指标（需要子类实现）

        参数:
            loss: 当前批次的损失值
            y_label: 真实标签
            y_hat: 模型输出的预测概率
            y_predict: 二分类预测标签
            epoch: 当前训练轮数
            phase: "train" 或 "val"，表示当前阶段

        可用工具:
            self.log_metric(name, value, step): 记录标量指标到 TensorBoard
            self.log_loss(loss_value): 记录损失到历史记录
        """
        raise NotImplementedError(
            "Subclasses of Trainer must implement compute_metrics method"
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

    def log_loss(self, loss_value: float):
        """
        记录损失值到历史记录

        参数:
            loss_value: 损失值
        """
        self.loss_history.append(loss_value)

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
            for batch_data in tqdm(data_loader, desc="Training"):
                loss = self.run_train_epoch(batch_data, epoch)
                total_loss += loss
        else:
            for batch_data in tqdm(data_loader, desc="Validation"):
                loss = self.run_eval_epoch(batch_data, epoch)
                total_loss += loss
        return total_loss

    def run_train_epoch(self, batch_data: Tuple[Any, ...], epoch: int) -> float:
        """
        执行一个训练批次

        参数:
            batch_data: 从DataLoader获取的一个批次数据
            epoch: 当前轮数

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
        # 计算和记录指标
        self.compute_metrics(loss, y_label, y_hat, y_predict, epoch, "train")
        # 反向传播和优化
        loss.backward()
        self.opt.step()
        return loss.item()

    def run_eval_epoch(self, batch_data: Tuple[Any, ...], epoch: int) -> float:
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
            # 计算和记录指标
            self.compute_metrics(loss, y_label, y_hat, y_predict, epoch, "val")
        return loss.item()


__all__ = ["Trainer"]
