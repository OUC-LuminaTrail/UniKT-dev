import torch
from abc import ABC, abstractmethod
import time
import os
import random
import numpy as np
from tqdm import tqdm
from typing import Tuple, Any
from torch_geometric.profile import count_parameters
from utility.early_stopping import EarlyStopping, EarlyStoppingConfig
import swanlab
from dotenv import load_dotenv


def seed_everything(seed: int | None, deterministic: bool = True) -> int | None:
    r"""Set random seeds across common libraries for reproducibility."""
    if seed is None:
        return None

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)

    return seed


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
        train_data: torch.utils.data.DataLoader,
        val_data: torch.utils.data.DataLoader = None,
        lr_scheduler=None,
        early_stopping=None,
        hyperparams=None,
        log_dir: str = None,
        device: torch.device = None,
        checkpoint_path: str = None,
        use_swanlab: bool = True,
        seed: int | None = None,
    ):
        if device is None:
            self.device_ = self.try_gpu()
        else:
            self.device_ = torch.device(device)
        self.model: torch.nn.Module = model
        self.epochs: int = epochs
        self.opt = opt
        self.loss = loss
        self.train_data: torch.utils.data.DataLoader = train_data
        self.val_data: torch.utils.data.DataLoader = val_data
        self.lr_scheduler = lr_scheduler
        self.start_epoch = 0
        self.hyperparams = hyperparams
        if seed is None and hyperparams is not None:
            seed = getattr(hyperparams, "seed", None)
        self.seed = seed_everything(seed)

        # 自动优化 DataLoader 参数
        if isinstance(self.train_data, torch.utils.data.DataLoader):
            # 设置 num_workers，设置为 CPU 核心数
            cpu_count = os.cpu_count() or 1
            num_workers = min(cpu_count, 8)  # 限制最大为8以避免过多线程开销
            is_cuda = self.device_.type == "cuda"

            def _optimize_loader(loader):
                if not isinstance(loader, torch.utils.data.DataLoader):
                    return
                loader.num_workers = num_workers
                loader.pin_memory = is_cuda
                if num_workers > 0:
                    loader.prefetch_factor = 2

            _optimize_loader(self.train_data)
            if self.val_data is not None:
                _optimize_loader(self.val_data)

            print(
                f"DataLoader optimized: num_workers={num_workers}, pin_memory={is_cuda}"
            )

        # 初始化早停
        self.early_stopping: EarlyStopping | None = None
        if early_stopping is not None:
            if isinstance(early_stopping, EarlyStopping):
                self.early_stopping = early_stopping
            elif isinstance(early_stopping, EarlyStoppingConfig):
                self.early_stopping = EarlyStopping(early_stopping)
            elif isinstance(early_stopping, dict):
                self.early_stopping = EarlyStopping(**early_stopping)
        elif hyperparams is not None:
            # 从超参数中读取早停配置（来自命令行）
            es_patience = getattr(hyperparams, "es_patience", None)
            if es_patience is not None:
                monitor = getattr(hyperparams, "es_monitor", "auc")
                mode = getattr(hyperparams, "es_mode", "max")
                min_delta = getattr(hyperparams, "es_min_delta", 0.0)
                self.early_stopping = EarlyStopping(
                    EarlyStoppingConfig(
                        monitor=monitor,
                        mode=mode,
                        patience=es_patience,
                        min_delta=min_delta,
                    )
                )

        # 以当前时间戳命名日志文件夹
        dir = time.strftime("%Y%m%d-%H%M%S")
        if log_dir is not None:
            dir = os.path.join(log_dir, dir)
        self.log_dir = os.path.join("runs", dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 初始化超参数管理器
        self.hyperparam_manager = None
        if hyperparams is not None:
            self.setup_hyperparameters(hyperparams)

        # 断点续训
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # SwanLab 初始化
        self.use_swanlab = use_swanlab
        if self.use_swanlab:
            self.init_swanlab(
                project_name="kt-exp-graph",
                experiment_name=f"Run_{dir}",
            )

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
        if self.seed is not None:
            self.hyperparam_manager.add_metadata("seed", self.seed)

        # 添加设备信息
        if self.device_ is not None:
            device_info = self.get_device_info()
            for key, value in device_info.items():
                self.hyperparam_manager.add_metadata(key, value)

        # 保存超参数
        self.hyperparam_manager.save()

        # 打印摘要
        print(self.hyperparam_manager.get_summary())

    def save_checkpoint(self, epoch, path, weights_only=True):
        """
        保存完整检查点，包含模型权重、优化器状态等
        """
        if weights_only:
            torch.save(self.model.state_dict(), path)
            print(f"Model weights saved to {path}")
            return
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
        }
        if self.lr_scheduler is not None:
            state["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.early_stopping is not None:
            state["early_stopping_state"] = {
                "best_score": self.early_stopping.best_score,
                "best_epoch": self.early_stopping.best_epoch,
                "num_bad_epochs": self.early_stopping.num_bad_epochs,
            }

        torch.save(state, path)
        print(f"Model checkpoint saved to {path}")

    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device_)

        # 检查检查点文件是否正确
        required_keys = ["model_state_dict", "optimizer_state_dict"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(
                    f"Checkpoint is missing required key: {key}. Cannot resume training."
                )

        # 加载状态
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])

        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1

        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.early_stopping is not None and "early_stopping_state" in checkpoint:
            es_state = checkpoint["early_stopping_state"]
            self.early_stopping.best_score = es_state.get("best_score")
            self.early_stopping.best_epoch = es_state.get("best_epoch")
            self.early_stopping.num_bad_epochs = es_state.get("num_bad_epochs", 0)

        print(f"Resumed training from epoch {self.start_epoch}")

    def init_swanlab(self, project_name: str, experiment_name: str = None):
        """
        初始化 SwanLab 实验追踪

        参数:
            project_name: SwanLab 项目名称
            run_name: SwanLab 运行名称（可选）
        """
        import torch
        from swanlab.plugin.notification import LarkCallback
        
        # 加载环境变量
        load_dotenv()

        callbacks = []
        lark_webhook = os.getenv("LARK_WEBHOOK_URL")
        lark_secret = os.getenv("LARK_SECRET")
        if lark_webhook:
            callbacks.append(LarkCallback(webhook_url=lark_webhook, secret=lark_secret))

        swanlab.init(
            project_name=project_name,
            experiment_name=experiment_name,
            config=self.hyperparam_manager.get_hyperparameters_dict(),
            callbacks=callbacks,
            group=self.model.__class__.__name__,
            tags=["cuda" if torch.cuda.is_available() else "cpu"],
        )
        print(
            f"SwanLab initialized for project: {project_name}, run: {experiment_name}"
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

    @staticmethod
    def try_gpu():
        r"""
        获取可用的GPU设备
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device_info(self):
        """
        获取设备信息，包括CUDA设备型号

        Returns:
            dict: 包含设备类型和设备名称的字典
        """
        device_info = {}

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
            device_info["device_type"] = "CPU"

        return device_info

    def run(self):
        self.model.to(self.device_)  # 将模型移动到设备中
        self.loss = self.loss.to(self.device_)  # 将损失函数移动到设备中

        for epoch in range(self.start_epoch, self.epochs):
            print(f"Epoch {epoch+1}")
            # 训练
            train_total_loss = self.process_data(self.train_data, epoch, is_train=True)
            if self.use_swanlab:
                swanlab.log({"Train/Loss-epoch": train_total_loss}, step=epoch)
            # 验证
            if self.val_data is not None:
                val_total_loss = self.process_data(self.val_data, epoch, is_train=False)
                if self.use_swanlab:
                    swanlab.log({"Val/Loss-epoch": val_total_loss}, step=epoch)

                # 早停检查
                if self.early_stopping is not None and hasattr(
                    self, "_last_val_metrics"
                ):
                    monitor_value = self._select_monitor_value(
                        self._last_val_metrics, val_total_loss
                    )
                    should_stop = self.early_stopping.step(monitor_value, epoch)
                    if self.use_swanlab:
                        swanlab.log(
                            {"ES/BadEpochs": self.early_stopping.num_bad_epochs},
                            step=epoch,
                        )
                    if self.early_stopping.best_score is not None:
                        if self.use_swanlab:
                            swanlab.log(
                                {"ES/Best": self.early_stopping.best_score}, step=epoch
                            )
                    if should_stop:
                        print(
                            f"Early stopping triggered at epoch {epoch+1}. Best {self._monitor_name()} = "
                            f"{self.early_stopping.best_score:.4f} at epoch {int(self.early_stopping.best_epoch)+1 if self.early_stopping.best_epoch is not None else '?'}"
                        )
                        break

            # 学习率调度器更新
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                if self.use_swanlab:
                    swanlab.log(
                        {"Learning Rate": self.lr_scheduler.get_last_lr()[0]},
                        step=epoch,
                    )

            # 保存最新检查点用于续训
            self.save_checkpoint(
                epoch,
                os.path.join(self.log_dir, "last_checkpoint.pth"),
                weights_only=False,
            )

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
        记录标量指标到 SwanLab

        参数:
            name: 指标名称（如 "Train-ACC/batch"）
            value: 指标值
            step: 当前步数（通常是 epoch 或 batch 数）
        """
        if self.use_swanlab:
            swanlab.log({name: value}, step=step)

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

    def compute_loss(self, forward_outputs: Tuple[Any, ...]) -> torch.Tensor:
        """
        计算损失函数。
        子类可以重写此方法以支持更复杂的损失计算（如多任务损失、对比损失等）。
        
        参数:
            forward_outputs: forward_pass 的返回值元组

        返回:
            loss: 计算得到的标量损失张量
        """
        y_hat = forward_outputs[0]
        y_label = forward_outputs[1]
        return self.loss(y_hat, y_label)

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
        loss = self.compute_loss((y_hat, y_label, y_predict))
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
            loss = self.compute_loss((y_hat, y_label, y_predict))
            # 累积到 epoch 容器
            if hasattr(self, "_val_accum"):
                self._val_accum["y_hat"].append(y_hat.detach().cpu())
                self._val_accum["y_label"].append(y_label.detach().cpu())
                self._val_accum["y_pred"].append(y_predict.detach().cpu())
        return loss.item()

    def _aggregate_and_log(self, epoch: int, phase: str):
        r"""
        将本轮所有 batch 的预测与标签拼接，计算并记录按 epoch 聚合的 AUC ACC 和 RMSE。

        参数:
            epoch: 当前轮数
            phase: "train" 或 "val"
        - Train/ACC-epoch, Train/AUC-epoch
        - Val/ACC-epoch, Val/AUC-epoch
        """
        from sklearn.metrics import (
            roc_auc_score,
            accuracy_score,
            root_mean_squared_error,
        )

        accum = self._train_accum if phase == "train" else self._val_accum
        # 若没有数据，直接返回
        if not accum or len(accum["y_label"]) == 0:
            return

        y_label = torch.cat(accum["y_label"]).numpy()
        y_pred = torch.cat(accum["y_pred"]).numpy()
        y_hat = torch.cat(accum["y_hat"]).numpy()

        # 检查预测值是否为概率值，如果不是则自动应用 sigmoid
        if not ((y_hat >= 0).all() and (y_hat <= 1).all()):
            import numpy as np

            y_hat = 1 / (1 + np.exp(-y_hat))

        prefix = "Train/" if phase == "train" else "Val/"
        # ACC
        acc = accuracy_score(y_label, y_pred)
        self.log_metric(f"{prefix}ACC-epoch", acc, epoch)
        # AUC
        auc = None
        try:
            auc = roc_auc_score(y_label, y_hat)
            self.log_metric(f"{prefix}AUC-epoch", auc, epoch)
        except ValueError:
            auc = None
        # RMSE
        rmse = root_mean_squared_error(y_label, y_hat)
        self.log_metric(f"{prefix}RMSE-epoch", rmse, epoch)
        # 如果是验证阶段，保存最佳模型
        if phase == "val":
            # 保存最新一次验证指标
            self._last_val_metrics = {
                "acc": acc,
                "auc": auc if "auc" in locals() else None,
                "rmse": rmse,
            }
            monitor_value = self._select_monitor_value(self._last_val_metrics, None)
            if monitor_value is not None:
                self._save_best_model_checkpoint(monitor_value, epoch)

    def _monitor_name(self) -> str:
        if self.early_stopping is None:
            return "auc"
        return (self.early_stopping.cfg.monitor or "auc").lower()

    def _select_monitor_value(
        self, metrics: dict, val_loss: float | None
    ) -> float | None:
        """根据配置选择监控指标的值。"""
        name = self._monitor_name()
        if name == "loss":
            return float(val_loss) if val_loss is not None else None
        if name in metrics and metrics[name] is not None:
            return float(metrics[name])
        # 回退策略：优先 auc -> acc -> -rmse
        if metrics.get("auc") is not None:
            return float(metrics["auc"])
        if metrics.get("acc") is not None:
            return float(metrics["acc"])
        if metrics.get("rmse") is not None:
            # 如果要求最小化但后续比较使用 max，可在配置中选择 'min'
            return float(metrics["rmse"])
        return None

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
            print(
                f"Saving best model at epoch {epoch+1} with {self._monitor_name()} {metric:.4f}"
            )
            self.save_checkpoint(epoch, checkpoint_path, weights_only=True)


__all__ = ["Trainer", "seed_everything"]
