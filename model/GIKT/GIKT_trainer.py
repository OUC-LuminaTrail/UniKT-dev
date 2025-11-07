"""
GIKT 模型训练器
定义 GIKT 模型特定的训练逻辑
"""
import torch
from utility.net_trainer import Trainer
from sklearn.metrics import roc_auc_score, accuracy_score


class GIKTTrainer(Trainer):
    """
    GIKT模型训练器
    """

    def __init__(
        self, model, epochs, opt, loss, train_data, val_data=None, lr_scheduler=None
    ):
        super().__init__(model, epochs, opt, loss, train_data, val_data, lr_scheduler)

    def forward_pass(self, batch_data):
        sequence, response, mask = batch_data
        # 将数据移动到设备
        sequence = sequence.to(self.device_)
        response = response.to(self.device_)
        mask = mask.to(torch.bool).to(self.device_)

        # 模型前向传播
        y_hat = self.model(sequence, response, mask)
        # 使用mask选择有效位置
        y_hat = torch.masked_select(y_hat, mask)
        # 生成二分类预测（阈值0.5）
        y_predict = torch.ge(y_hat, torch.tensor(0.5).to(self.device_)).to(torch.int)
        # 获取真实标签
        y_label = torch.masked_select(response.float(), mask)

        return y_hat, y_label, y_predict

    def compute_metrics(
        self,
        y_label: torch.Tensor,
        y_hat: torch.Tensor,
        y_predict: torch.Tensor,
        epoch: int,
        phase: str,
    ):
        """
        GIKT模型的指标计算

        计算并记录以下指标：
        - ACC: 准确率
        - AUC: ROC曲线下面积
        """
        prefix = "Train/" if phase == "train" else "Val/"

        # 将数据移动到CPU并转换为numpy数组
        y_label_np = y_label.cpu().numpy()
        y_predict_np = y_predict.cpu().numpy()
        y_hat_np = y_hat.detach().cpu().numpy()  # 使用概率值计算AUC，而不是硬标签

        # 计算并记录准确率
        acc = accuracy_score(y_label_np, y_predict_np)
        self.log_metric(f"{prefix}ACC-epoch", acc, epoch)

        # 计算并记录AUC
        try:
            auc = roc_auc_score(y_label_np, y_hat_np)
            self.log_metric(f"{prefix}AUC-epoch", auc, epoch)
        except ValueError:
            # 如果只有一个类别，无法计算AUC
            print(f"Warning: Cannot compute AUC for {phase} phase (only one class in batch)")
