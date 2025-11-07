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
        # 模型内已将 response 右移一位作为输入，在时刻 t 的输出预测的是 t+1 的标签
        # 因此 y_hat[:, :-1] 对应 response[:, 1:]
        y_hat_full = self.model(sequence, response, mask)  # [B, S]

        # 提取有效位置的预测和标签
        y_hat_seq = y_hat_full[:, :-1]
        y_label_seq = response.float()[:, 1:]
        valid_mask = mask[:, 1:]

        # 使用 mask 选择有效位置
        y_hat = torch.masked_select(y_hat_seq, valid_mask)
        y_label = torch.masked_select(y_label_seq, valid_mask)

        # 若该批次没有任何有效位置，使用占位避免后续计算报错
        if y_label.numel() == 0:
            y_hat = torch.tensor([0.5], dtype=torch.float, device=self.device_)
            y_label = torch.tensor([0.0], dtype=torch.float, device=self.device_)

        # 生成二分类预测（阈值0.5）
        y_predict = torch.ge(y_hat, torch.tensor(0.5).to(self.device_)).to(torch.int)

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

        # 若当前批次无有效样本，跳过指标计算
        if y_label.numel() == 0:
            return

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
