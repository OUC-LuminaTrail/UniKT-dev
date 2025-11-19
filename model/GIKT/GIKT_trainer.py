"""
GIKT 模型训练器
定义 GIKT 模型特定的训练逻辑
"""

import torch
from utility.net_trainer import Trainer


class GIKTTrainer(Trainer):
    """
    GIKT模型训练器
    """

    def __init__(
        self,
        args=None,
        data_src=None,
        log_dir=None,
    ):
        # 构建数据
        from model.GIKT.GIKT_data import GIKTModelData

        model_data = GIKTModelData(data_src)
        train_data, val_data, self.graph = model_data.prepare_data(args)
        model, opt, loss, lr_scheduler = self.init_model(args, data_src)
        super().__init__(
            model=model,
            epochs=args.epochs,
            opt=opt,
            loss=loss,
            train_data=train_data,
            val_data=val_data,
            lr_scheduler=lr_scheduler,
            hyperparams=args,
            log_dir=log_dir,
            device=args.device,
            use_amp=args.use_amp,
        )

    def init_model(self, args, data_src):
        from model.GIKT.GIKT_model import GIKT

        print("Initializing GIKT model...")
        model = GIKT(args, data_src.get_metadata())

        # 二分类交叉熵损失
        loss_fn = torch.nn.BCEWithLogitsLoss()
        # 优化器
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        # 学习率调度器
        lr_scheduler = None
        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay
            )

        return model, optimizer, loss_fn, lr_scheduler

    def forward_pass(self, batch_data):
        sequence, response, mask = batch_data
        # 将数据移动到设备
        sequence = sequence.to(self.device_)
        response = response.to(self.device_)
        mask = mask.to(torch.bool).to(self.device_)
        self.graph = self.graph.to(self.device_)

        # 模型前向传播
        # 模型在时刻 t 的输出预测的是 t+1 的标签
        y_hat_full = self.model(sequence, response, mask, self.graph)  # [B, S]

        # 提取有效位置的预测和标签
        y_hat_seq = y_hat_full[:, :-1]
        y_label_seq = response.float()[:, 1:]
        mask_curr = mask[:, :-1]
        mask_next = mask[:, 1:]
        valid_mask = mask_curr & mask_next  # [B, S-1]

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
