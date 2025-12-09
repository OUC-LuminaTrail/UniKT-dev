"""
SQGKT 模型训练器
定义 SQGKTTrainer 类，用于训练和评估 SQGKT 模型。
"""

import torch
from utils.net_trainer import Trainer


class SQGKTTrainer(Trainer):
    """
    SQGKT模型训练器
    """

    def __init__(
        self,
        args=None,
        data_src=None,
    ):
        # 构建数据
        from model.SQGKT.SQGKT_data import SQGKTModelData

        model_data = SQGKTModelData(data_src)
        (
            train_data,
            val_data,
            self.uq_matrix,
            self.qs_matrix,
            self.qs_q_neighbor_list,
            self.qs_s_neighbor_list,
            self.uq_u_neighbor_list,
            self.uq_q_neighbor_list,
        ) = model_data.prepare_data(args)
        
        # 将numpy数组转换为torch张量
        self.uq_matrix = torch.from_numpy(self.uq_matrix)  # 3维张量: [num_users, num_questions, 3]
        self.qs_matrix = torch.from_numpy(self.qs_matrix)  # 2维张量: [num_questions, num_skills]
        self.qs_q_neighbor_list = torch.from_numpy(self.qs_q_neighbor_list)
        self.qs_s_neighbor_list = torch.from_numpy(self.qs_s_neighbor_list)
        self.uq_u_neighbor_list = torch.from_numpy(self.uq_u_neighbor_list)
        self.uq_q_neighbor_list = torch.from_numpy(self.uq_q_neighbor_list)
        
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
            log_dir=args.log_dir,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            seed=args.seed,
        )

    def init_model(self, args, data_src):
        from model.SQGKT import SQGKT

        print("Initializing SQGKT model...")
        model = SQGKT(args, data_src.get_metadata())

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
        sequence, response, mask, user_ids = batch_data
        # 将数据移动到设备
        sequence = sequence.to(self.device_)
        response = response.to(self.device_)
        mask = mask.to(torch.bool).to(self.device_)
        user_ids = user_ids.to(self.device_)

        # 模型前向传播
        y_hat_full = self.model(
            user_ids,
            sequence,
            response,
            mask,
            self.uq_matrix.to(self.device_),
            self.qs_matrix.to(self.device_),
            self.qs_q_neighbor_list.to(self.device_),
            self.qs_s_neighbor_list.to(self.device_),
            self.uq_u_neighbor_list.to(self.device_),
            self.uq_q_neighbor_list.to(self.device_),
        )  # [B, S]

        # 提取有效位置的预测和标签
        # 跳过第一个时间步
        y_hat_seq = y_hat_full[:, 1:]
        y_label_seq = response.float()[:, 1:]
        valid_mask = mask[:, 1:]

        # 使用 mask 选择有效位置
        y_hat = torch.masked_select(y_hat_seq, valid_mask)
        y_label = torch.masked_select(y_label_seq, valid_mask)

        # 若该批次没有任何有效位置，使用占位避免后续计算报错
        if y_label.numel() == 0:
            # Logits 0.0 对应概率 0.5
            y_hat = torch.tensor([0.0], dtype=torch.float, device=self.device_)
            y_label = torch.tensor([0.0], dtype=torch.float, device=self.device_)

        # 生成二分类预测（Logits 阈值 0.0 对应概率 0.5）
        y_predict = torch.ge(y_hat, torch.tensor(0.0).to(self.device_)).to(torch.int)

        return y_hat, y_label, y_predict
