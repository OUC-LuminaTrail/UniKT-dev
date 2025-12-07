"""
GIKT 模型训练器
定义 GIKT 模型特定的训练逻辑
"""

import torch
from utility.net_trainer import Trainer


class HGIKTTrainer(Trainer):
    """
    HGIKT模型训练器
    """

    def __init__(
        self,
        args=None,
        data_src=None,
    ):
        # 构建数据
        from model.HGIKT import HGIKTModelData

        model_data = HGIKTModelData(data_src)
        data_dict = model_data.prepare_data(args)
        
        # 解包数据
        train_data = data_dict['train_dataloader']
        val_data = data_dict['val_dataloader']
        self.hypergraph = data_dict['skill_hypergraph']
        self.hetero_graph = data_dict['hetero_graph']
        self.edge_weights = data_dict.get('edge_weights', None)
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
        )

    def init_model(self, args, data_src):
        from model.HGIKT.HGIKT_model import HGIKT

        print("Initializing HGIKT model...")
        model = HGIKT(args, data_src.get_metadata())

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
        self.hetero_graph = self.hetero_graph.to(self.device_)
        self.hypergraph = self.hypergraph.to(self.device_)
        
        # 如果有边权重，将其转换为张量并移动到设备
        edge_weights_tensor = None
        if self.edge_weights is not None:
            edge_weights_tensor = torch.tensor(
                self.edge_weights, dtype=torch.float32, device=self.device_
            )

        # 模型前向传播
        # 模型在时刻 t 的输出预测的是 t+1 的标签
        y_hat_full = self.model(
            sequence, response, mask, self.hetero_graph, self.hypergraph, edge_weights_tensor
        )  # [B, S]
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
            # Logits 0.0 对应概率 0.5
            y_hat = torch.tensor([0.0], dtype=torch.float, device=self.device_)
            y_label = torch.tensor([0.0], dtype=torch.float, device=self.device_)

        # 生成二分类预测（Logits 阈值 0.0 对应概率 0.5）
        y_predict = torch.ge(y_hat, torch.tensor(0.0).to(self.device_)).to(torch.int)

        return y_hat, y_label, y_predict
