"""
GIKT 模型训练脚本
"""


def parse_args():
    """解析命令行参数"""
    import argparse
    parser = argparse.ArgumentParser(description="GIKT Training Arguments")
    
    # 模型参数
    model_params = parser.add_argument_group("Model Parameters")
    model_params.add_argument(
        "--hidden_dim", type=int, default=100, help="Dimension of hidden layers"
    )
    model_params.add_argument(
        "--embedding_dim", type=int, default=100, help="Dimension of embeddings"
    )
    model_params.add_argument(
        "--lstm_layers", type=int, default=2, help="Number of LSTM layers"
    )
    model_params.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout probability"
    )
    model_params.add_argument("--n_hop", type=int, default=3, help="Number of GNN hops")

    # 数据参数
    data_params = parser.add_argument_group("Data Parameters")
    data_params.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["assistments09", "assistments12", "assistments17", "ednet_kt1"],
        required=True,
        help="Select dataset to use",
    )
    data_params.add_argument(
        "--data_base_path",
        type=str,
        default="./data",
        help="Path to the data files",
    )
    data_params.add_argument(
        "--min_seq_len", type=int, default=10, help="Minimum sequence length"
    )
    data_params.add_argument(
        "--max_seq_len", type=int, default=200, help="Maximum sequence length"
    )

    # 训练参数
    train_params = parser.add_argument_group("Training Parameters")
    train_params.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_params.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    train_params.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_params.add_argument(
        "--lr_decay", type=float, default=None, help="Learning rate decay factor"
    )

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )

    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()
    import torch

    from model.GIKT.GIKT_model import GIKT
    from model.GIKT.GIKT_data import build_data
    from model.GIKT.GIKT_trainer import GIKTTrainer
    from utility.data_process.assist09 import Assistments2009Data
    from utility.data_process.assist12 import Assistments2012Data
    from utility.data_process.assist17 import Assistments2017Data
    from utility.data_process.ednet_kt1 import EdNetKT1Data

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 构建数据
    print("Building datasets...")
    if args.dataset == "assistments09":
        data_src = Assistments2009Data(args=args)
    elif args.dataset == "assistments12":
        data_src = Assistments2012Data(args=args)
    elif args.dataset == "assistments17":
        data_src = Assistments2017Data(args=args)
    elif args.dataset == "ednet_kt1":
        data_src = EdNetKT1Data(args=args)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    train_data, val_data, graph = build_data(args, data_src)

    # 初始化模型
    print("Initializing GIKT model...")
    model = GIKT(args, graph)

    # 损失函数和优化器
    loss_fn = torch.nn.CrossEntropyLoss()  # 二分类交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器
    lr_scheduler = None
    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_decay
        )

    # 初始化训练器
    print("Initializing trainer...")
    trainer = GIKTTrainer(
        model=model,
        epochs=args.epochs,
        opt=optimizer,
        loss=loss_fn,
        train_data=train_data,
        val_data=val_data,
        lr_scheduler=lr_scheduler,
    )

    # 开始训练
    print("Starting training...")
    trainer.run()


if __name__ == "__main__":
    main()
