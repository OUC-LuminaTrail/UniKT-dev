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
        "--dropout", type=float, default=0.4, help="Dropout probability"
    )
    model_params.add_argument("--n_hop", type=int, default=3, help="Number of GNN hops")
    model_params.add_argument("--top_k", type=int, default=5, help="Top K neighbors to consider")

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
        "--fold", type=int, default=0, help="Index of folds for K-Fold cross-validation"
    )

    # 训练参数
    train_params = parser.add_argument_group("Training Parameters")
    train_params.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    train_params.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    train_params.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_params.add_argument(
        "--lr_decay", type=float, default=None, help="Learning rate decay factor"
    )
    train_params.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay (L2 regularization)"
    )

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda or cpu)"
    )

    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()
    import torch
    import numpy
    from model.GIKT.GIKT_trainer import GIKTTrainer
    from utility.data_process.assist09 import Assistments2009Data
    from utility.data_process.assist12 import Assistments2012Data
    from utility.data_process.assist17 import Assistments2017Data
    from utility.data_process.ednet_kt1 import EdNetKT1Data

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    
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

    print("Initializing trainer...")
    trainer = GIKTTrainer(
        args=args,
        data_src=data_src,
    )

    # 开始训练
    print("Starting training...")
    trainer.run()


if __name__ == "__main__":
    main()
