"""
HGIKT 模型训练脚本
"""


def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(description="HGIKT Training Arguments")

    # 模型参数
    model_params = parser.add_argument_group("Model Parameters")
    model_params.add_argument(
        "--lstm_hidden_dim",
        type=int,
        default=100,
        help="Dimension of LSTM hidden layers",
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
    model_params.add_argument(
        "--heads", type=int, default=2, help="Number of GNN attention heads"
    )
    model_params.add_argument(
        "--history_neighbour", type=int, default=5, help="Top K neighbors to consider"
    )
    model_params.add_argument(
        "--att_bound", type=float, default=0.2, help="Attention boundary value"
    )

    # 动态权重超图参数
    hypergraph_params = parser.add_argument_group("Hypergraph Parameters")
    hypergraph_params.add_argument(
        "--use_difficulty_weighted_hypergraph",
        action="store_true",
        default=True,
        help="Use difficulty-weighted hypergraph instead of regular hypergraph",
    )
    hypergraph_params.add_argument(
        "--num_difficulty_clusters",
        type=int,
        default=3,
        help="Number of difficulty clusters for weighted hypergraph (default: 3 for easy/medium/hard)",
    )

    # 门控融合子空间参数
    fusion_params = parser.add_argument_group("Fusion Parameters")
    fusion_params.add_argument(
        "--num_subspaces",
        type=int,
        default=4,
        help="Number of subspaces for learned fusion (default: 4)",
    )
    fusion_params.add_argument(
        "--sub_dim",
        type=int,
        default=32,
        help="Dimension of each subspace for learned fusion (default: 32)",
    )

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
    train_params.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs"
    )
    train_params.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    train_params.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_params.add_argument(
        "--lr_decay", type=float, default=None, help="Learning rate decay factor"
    )
    train_params.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )

    # 早停参数
    es_params = parser.add_argument_group("Early Stopping")
    es_params.add_argument(
        "--es_monitor",
        type=str,
        choices=["auc", "acc", "rmse", "loss"],
        default="auc",
        help="Metric to monitor for early stopping",
    )
    es_params.add_argument(
        "--es_mode",
        type=str,
        choices=["max", "min"],
        default="max",
        help="Optimization mode for monitored metric",
    )
    es_params.add_argument(
        "--es_patience",
        type=int,
        default=None,
        help="Patience for early stopping (None to disable)",
    )
    es_params.add_argument(
        "--es_min_delta",
        type=float,
        default=0.0,
        help="Minimum change to qualify as improvement",
    )
    es_params.add_argument(
        "--es_restore_best",
        action="store_true",
        default=False,
        help="Restore best weights when early stopping triggers (flag)",
    )

    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_dir", type=str, default=None, help="Directory to save logs and models"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")

    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()
    import torch
    import numpy
    from model.HGIKT import HGIKTTrainer
    from utility.data_process import Assistments2009Data
    from utility.data_process import Assistments2012Data
    from utility.data_process import Assistments2017Data
    from utility.data_process import EdNetKT1Data

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
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print("Initializing trainer...")
    trainer = HGIKTTrainer(
        args=args,
        data_src=data_src,
    )

    # 开始训练
    print("Starting training...")
    trainer.run()


if __name__ == "__main__":
    main()
