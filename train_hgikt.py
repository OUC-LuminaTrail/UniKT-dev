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
        "--hidden_dim", type=int, default=100, help="Dimension of hidden layers"
    )
    model_params.add_argument(
        "--lstm_layers", type=int, default=1, help="Number of LSTM layers"
    )
    model_params.add_argument(
        "--dropout", type=float, default=0.54, help="Dropout probability"
    )
    model_params.add_argument("--n_hop", type=int, default=3, help="Number of GNN hops")
    model_params.add_argument(
        "--heads", type=int, default=4, help="Number of GNN attention heads"
    )
    model_params.add_argument(
        "--history_neighbour", type=int, default=10, help="Top K neighbors to consider"
    )
    model_params.add_argument(
        "--att_bound", type=float, default=0.0042, help="Attention boundary value"
    )
    model_params.add_argument(
        "--attention_dim", type=int, default=70, help="Dimension of attention layers"
    )
    model_params.add_argument(
        "--num_difficulty_clusters",
        type=int,
        default=5,
        help="Number of difficulty clusters for weighted hypergraph",
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
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    train_params.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    train_params.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to model checkpoints"
    )
    train_params.add_argument("--lr", type=float, default=0.0014, help="Learning rate")
    train_params.add_argument(
        "--lr_decay", type=float, default=None, help="Learning rate decay factor"
    )
    train_params.add_argument(
        "--weight_decay",
        type=float,
        default=3e-6,
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
        default=10,
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
    from model.HGIKT import HGIKTTrainer
    from utils.data_process import get_data_source

    # 构建数据
    print("Building datasets...")
    data_src = get_data_source(dataset_name=args.dataset, args=args)

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
