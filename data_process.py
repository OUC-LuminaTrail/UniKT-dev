from argparse import ArgumentParser


def add_data_process_args(parser):
    """
    数据处理脚本参数解析
    """
    parser.add_argument(
        "--data_base_path",
        default="./data",
        type=str,
        help="Data base path",
    )
    parser.add_argument(
        "--download", action="store_true", help="Whether to download the dataset"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["assistments09", "assistments12", "assistments17", "ednet_kt1"],
        required=True,
        help="Select dataset to process",
    )
    parser.add_argument(
        "--min_seq_len", type=int, default=10, help="Minimum sequence length"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=200, help="Maximum sequence length"
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=5,
        help="Number of folds for K-Fold cross-validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def process_data(args):
    """
    根据命令行参数处理数据

    参数:
        args: 命令行参数对象
    """
    from utility.data_process.assist09 import Assistments2009Data
    from utility.data_process.assist12 import Assistments2012Data
    from utility.data_process.assist17 import Assistments2017Data
    from utility.data_process.ednet_kt1 import EdNetKT1Data

    if args.dataset == "assistments09":
        data_processor = Assistments2009Data(args=args)
    elif args.dataset == "assistments12":
        data_processor = Assistments2012Data(args=args)
    elif args.dataset == "assistments17":
        data_processor = Assistments2017Data(args=args)
    elif args.dataset == "ednet_kt1":
        data_processor = EdNetKT1Data(args=args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # 清理数据
    data_processor.clear_data()
    # 保存预处理后的数据
    data_processor.save_data()


if __name__ == "__main__":
    parser = ArgumentParser(description="Data Processing Script")
    parser = add_data_process_args(parser)
    args = parser.parse_args()
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_base_path}")
    process_data(args)
    print("Data processing complete.")
