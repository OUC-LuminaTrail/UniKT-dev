from argparse import ArgumentParser


SUPPORTED_DATASETS = [
    "assistments09",
    "assistments12",
    "assistments17",
    "ednet_kt1",
]


def _build_common_args(parser):
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--data_base_path",
        default="./data",
        type=str,
        help="Data base path for raw/processed files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def build_parser():
    parser = ArgumentParser(description="Data Processing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download subcommand
    dl = subparsers.add_parser(
        "download", help="Download raw dataset archive and extract"
    )
    _build_common_args(dl)
    dl.add_argument(
        "--data_url",
        type=str,
        default=None,
        help="Override data URL for downloading (optional)",
    )
    dl.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-download even if file already exists",
    )
    dl.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of download retries (default: 3)",
    )
    dl.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallel download (default: 4)",
    )
    dl.set_defaults(func=cmd_download)

    # process subcommand
    proc = subparsers.add_parser(
        "process", help="Process raw data into standardized format"
    )
    _build_common_args(proc)
    proc.add_argument(
        "--min_seq_len", type=int, default=10, help="Minimum sequence length"
    )
    proc.add_argument(
        "--max_seq_len", type=int, default=200, help="Maximum sequence length"
    )
    proc.add_argument(
        "--kfold",
        type=int,
        default=5,
        help="Number of folds for K-Fold cross-validation (>=2 to enable)",
    )
    proc.set_defaults(func=cmd_process)

    return parser


def _create_data_processor(args):
    from utils.data_process.assist09 import Assistments2009Data
    from utils.data_process.assist12 import Assistments2012Data
    from utils.data_process.assist17 import Assistments2017Data
    from utils.data_process.ednet_kt1 import EdNetKT1Data

    if args.dataset == "assistments09":
        return Assistments2009Data(args=args)
    if args.dataset == "assistments12":
        return Assistments2012Data(args=args)
    if args.dataset == "assistments17":
        return Assistments2017Data(args=args)
    if args.dataset == "ednet_kt1":
        return EdNetKT1Data(args=args)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def cmd_download(args):
    """Handle `download` subcommand."""
    dp = _create_data_processor(args)
    # override data_url if provided
    if getattr(args, "data_url", None):
        dp.data_url = args.data_url

    if not dp.data_url:
        raise ValueError(
            "No data_url available for this dataset. Provide --data_url explicitly."
        )

    print(f"Downloading dataset {args.dataset} to {dp.data_folder}")
    
    # 获取下载参数
    force_download = getattr(args, "force", False)
    max_retries = getattr(args, "max_retries", 3)
    num_threads = getattr(args, "num_threads", 4)
    
    # 调用 fetch_data 并传递参数
    dp.fetch_data(
        force_download=force_download,
        max_retries=max_retries,
        num_threads=num_threads
    )
    # 持久化元信息
    dp.save_metadata()
    print(f"Download complete.")


def cmd_process(args):
    """Handle `process` subcommand."""
    dp = _create_data_processor(args)
    # 清理数据
    dp.clear_data()
    # 添加交叉验证标签
    if hasattr(args, "kfold") and args.kfold and args.kfold > 1:
        dp.add_kfold_labels(n_splits=args.kfold)
    # 保存预处理后的数据
    dp.save_data()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "process":
        cmd_process(args)
    else:
        parser.print_help()
