"""数据下载和处理的命令行工具。"""

from argparse import ArgumentParser

from utils.config import DataParams, GeneralParams, SamplingParams
from utils.core import get_logger
from utils.data_process import get_data_source

logger = get_logger(__name__)


SUPPORTED_DATASETS = [
    "assistments09",
    "assistments12",
    "assistments17",
    "ednet_kt1",
]


def build_parser():
    parser = ArgumentParser(description="Data Processing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download subcommand
    dl = subparsers.add_parser(
        "download", help="Download raw dataset archive and extract"
    )
    DataParams.add_args(dl)
    GeneralParams.add_args(dl)
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
    DataParams.add_args(proc)
    GeneralParams.add_args(proc)
    SamplingParams.add_args(proc)
    proc.set_defaults(func=cmd_process)

    return parser


def cmd_download(args):
    """Handle `download` subcommand."""
    dp = get_data_source(args.dataset, args)
    # override data_url if provided
    if getattr(args, "data_url", None):
        dp.data_url = args.data_url

    if not dp.data_url:
        raise ValueError(
            "No data_url available for this dataset. Provide --data_url explicitly."
        )

    logger.info(f"Downloading dataset {args.dataset} to {dp.data_folder}")

    # 获取下载参数
    force_download = getattr(args, "force", False)
    max_retries = getattr(args, "max_retries", 3)
    num_threads = getattr(args, "num_threads", 4)

    # 调用 fetch_data 并传递参数
    dp.fetch_data(
        force_download=force_download, max_retries=max_retries, num_threads=num_threads
    )
    # 持久化元信息
    dp.save_metadata()
    logger.info("Download complete.")


def cmd_process(args):
    """Handle `process` subcommand."""
    dp = get_data_source(args.dataset, args)
    dp.clear_data()

    if args.sample_users is not None and args.sample_users > 0:
        dp.sample_users(
            args.sample_users,
            random_sample=args.random_sample,
            attempts_bins=args.sample_attempts_bins,
            correct_bins=args.sample_correct_bins,
        )

    if args.kfold and args.kfold > 1:
        dp.add_kfold_labels(n_splits=args.kfold)

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
