"""Command-line tool for downloading and processing datasets."""

from argparse import ArgumentParser

from utils.config import DataParams, GeneralParams, SamplingParams
from utils.core import get_logger
from utils.data_process import get_data_source

logger = get_logger(__name__)


def build_parser():
    """Build the command-line argument parser for data processing."""
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
    proc.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra processing steps",
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

    # Get download parameters
    force_download = getattr(args, "force", False)
    max_retries = getattr(args, "max_retries", 3)
    num_threads = getattr(args, "num_threads", 4)

    # Call fetch_data with the extracted parameters
    dp.fetch_data(
        force_download=force_download, max_retries=max_retries, num_threads=num_threads
    )
    # Persist metadata
    dp.save_metadata()
    logger.info("Download complete.")


def cmd_process(args):
    """Handle `process` subcommand."""
    dp = get_data_source(args.dataset, args)
    dp.clean_raw_data()
    # The raw interaction frame is no longer needed after cleaning; release it so
    # it doesn't pile up against the (much larger) split-stage intermediates.
    # NOTE: question_data_raw is still consumed by transform_data, so keep it.
    for attr in ("sequence_data_raw",):
        if hasattr(dp, attr):
            setattr(dp, attr, None)

    dp.transform_data()
    # Cleaned data and question metadata are fully consumed by transform_data;
    # release before the memory-heavy split stages.
    for attr in ("cleaned_raw_data", "question_data_raw"):
        if hasattr(dp, attr):
            setattr(dp, attr, None)

    if args.sample_size is not None or args.sample_ratio is not None:
        dp.sample(
            sample_size=args.sample_size,
            sample_ratio=args.sample_ratio,
            sample_strategy=args.sample_strategy,
            attempts_bins=args.sample_attempts_bins,
            correct_bins=args.sample_correct_bins,
        )

    if args.kfold and args.kfold > 1:
        dp.add_kfold_labels(n_splits=args.kfold, test_ratio=args.test_ratio)

    dp.build_split_question_sequence_data()
    dp.build_split_skill_sequence_data()
    if "windowlate" in args.extra:
        dp.build_windowlate_data()

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
