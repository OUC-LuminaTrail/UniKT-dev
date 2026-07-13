"""Command-line tool for downloading and processing datasets."""

from argparse import ArgumentParser

from omegaconf import OmegaConf

from utils.config import GeneralConfig, RunDataConfig, register_config_group
from utils.core import get_logger, seed_everything
from utils.data_process import get_data_source

logger = get_logger(__name__)


def _build_partial_rc(ns):
    """Build a data+general RunConfig view from the parsed namespace.

    Only ``data`` and ``general`` nodes are needed for ETL; config defaults come
    from the structured schema, user flags (dot-path) override them.
    """
    nested: dict = {}
    for key, value in vars(ns).items():
        if "." not in key:
            continue
        parts = key.split(".")
        cursor = nested
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    base = OmegaConf.structured({"data": RunDataConfig, "general": GeneralConfig})
    return OmegaConf.merge(base, OmegaConf.create(nested))


def build_parser():
    """Build the command-line argument parser for data processing."""
    parser = ArgumentParser(description="Data Processing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download subcommand
    dl = subparsers.add_parser(
        "download", help="Download raw dataset archive and extract"
    )
    register_config_group(dl, "data", RunDataConfig)
    register_config_group(dl, "general", GeneralConfig)
    dl.add_argument(
        "--data_url",
        type=str,
        default=None,
        help="Override data URL for downloading (optional)",
    )
    dl.add_argument(
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

    # process subcommand
    proc = subparsers.add_parser(
        "process", help="Process raw data into standardized format"
    )
    register_config_group(proc, "data", RunDataConfig)
    register_config_group(proc, "general", GeneralConfig)
    proc.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Extra processing steps",
    )

    return parser


def cmd_download(rc, ns):
    """Handle `download` subcommand."""
    dp = get_data_source(rc)
    if getattr(ns, "data_url", None):
        dp.data_url = ns.data_url

    if not dp.data_url:
        raise ValueError(
            "No data_url available for this dataset. Provide --data_url explicitly."
        )

    logger.info(f"Downloading dataset {rc.data.dataset} to {dp.data_folder}")
    dp.fetch_data(
        force_download=getattr(ns, "force", False),
        max_retries=getattr(ns, "max_retries", 3),
        num_threads=getattr(ns, "num_threads", 4),
    )
    dp.save_metadata()
    logger.info("Download complete.")


def cmd_process(rc, ns):
    """Handle `process` subcommand."""
    seed_everything(rc.general.seed, deterministic=False)
    dp = get_data_source(rc)
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

    if rc.data.sample_size is not None or rc.data.sample_ratio is not None:
        dp.sample(
            sample_size=rc.data.sample_size,
            sample_ratio=rc.data.sample_ratio,
            sample_strategy=rc.data.sample_strategy,
            attempts_bins=rc.data.sample_attempts_bins,
            correct_bins=rc.data.sample_correct_bins,
        )

    if rc.data.kfold and rc.data.kfold > 1:
        dp.add_kfold_labels(n_splits=rc.data.kfold, test_ratio=rc.data.test_ratio)

    dp.build_split_question_sequence_data()
    dp.build_split_skill_sequence_data()
    if "windowlate" in (ns.extra or []):
        dp.build_windowlate_data()

    dp.save_data()


if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    rc = _build_partial_rc(ns)

    if ns.command == "download":
        cmd_download(rc, ns)
    elif ns.command == "process":
        cmd_process(rc, ns)
    else:
        parser.print_help()
