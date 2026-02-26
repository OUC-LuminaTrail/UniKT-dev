import hashlib
import json
import os
import shutil
import time
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import requests
import tqdm
from sklearn.model_selection import KFold

from utils.core import get_logger

logger = get_logger(__name__)


class DataSource(ABC):
    """Base class for data source management.

    Provides common functionality for downloading, loading, processing,
    and saving dataset files with metadata tracking and integrity checks.
    """

    def __init__(
        self, dataset: str, data_base_path: str, data_url: str = None, seed: int = 42
    ):
        super().__init__()
        self.dataset = dataset.lower()
        self.data_base_path = data_base_path
        self.data_folder = os.path.join(self.data_base_path, self.dataset)
        self.metadata_path = os.path.join(self.data_folder, "metadata.json")
        self.raw_data = None
        self.cleared_data = None
        self.sequence_data = None
        self.question_data = None
        self.data_url = data_url
        self.metadata = {}
        self.seed = seed
        self.set_random_seed()

    def set_random_seed(self):
        """Set random seeds for reproducibility."""
        import random

        import numpy as np

        seed = self.seed if self.seed is not None else 42
        random.seed(seed)
        np.random.seed(seed)
        logger.debug(f"Random seed set to {seed}")

    def _download_chunk(self, url, start, end, chunk_path, pbar, chunk_size=8192):
        """Download a single chunk of a file for multi-threaded downloads."""
        headers = {"Range": f"bytes={start}-{end}"}
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        with open(chunk_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    if pbar:
                        pbar.update(len(chunk) / (1024 * 1024))

    def _download_with_requests(self, archive_path, num_threads, attempt, max_retries):
        """Download file with multi-threading support."""
        head_response = requests.head(self.data_url, timeout=30)
        head_response.raise_for_status()

        total_bytes = int(head_response.headers.get("content-length", 0))
        accept_ranges = head_response.headers.get("accept-ranges", "none")

        if accept_ranges != "bytes" or total_bytes < 10 * 1024 * 1024:
            if accept_ranges != "bytes":
                logger.warning(
                    "Server does not support range requests, using single-threaded download"
                )
            self._download_single_thread(archive_path)
            return

        total_mb = total_bytes / (1024 * 1024)
        chunk_size = total_bytes // num_threads
        temp_dir = archive_path + ".parts"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            with tqdm.tqdm(
                total=total_mb,
                unit="MB",
                unit_scale=False,
                desc=f"Downloading (attempt {attempt}/{max_retries})",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f}MB [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = []
                    for i in range(num_threads):
                        start = i * chunk_size
                        end = (
                            start + chunk_size - 1
                            if i < num_threads - 1
                            else total_bytes - 1
                        )
                        chunk_path = os.path.join(temp_dir, f"chunk_{i}")

                        future = executor.submit(
                            self._download_chunk,
                            self.data_url,
                            start,
                            end,
                            chunk_path,
                            pbar,
                        )
                        futures.append(future)

                    for future in futures:
                        future.result()

            logger.debug("Merging downloaded chunks...")
            with open(archive_path, "wb") as outfile:
                for i in range(num_threads):
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}")
                    with open(chunk_path, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _download_single_thread(self, archive_path):
        """Download file with single thread."""
        with requests.get(self.data_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_bytes = int(r.headers.get("content-length", 0))
            total_mb = total_bytes / (1024 * 1024)
            chunk_size = 8192

            with open(archive_path, "wb") as f:
                with tqdm.tqdm(
                    total=total_mb,
                    unit="MB",
                    unit_scale=False,
                    desc="Downloading",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f}MB [{elapsed}<{remaining}, {rate_fmt}]",
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk) / (1024 * 1024))

    def fetch_data(self, force_download=False, max_retries=3, num_threads=4):
        """Download and extract data archive with retry support.

        Args:
            force_download: Force re-download even if file exists.
            max_retries: Maximum number of download retry attempts.
            num_threads: Number of threads for multi-threaded download.
        """

        if self.data_url is None:
            raise ValueError("Data URL is not provided.")

        os.makedirs(self.data_folder, exist_ok=True)

        file_name = self.data_url.split("/")[-1]
        if not file_name:
            file_name = "downloaded_data"
        archive_path = os.path.join(self.data_folder, file_name)

        if not os.path.exists(archive_path) or force_download:
            if force_download and os.path.exists(archive_path):
                logger.warning(
                    f"Force download enabled, removing existing file: {archive_path}"
                )
                os.remove(archive_path)

            logger.info(f"Downloading data from {self.data_url}")

            for attempt in range(max_retries):
                try:
                    self._download_with_requests(
                        archive_path, num_threads, attempt + 1, max_retries
                    )
                    logger.info(f"Download finished: {archive_path}")
                    break
                except Exception as e:
                    if os.path.exists(archive_path):
                        os.remove(archive_path)
                        logger.debug(f"Removed incomplete download: {archive_path}")

                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        logger.error(
                            f"Download failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"Failed to download data after {max_retries} attempts: {e}"
                        )
        else:
            logger.info(f"Dataset already exists, skip downloading: {archive_path}")

        archive_md5 = self.compute_md5(archive_path)
        self.add_metadata("raw_archive_md5", archive_md5)
        self.add_metadata("raw_archive_filename", file_name)

        extract_target = os.path.join(self.data_folder, "raw")
        os.makedirs(extract_target, exist_ok=True)

        should_extract = self._should_extract(force_download, extract_target)

        if should_extract:
            logger.info(f"Extracting archive: {archive_path}")
            self._extract_archive(archive_path, file_name, extract_target)
            logger.info(f"Extraction finished: {extract_target}")
        else:
            logger.info(
                f"Raw data directory not empty, skip extraction: {extract_target}"
            )

        self.add_metadata("raw_data_path", extract_target)

    def _should_extract(self, force_download: bool, extract_target: str) -> bool:
        """Determine whether extraction is needed."""
        if force_download:
            if any(Path(extract_target).iterdir()):
                logger.warning(
                    f"Force mode enabled, removing existing raw data: {extract_target}"
                )
                shutil.rmtree(extract_target)
                os.makedirs(extract_target, exist_ok=True)
            return True
        return not any(Path(extract_target).iterdir())

    def _extract_archive(self, archive_path: str, file_name: str, extract_target: str):
        """Extract archive file based on its extension."""
        import gzip
        import tarfile

        lower_name = file_name.lower()

        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_target)
        elif lower_name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_target)
        elif lower_name.endswith(".tar"):
            with tarfile.open(archive_path, "r:") as tf:
                tf.extractall(extract_target)
        elif lower_name.endswith(".gz") and not lower_name.endswith(".tar.gz"):
            uncompressed_name = lower_name[:-3]
            target_file = os.path.join(extract_target, uncompressed_name)
            with (
                gzip.open(archive_path, "rb") as f_in,
                open(target_file, "wb") as f_out,
            ):
                shutil.copyfileobj(f_in, f_out)
        else:
            dest_path = os.path.join(extract_target, file_name)
            if archive_path != dest_path:
                shutil.copy2(archive_path, dest_path)

    @abstractmethod
    def load_src_data(self):
        """Load source data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement load_data method")

    def load_processed_data(self):
        """Load processed data files with integrity checks.

        Raises:
            FileNotFoundError: Processed data files not found.
            ValueError: MD5 checksum mismatch.
        """
        self.load_metadata()

        sequence_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_sequence.parquet"
        )
        question_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_question.parquet"
        )

        self._validate_data_files_exist([sequence_data_path, question_data_path])
        self._validate_data_integrity(sequence_data_path, "sequence_data_md5")
        self._validate_data_integrity(question_data_path, "question_data_md5")
        self._load_parquet_files(sequence_data_path, question_data_path)

    def _validate_data_files_exist(self, file_paths: list[str]):
        """Validate that all required data files exist."""
        missing_files = [
            f"  - {path}" for path in file_paths if not os.path.exists(path)
        ]

        if missing_files:
            missing_str = "\n".join(missing_files)
            raise FileNotFoundError(
                f"Processed data files not found for dataset '{self.dataset}':\n"
                f"{missing_str}\n\n"
                f"To fix this, please run preprocessing first:\n"
                f"   python data_process.py process -d {self.dataset}\n\n"
                f"Data base path: {self.data_folder}"
            )

    def _validate_data_integrity(self, file_path: str, md5_key: str):
        """Validate MD5 checksum of a data file."""
        if md5_key not in self.metadata:
            return

        actual_md5 = self.compute_md5(file_path)
        expected_md5 = self.metadata[md5_key]

        if actual_md5 != expected_md5:
            data_type = "sequence" if "sequence" in md5_key else "question"
            raise ValueError(
                f"{data_type.capitalize()} data file integrity check failed (MD5 mismatch).\n"
                f"Expected MD5: {expected_md5}\n"
                f"Actual MD5: {actual_md5}\n\n"
                f"The data may be corrupted or outdated. Please re-run preprocessing:\n"
                f"   python data_process.py process -d {self.dataset}"
            )

    def _load_parquet_files(self, sequence_path: str, question_path: str):
        """Load parquet files with error handling."""
        logger.info(f"Loading sequence data: {sequence_path}")
        self.sequence_data = pl.read_parquet(sequence_path)
        logger.info(
            f"Sequence data loaded successfully: {len(self.sequence_data)} rows"
        )

        logger.info(f"Loading question data: {question_path}")
        self.question_data = pl.read_parquet(question_path)
        logger.info(
            f"Question data loaded successfully: {len(self.question_data)} rows"
        )

    @abstractmethod
    def clear_data(self):
        """Clean and preprocess raw data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses should implement clear_data method")

    def save_data(self):
        """Save processed data to parquet files with metadata."""
        logger.info("Saving processed data...")

        if self.sequence_data is None or self.question_data is None:
            raise ValueError("Please run clear_data() before saving processed data.")

        sequence_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_sequence.parquet"
        )
        question_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_question.parquet"
        )

        self.question_data.write_parquet(question_data_path)
        self.add_metadata("question_data_md5", self.compute_md5(question_data_path))

        self.sequence_data.write_parquet(sequence_data_path)
        self.add_metadata("sequence_data_md5", self.compute_md5(sequence_data_path))
        self.add_metadata("random_seed", self.seed)

        self.save_metadata()
        logger.info("Processed data saved.")

    def compute_md5(self, file_path: str) -> str:
        """Compute MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        with open(file_path, "rb") as f:
            with tqdm.tqdm(
                total=file_size_mb,
                unit="MB",
                unit_scale=False,
                desc="Computing MD5",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f}MB [{elapsed}<{remaining}]",
            ) as pbar:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    hash_md5.update(chunk)
                    pbar.update(len(chunk) / (1024 * 1024))

        return hash_md5.hexdigest()

    def get_sequence_data(self):
        """Get sequence data as pandas DataFrame."""
        if self.sequence_data is None:
            self._load_or_raise()
        return self.sequence_data.to_pandas()

    def get_question_data(self):
        """Get question data as pandas DataFrame."""
        if self.question_data is None:
            self._load_or_raise()
        return self.question_data.to_pandas()

    def get_processed_data(self):
        """Get both sequence and question data."""
        if self.sequence_data is None or self.question_data is None:
            self._load_or_raise()
        return self.sequence_data, self.question_data

    def _load_or_raise(self):
        """Load processed data or raise ValueError."""
        try:
            self.load_processed_data()
        except FileNotFoundError:
            raise ValueError(
                "No processed data available. Please run clear_data() first."
            )

    def add_metadata(self, key: str, value):
        """Add a single metadata entry."""
        self.metadata[key] = value
        logger.debug(f"Added {key} = {value} to DataSource metadata")

    def add_metadatas(self, meta_dict: dict):
        """Add multiple metadata entries."""
        for key, value in meta_dict.items():
            self.add_metadata(key, value)

    def save_metadata(self):
        """Save metadata to JSON file."""
        self.add_metadata("dataset", self.dataset)
        self.add_metadata("data_base_path", self.data_base_path)

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def load_metadata(self):
        """Load metadata from JSON file."""
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path) as f:
            self.metadata = json.load(f)

    def get_metadata(self, key: str | None = None):
        """Get metadata entry or entire metadata dict."""
        if not self.metadata:
            self.load_metadata()
        return self.metadata if key is None else self.metadata.get(key)

    def add_kfold_labels(self, n_splits: int = 5):
        """Add K-fold cross-validation labels at user level.

        Ensures all data from the same user stays in the same fold
        to prevent data leakage.

        Args:
            n_splits: Number of folds (default: 5).

        Returns:
            DataFrame with added 'fold' column (values: 0 to n_splits-1).

        Raises:
            ValueError: If sequence_data is not loaded.
        """
        if self.sequence_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )

        logger.info(f"Adding K-Fold labels with n_splits={n_splits}")

        data = self.sequence_data.clone().with_columns([pl.lit(-1).alias("fold")])
        unique_users = data["user"].unique()
        user_to_fold = {}

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for fold_idx, (_, test_user_idx) in tqdm.tqdm(
            enumerate(kfold.split(unique_users)), total=n_splits, desc="Assigning folds"
        ):
            test_users = unique_users[test_user_idx]
            for user in test_users:
                user_to_fold[user] = fold_idx

        data = data.with_columns([pl.col("user").replace(user_to_fold).alias("fold")])
        self.sequence_data = data
        self.add_metadata("kfold_n_splits", n_splits)

        return data

    def get_user_stats(self):
        """Compute user statistics: attempts, correct count, skill count, correct rate.

        Returns:
            DataFrame with columns: user, attempts, correct, skill_count, correct_rate.

        Raises:
            ValueError: If sequence_data is not loaded.
        """
        if self.sequence_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )

        user_stats = self.sequence_data.group_by("user").agg(
            pl.len().alias("attempts"),
            pl.col("label").sum().alias("correct"),
        )

        user_skill = self.sequence_data.select(["user", "question"]).join(
            self.question_data.select(["question", "skill"]),
            on="question",
            how="left",
        )
        skill_stats = user_skill.group_by("user").agg(
            pl.col("skill").n_unique().alias("skill_count")
        )

        user_stats = user_stats.join(skill_stats, on="user", how="left").with_columns(
            [
                pl.col("skill_count").fill_null(0).cast(pl.Int32),
                (pl.col("correct") / pl.col("attempts")).alias("correct_rate"),
            ]
        )

        return user_stats

    def sample_users(
        self,
        n_samples: int,
        stratify: bool = True,
        attempts_bins: list = [20, 100],
        correct_bins: list = [0.4, 0.8],
    ):
        """Sample users with stratified sampling based on user statistics.

        Stratifies by attempts and correct rate dimensions.

        Args:
            n_samples: Number of users to sample.
            stratify: Enable stratified sampling (default: True).
            attempts_bins: Attempt count bin edges.
            correct_bins: Correct rate bin edges.

        Raises:
            ValueError: If n_samples exceeds total users or data not loaded.
        """
        if self.sequence_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )

        logger.info(f"Sampling {n_samples} users from dataset...")

        user_stats = self.get_user_stats()
        total_users = len(user_stats)

        if n_samples > total_users:
            raise ValueError(
                f"Requested sample size ({n_samples}) exceeds total users ({total_users})"
            )

        if n_samples == total_users:
            logger.info("Sample size equals total users, skipping sampling.")
            return

        original_records = len(self.sequence_data)

        if stratify:
            sampled_users = self._sample_users_stratified(
                user_stats, n_samples, attempts_bins, correct_bins
            )
        else:
            logger.info("Performing simple random sampling without stratification.")
            sampled_users = (
                user_stats.sample(n=n_samples, seed=self.seed)
                .select("user")
                .to_series()
                .to_list()
            )

        self._apply_sampling_to_data(
            sampled_users, n_samples, total_users, original_records
        )

    def _sample_users_stratified(
        self,
        user_stats: pl.DataFrame,
        n_samples: int,
        attempts_bins: list,
        correct_bins: list,
    ) -> list:
        """Perform stratified sampling on users."""
        user_stats = user_stats.with_columns(
            [
                (
                    self._make_bin_expr("attempts", attempts_bins) * 3
                    + self._make_bin_expr("correct_rate", correct_bins)
                ).alias("strata"),
                ((pl.col("user").cast(pl.Int64) + self.seed) % 2**31)
                .cast(pl.UInt32)
                .alias("rand_key"),
            ]
        )

        strata_info = user_stats.group_by("strata").agg(pl.len().alias("count"))

        total_in_strata = strata_info.select(pl.sum("count")).item()
        strata_info = strata_info.with_columns(
            ((pl.col("count") / total_in_strata * n_samples).cast(pl.Int32))
            .clip(1)
            .alias("quota")
        )

        strata_info = strata_info.sort("strata")
        logger.info(f"Created {len(strata_info)} strata for stratified sampling.")

        sampled_users = (
            user_stats.join(strata_info.select(["strata", "quota"]), on="strata")
            .sort(["strata", "rand_key"])
            .with_columns(
                pl.int_range(pl.len(), dtype=pl.UInt32).over("strata").alias("rank")
            )
            .filter(pl.col("rank") < pl.col("quota"))
            .select("user")
        )

        n_sampled = len(sampled_users)
        if n_sampled != n_samples:
            if n_sampled > n_samples:
                sampled_users = sampled_users.sample(n=n_samples, seed=self.seed)
            else:
                unsampled = user_stats.join(
                    sampled_users, on="user", how="anti"
                ).select("user")
                additional = unsampled.sample(n=n_samples - n_sampled, seed=self.seed)
                sampled_users = pl.concat([sampled_users, additional])

        return sampled_users.to_series().to_list()

    def _make_bin_expr(self, col: str, bins: list) -> pl.Expr:
        """Create binning expression for a column."""
        return (
            pl.when(pl.col(col) <= bins[0])
            .then(pl.lit(0, dtype=pl.Int8))
            .when(pl.col(col) <= bins[1])
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(2, dtype=pl.Int8))
        )

    def _apply_sampling_to_data(
        self,
        sampled_users: list,
        n_samples: int,
        total_users: int,
        original_records: int,
    ):
        """Apply sampled users to sequence and question data with ID remapping."""
        user_stats = self.get_user_stats()
        strata_distribution = self._compute_strata_distribution(
            sampled_users, user_stats
        )

        self.sequence_data = self.sequence_data.filter(
            pl.col("user").is_in(sampled_users)
        )

        self._remap_user_ids()
        self._remap_question_ids()

        num_users = self.sequence_data.select(pl.col("user").n_unique()).item()
        num_questions = self.question_data.select(pl.col("question").n_unique()).item()
        num_skills = self.question_data.select(pl.col("skill").n_unique()).item()
        sampled_records = len(self.sequence_data)

        sampling_config = {
            "n_samples_requested": n_samples,
            "n_samples_actual": len(sampled_users),
            "sampling_ratio": len(sampled_users) / total_users,
            "stratify": True,
            "attempts_bins": [20, 100],
            "correct_bins": [0.4, 0.8],
        }

        sampling_stats = {
            "original_users": total_users,
            "sampled_users": len(sampled_users),
            "original_records": original_records,
            "sampled_records": sampled_records,
            "strata_distribution": strata_distribution,
        }

        self.add_metadata("sampled", True)
        self.add_metadata("sampling_config", sampling_config)
        self.add_metadata("sampling_stats", sampling_stats)
        self.add_metadata("num_users", int(num_users))
        self.add_metadata("num_questions", int(num_questions))
        self.add_metadata("num_skills", int(num_skills))

        logger.info(
            f"Sampling complete: {len(sampled_users)}/{total_users} users, "
            f"{sampled_records}/{original_records} records"
        )
        logger.info(f"Sampling ratio: {sampling_config['sampling_ratio']:.2%}")

    def _remap_user_ids(self):
        """Remap user IDs to consecutive integers starting from 0."""
        user_id_map = (
            self.sequence_data.select(pl.col("user").unique())
            .sort("user")
            .with_row_index("new_user_id")
            .select([pl.col("user"), pl.col("new_user_id").cast(pl.Int32)])
        )
        self.sequence_data = (
            self.sequence_data.join(user_id_map, on="user", how="left")
            .drop("user")
            .rename({"new_user_id": "user"})
        )

    def _remap_question_ids(self):
        """Filter and remap question IDs to consecutive integers."""
        self.question_data = self.question_data.join(
            self.sequence_data.select(pl.col("question").unique()),
            on="question",
            how="semi",
        )

        question_id_map = (
            self.question_data.select(pl.col("question").unique())
            .sort("question")
            .with_row_index("new_question_id")
            .select([pl.col("question"), pl.col("new_question_id").cast(pl.Int32)])
        )
        self.question_data = (
            self.question_data.join(question_id_map, on="question", how="left")
            .drop("question")
            .rename({"new_question_id": "question"})
        )
        self.sequence_data = (
            self.sequence_data.join(question_id_map, on="question", how="left")
            .drop("question")
            .rename({"new_question_id": "question"})
        )

    def _compute_strata_distribution(
        self, sampled_users: list, user_stats: pl.DataFrame
    ) -> dict:
        """Compute distribution of users across strata."""

        def _strata_to_str(s):
            return f"{s // 3}_{s % 3}"

        strata_to_str = _strata_to_str

        strata_cols = user_stats.columns
        if "strata" not in strata_cols:
            return {}

        strata_info = user_stats.group_by("strata").agg(pl.len().alias("count"))
        strata_distribution = {
            strata_to_str(row["strata"]): {"original": row["count"], "sampled": 0}
            for row in strata_info.iter_rows(named=True)
        }

        sampled_strata = (
            user_stats.filter(pl.col("user").is_in(sampled_users))
            .group_by("strata")
            .agg(pl.len().alias("sampled"))
        )

        for row in sampled_strata.iter_rows(named=True):
            strata_str = strata_to_str(row["strata"])
            if strata_str in strata_distribution:
                strata_distribution[strata_str]["sampled"] = row["sampled"]

        return strata_distribution


def restrains_sequence_length(data, min_seq_len: int, max_seq_len: int = 0):
    """Filter sequences to be within min_seq_len and max_seq_len bounds.

    Args:
        data: Polars DataFrame or LazyFrame.
        min_seq_len: Minimum sequence length.
        max_seq_len: Maximum sequence length (0 or None means no limit).

    Returns:
        DataFrame or LazyFrame of same type as input.
    """
    is_lazy = isinstance(data, pl.LazyFrame)

    if min_seq_len > 1:
        valid_users = (
            data.group_by("user")
            .agg(pl.len().alias("count"))
            .filter(pl.col("count") >= min_seq_len)
            .select("user")
        )
        valid_users = (
            valid_users.collect().to_series() if is_lazy else valid_users.to_series()
        )
        data = data.filter(pl.col("user").is_in(valid_users))

    if max_seq_len is not None and max_seq_len > 0:
        data = _truncate_long_sequences(data, max_seq_len)

    return data


def _truncate_long_sequences(data, max_seq_len: int):
    """Truncate sequences longer than max_seq_len to last max_seq_len records."""
    time_col = _find_time_column(data)

    if time_col:
        data = data.sort(["user", time_col])
        data = data.with_columns(
            pl.arange(0, pl.count(), dtype=pl.UInt32).over("user").alias("row_num")
        )
        data = data.with_columns(pl.col("row_num").max().over("user").alias("total"))
        data = data.filter((pl.col("total") - pl.col("row_num")) < max_seq_len)
        data = data.drop(["row_num", "total"])
    else:
        data = data.group_by("user").map_groups(
            lambda df: df.slice(-max_seq_len, max_seq_len),
            schema=data.schema,
        )

    return data


def _find_time_column(data) -> str | None:
    """Find a suitable time column for sorting sequences."""
    for col in ["timestamp", "order_id", "start_time", "startTime"]:
        if col in data.columns:
            return col
    return None


def map_to_continuous_ids(data, columns: list[str]):
    """Map specified columns to consecutive integer IDs.

    Args:
        data: Polars DataFrame.
        columns: List of column names to map.

    Returns:
        DataFrame with mapped columns.
    """
    for col in columns:
        unique_vals = data[col].unique().sort().to_list()
        value_map = {val: idx for idx, val in enumerate(unique_vals)}
        data = data.with_columns(
            pl.col(col).replace(value_map).cast(pl.Int32).alias(col)
        )
    return data


def build_question_data_from_cleared(
    cleared_data,
    skill_column: str = "skill",
    question_column: str = "question",
    separator: str = None,
):
    """Build question data from cleaned records.

    If separator is provided, splits multi-skill entries into multiple rows.
    Ensures each question-skill pair is unique and maps skills to continuous IDs.

    Args:
        cleared_data: Cleaned polars DataFrame.
        skill_column: Name of skill column (default: "skill").
        question_column: Name of question column (default: "question").
        separator: Separator for multi-skill values (default: None, no splitting).

    Returns:
        DataFrame with unique question-skill pairs and continuous skill IDs.
    """
    if separator is not None:
        other_cols = [col for col in cleared_data.columns if col != skill_column]
        question_data = (
            cleared_data.with_columns(
                [pl.col(skill_column).str.split(separator).alias(skill_column + "_list")]
            )
            .explode(skill_column + "_list")
            .select(other_cols + [pl.col(skill_column + "_list").alias(skill_column)])
            .unique(subset=[question_column, skill_column], keep="first")
        )
    else:
        question_data = cleared_data.unique(
            subset=[question_column, skill_column], keep="first"
        )

    question_data = map_to_continuous_ids(question_data, columns=[skill_column])
    return question_data


__all__ = [
    "DataSource",
    "restrains_sequence_length",
    "map_to_continuous_ids",
    "build_question_data_from_cleared",
]
