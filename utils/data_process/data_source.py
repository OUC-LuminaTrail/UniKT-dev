from abc import ABC, abstractmethod
import os
import pandas
from sklearn.model_selection import KFold
from utils.core import get_logger

logger = get_logger(__name__)


class DataSource(ABC):
    """
    数据源基类

    参数:
        dataset: 数据集名称（自动转换为小写）
        data_base_path: 数据存储的基础路径
        data_url: 数据下载链接 (可选)

    属性:
        dataset: 数据集名称
        data_base_path: 数据存储的基础路径
        data_folder: 数据集文件夹路径
        data_processed_folder: 预处理后数据的存储路径
        raw_data: 原始数据 (Pandas DataFrame)
        processed_data: 预处理后的数据 (Pandas DataFrame)
        data_url: 数据下载链接 (可选)
        metadata: 数据元信息字典

    已实现的方法:
        add_metadata(key, value): 添加数据元信息
        save_metadata(): 保存数据元信息到 JSON 文件
        add_kfold_labels(n_splits, random_state, user_id_column): 添加K折交叉验证的分层划分标签
    """

    def __init__(
        self, dataset: str, data_base_path: str, data_url: str = None, seed: int = 42
    ):
        super().__init__()
        self.dataset = dataset.lower()
        # 数据存储的基础路径
        self.data_base_path = data_base_path
        # 数据集文件夹路径
        self.data_folder = os.path.join(self.data_base_path, self.dataset)
        # 元数据JSON文件路径
        self.metadata_path = os.path.join(self.data_folder, "metadata.json")
        self.raw_data = None
        self.cleared_data = None  # 预处理前的中间数据
        self.sequence_data = None  # 预处理后的答题序列数据
        self.question_data = None  # 预处理后的题目信息数据
        self.data_url = data_url
        self.metadata = {}
        # 设置随机种子
        self.seed = seed
        self.set_random_seed()

    def set_random_seed(self):
        import random
        import numpy as np

        if self.seed is None:
            self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.add_metadata("random_seed", self.seed)

    def _download_chunk(self, url, start, end, chunk_path, pbar, chunk_size=8192):
        """
        下载文件的一个块（用于多线程下载）

        参数:
            url: 下载URL
            start: 起始字节
            end: 结束字节
            chunk_path: 块文件保存路径
            pbar: 进度条对象
            chunk_size: 每次读取的块大小
        """
        import requests

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
        """
        使用requests库下载文件（支持多线程）

        参数:
            archive_path: 文件保存路径
            num_threads: 线程数
            attempt: 当前尝试次数
            max_retries: 最大重试次数
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor
        import tqdm

        # 首先发送HEAD请求检查服务器是否支持Range请求
        head_response = requests.head(self.data_url, timeout=30)
        head_response.raise_for_status()

        total_bytes = int(head_response.headers.get("content-length", 0))
        accept_ranges = head_response.headers.get("accept-ranges", "none")

        # 如果服务器不支持Range或文件太小，使用单线程下载
        if accept_ranges != "bytes" or total_bytes < 10 * 1024 * 1024:  # 小于10MB
            if accept_ranges != "bytes":
                logger.warning(
                    "Server does not support range requests, using single-threaded download"
                )
            self._download_single_thread(archive_path)
            return

        # 多线程下载
        total_mb = total_bytes / (1024 * 1024)
        chunk_size = total_bytes // num_threads

        # 创建临时目录存储分块文件
        temp_dir = archive_path + ".parts"
        import os

        os.makedirs(temp_dir, exist_ok=True)

        try:
            with tqdm.tqdm(
                total=total_mb,
                unit="MB",
                unit_scale=False,
                desc=f"Downloading (attempt {attempt}/{max_retries})",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f}MB [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                # 创建下载任务
                futures = []
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
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
                        futures.append((i, future, chunk_path))

                    # 等待所有任务完成
                    for i, future, chunk_path in futures:
                        future.result()  # 如果有异常会在这里抛出

            # 合并所有分块文件
            logger.debug("Merging downloaded chunks...")
            with open(archive_path, "wb") as outfile:
                for i in range(num_threads):
                    chunk_path = os.path.join(temp_dir, f"chunk_{i}")
                    with open(chunk_path, "rb") as infile:
                        import shutil

                        shutil.copyfileobj(infile, outfile)

        finally:
            # 清理临时文件
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _download_single_thread(self, archive_path):
        """
        单线程下载文件

        参数:
            archive_path: 文件保存路径
        """
        import requests
        import tqdm

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
        """
        下载数据（支持多线程下载、自动重试和强制覆盖）

        参数:
            force_download: 是否强制重新下载（即使文件已存在）
            max_retries: 最大重试次数
            num_threads: 多线程下载的线程数（仅在支持Range请求时使用）
        """
        import os
        import shutil
        from pathlib import Path
        import time

        try:
            import requests
        except Exception:
            requests = None
        import urllib.request
        import tarfile
        import zipfile
        import gzip

        if self.data_url is None:
            raise ValueError("Data URL is not provided.")

        os.makedirs(os.path.join(self.data_base_path, self.dataset), exist_ok=True)

        # 推断文件名
        file_name = self.data_url.split("/")[-1]
        if not file_name:  # 处理以 / 结尾的 URL
            file_name = "downloaded_data"
        archive_path = os.path.join(self.data_folder, file_name)

        # 检查是否需要下载
        if os.path.exists(archive_path) and not force_download:
            logger.info(f"Dataset already exists, skip downloading: {archive_path}")
        else:
            if force_download and os.path.exists(archive_path):
                logger.warning(
                    f"Force download enabled, removing existing file: {archive_path}"
                )
                os.remove(archive_path)

            logger.info(f"Downloading data from {self.data_url}")

            # 尝试下载，支持重试
            for attempt in range(max_retries):
                try:
                    if requests is not None:
                        self._download_with_requests(
                            archive_path, num_threads, attempt + 1, max_retries
                        )
                    else:
                        # urllib 回退
                        logger.warning(
                            "Using urllib as fallback (no multi-threading support)"
                        )
                        urllib.request.urlretrieve(self.data_url, archive_path)

                    logger.info(f"Download finished: {archive_path}")
                    break  # 下载成功，跳出重试循环

                except Exception as e:
                    # 清理失败的下载文件
                    if os.path.exists(archive_path):
                        os.remove(archive_path)
                        logger.debug(f"Removed incomplete download: {archive_path}")

                    if attempt < max_retries - 1:
                        wait_time = 2**attempt  # 指数退避
                        logger.error(
                            f"Download failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"Failed to download data after {max_retries} attempts: {e}"
                        )

        # 计算并保存 MD5
        archive_md5 = self.compute_md5(archive_path)
        self.add_metadata("raw_archive_md5", archive_md5)
        self.add_metadata("raw_archive_filename", file_name)

        # 解压逻辑
        extract_target = os.path.join(self.data_folder, "raw")
        os.makedirs(extract_target, exist_ok=True)

        # 判断是否需要重新解压
        should_extract = False
        if force_download:
            # 强制模式：清空并重新解压
            if any(Path(extract_target).iterdir()):
                logger.warning(
                    f"Force mode enabled, removing existing raw data: {extract_target}"
                )
                shutil.rmtree(extract_target)
                os.makedirs(extract_target, exist_ok=True)
            should_extract = True
        elif not any(Path(extract_target).iterdir()):
            # 目录为空，需要解压
            should_extract = True
        else:
            logger.info(
                f"Raw data directory not empty, skip extraction: {extract_target}"
            )
            return extract_target

        if should_extract:
            logger.info(f"Extracting archive: {archive_path}")
            lower_name = file_name.lower()
            try:
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
                    # 处理单文件 .gz
                    uncompressed_name = lower_name[:-3]
                    target_file = os.path.join(extract_target, uncompressed_name)
                    with (
                        gzip.open(archive_path, "rb") as f_in,
                        open(target_file, "wb") as f_out,
                    ):
                        shutil.copyfileobj(f_in, f_out)
                else:
                    # 非压缩文件，直接复制
                    dest_path = os.path.join(extract_target, file_name)
                    if archive_path != dest_path:
                        shutil.copy2(archive_path, dest_path)
            except Exception as e:
                raise RuntimeError(f"Failed to extract archive: {e}")

            logger.info(f"Extraction finished: {extract_target}")

        self.add_metadata("raw_data_path", extract_target)

    @abstractmethod
    def load_src_data(self):
        """
        加载原始数据
        """
        raise NotImplementedError("Subclasses should implement load_data method")

    def load_processed_data(self):
        """
        加载预处理后的数据

        异常:
            FileNotFoundError: 预处理数据文件不存在
            ValueError: 数据完整性检查失败（MD5不匹配）
        """
        self.load_metadata()
        # 加载预处理后的数据文件
        sequence_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_sequence.parquet"
        )
        question_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_question.parquet"
        )

        # 检查文件是否存在，提供详细的错误信息和修复建议
        missing_files = []
        if not os.path.exists(sequence_data_path):
            missing_files.append(f"  - {sequence_data_path}")
        if not os.path.exists(question_data_path):
            missing_files.append(f"  - {question_data_path}")

        if missing_files:
            missing_str = "\n".join(missing_files)
            raise FileNotFoundError(
                f"Processed data files not found for dataset '{self.dataset}':\n"
                f"{missing_str}\n\n"
                f"💡 To fix this, please run preprocessing first:\n"
                f"   python data_process.py process -d {self.dataset}\n\n"
                f"📁 Data base path: {self.data_folder}"
            )

        # 检测文件的MD5值是否匹配
        md5_hash = self.compute_md5(sequence_data_path)
        if (
            "sequence_data_md5" in self.metadata
            and md5_hash != self.metadata["sequence_data_md5"]
        ):
            raise ValueError(
                f"Processed data file integrity check failed (MD5 mismatch).\n"
                f"Expected MD5: {self.metadata.get('sequence_data_md5', 'unknown')}\n"
                f"Actual MD5: {md5_hash}\n\n"
                f"💡 The data may be corrupted or outdated. Please re-run preprocessing:\n"
                f"   python data_process.py process -d {self.dataset}"
            )
        md5_hash = self.compute_md5(question_data_path)
        if (
            "question_data_md5" in self.metadata
            and md5_hash != self.metadata["question_data_md5"]
        ):
            raise ValueError(
                f"Question data file integrity check failed (MD5 mismatch).\n"
                f"Expected MD5: {self.metadata.get('question_data_md5', 'unknown')}\n"
                f"Actual MD5: {md5_hash}\n\n"
                f"💡 The data may be corrupted or outdated. Please re-run preprocessing:\n"
                f"   python data_process.py process -d {self.dataset}"
            )

        # 加载数据
        self._load_processed_data_safely(sequence_data_path, question_data_path)

    def _load_processed_data_safely(self, sequence_data_path, question_data_path):
        """
        安全加载parquet数据,带详细错误信息

        参数:
            sequence_data_path: 序列数据路径
            question_data_path: 问题数据路径

        异常:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误
            MemoryError: 内存不足
        """
        # 加载序列数据
        try:
            logger.info(f"Loading sequence data: {sequence_data_path}")
            self.sequence_data = pandas.read_parquet(sequence_data_path)
            logger.info(
                f"✓ Sequence data loaded successfully: {self.sequence_data.shape}"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Data file not found: {sequence_data_path}\n"
                f"Please run preprocessing first: python data_process.py process -d {self.dataset}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load sequence data: {e}")
            raise

        # 加载问题数据
        try:
            logger.info(f"Loading question data: {question_data_path}")
            self.question_data = pandas.read_parquet(question_data_path)
            logger.info(
                f"✓ Question data loaded successfully: {self.question_data.shape}"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Data file not found: {question_data_path}\n"
                f"Please run preprocessing first: python data_process.py process -d {self.dataset}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load question data: {e}")
            raise

    @abstractmethod
    def clear_data(self):
        """
        预处理数据

        注：如果原始数据未加载，应先调用 load_src_data()
        处理完成后应将结果存储在 self.processed_data 中
        """
        raise NotImplementedError("Subclasses should implement clear_data method")

    def save_data(self):
        """
        保存预处理后的数据

        注：在该方法中应调用 save_metadata() 保存元信息
        """
        logger.info("Saving processed data...")
        if self.sequence_data is None or self.question_data is None:
            raise ValueError("Please run clear_data() before saving processed data.")

        # 保存预处理后的答题序列数据
        sequence_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_sequence.parquet"
        )
        question_data_path = os.path.join(
            self.data_folder, f"{self.dataset}_question.parquet"
        )
        self.question_data.to_parquet(question_data_path, index=False)
        md5_hash = self.compute_md5(question_data_path)
        self.add_metadata("question_data_md5", md5_hash)
        self.sequence_data.to_parquet(sequence_data_path, index=False)
        md5_hash = self.compute_md5(sequence_data_path)
        self.add_metadata("sequence_data_md5", md5_hash)
        self.save_metadata()
        logger.info("Processed data saved.")

    def compute_md5(self, file_path: str) -> str:
        """
        计算文件的MD5值

        参数:
            file_path: 文件路径

        返回:
            文件的MD5值字符串
        """
        import hashlib
        from tqdm import tqdm

        hash_md5 = hashlib.md5()
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)

        with open(file_path, "rb") as f:
            with tqdm(
                total=file_size_mb,
                unit="MB",
                unit_scale=False,
                desc="Computing MD5",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.2f}/{total:.2f}MB [{elapsed}<{remaining}]",
            ) as pbar:
                while True:
                    chunk = f.read(65536)  # 64KB chunks for faster processing
                    if not chunk:
                        break
                    hash_md5.update(chunk)
                    pbar.update(len(chunk) / (1024 * 1024))

        return hash_md5.hexdigest()

    def get_sequence_data(self):
        """
        获取预处理后的数据

        返回:
            预处理后的数据
        """
        if self.sequence_data is None:
            try:
                self.load_processed_data()
            except FileNotFoundError:
                raise ValueError(
                    "No processed data available. Please run clear_data() first."
                )
        return self.sequence_data

    def get_question_data(self):
        """
        获取题目信息数据

        返回:
            题目信息数据
        """
        if self.question_data is None:
            try:
                self.load_processed_data()
            except FileNotFoundError:
                raise ValueError(
                    "No processed data available. Please run clear_data() first."
                )
        return self.question_data

    def get_processed_data(self):
        """
        获取预处理后的数据和题目信息数据

        返回:
            预处理后的数据和题目信息数据的元组 (sequence_data, question_data)
        """
        if self.sequence_data is None or self.question_data is None:
            try:
                self.load_processed_data()
            except FileNotFoundError:
                raise ValueError(
                    "No processed data available. Please run clear_data() first."
                )
        return self.sequence_data, self.question_data

    def add_metadata(self, key: str, value):
        """
        添加数据元信息

        参数:
            key: 元信息键
            value: 元信息值
        """
        self.metadata[key] = value

    def add_metadatas(self, meta_dict: dict):
        """
        批量添加数据元信息

        参数:
            meta_dict: 元信息字典
        """
        for key, value in meta_dict.items():
            self.add_metadata(key, value)

    def save_metadata(self):
        """
        保存数据元信息
        """
        self.add_metadata("dataset", self.dataset)
        self.add_metadata("data_base_path", self.data_base_path)
        import json

        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

    def load_metadata(self):
        """
        加载数据元信息
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        import json

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

    def get_metadata(self, key: str | None = None):
        """
        获取指定键的元信息

        参数:
            key: 元信息键

        返回:
            元信息值
        """
        if not self.metadata:
            self.load_metadata()
        if key is None:
            return self.metadata
        return self.metadata.get(key, None)

    def add_kfold_labels(self, n_splits: int = 5):
        """
        为数据集添加K折交叉验证的分层划分标签

        按用户维度进行分层 K折划分，确保每个用户的所有数据都在同一个fold中，
        避免数据泄露。

        参数:
            n_splits: K折的数量，默认为5
            random_state: 随机种子，确保可重复性，默认为42

        返回:
            添加了fold标签的数据集 DataFrame（列名为 'fold'，值为 0 到 n_splits-1）

        异常:
            ValueError: 如果processed_data未加载或user_id_column列不存在

        说明:
            - 添加的新列名为 'fold'（值为 0 到 n_splits-1）
            - 如果按用户分层，每个用户的所有数据都会分到同一个fold
            - 元数据中会记录 'kfold_n_splits' 和 'kfold_random_state'
            - 会覆盖已存在的 'fold' 列
        """
        from tqdm import tqdm

        if self.sequence_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )

        logger.info(f"Adding K-Fold labels with n_splits={n_splits}")

        # 复制数据以避免修改原始数据
        data = self.sequence_data.copy()
        data["fold"] = -1

        # 获取唯一的用户ID
        unique_users = data["user"].unique()
        user_to_fold = {}

        # 对用户ID进行KFold划分
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        for fold_idx, (_, test_user_idx) in tqdm(
            enumerate(kfold.split(unique_users)), total=n_splits, desc="Assigning folds"
        ):
            test_users = unique_users[test_user_idx]
            for user in test_users:
                user_to_fold[user] = fold_idx

        # 为每个用户的所有行分配对应的fold值
        data["fold"] = data["user"].map(user_to_fold)

        # 更新processed_data
        self.sequence_data = data

        # 更新元数据
        self.add_metadata("kfold_n_splits", n_splits)

        return data


def restrains_sequence_length(data, min_seq_len: int, max_seq_len: int = 0):
    """
    限制序列长度在min_seq_len和max_seq_len之间
    """
    # 过滤答题次数少于min_seq_len的学生
    if min_seq_len > 1:
        is_valid_user = data.groupby("user").size() >= min_seq_len
        valid_user_ids = is_valid_user[is_valid_user].index.tolist()
        data = data[data["user"].isin(valid_user_ids)].reset_index(drop=True)

    # 答题次数多于max_seq_len的学生将多余的记录删除
    if max_seq_len is not None:
        # 保留每个用户的最后max_seq_len条记录
        data = data.groupby("user", group_keys=False).tail(max_seq_len)
        data = data.reset_index(drop=True)
    return data


def map_to_continuous_ids(data, columns: list[str]):
    """
    将指定列映射为连续的整数ID

    参数:
        data: 输入数据 DataFrame
        columns: 需要映射的列名列表

    返回:
        映射后的数据 DataFrame
    """
    for col in columns:
        data[col] = data[col].astype("category").cat.codes.astype(int)
    return data


def build_question_data_from_cleared(
    cleared_data,
    skill_column: str = "skill",
    question_column: str = "question",
    seperator: str = "_",
):
    """
    从清理后的数据中构建题目信息数据

    该方法会处理技能列中可能存在的多技能情况（使用_分隔）
    并将其展开为多行，确保每个问题-技能对唯一

    参数:
        cleared_data: 清理后的数据 DataFrame
        skill_column: 技能列名，默认为"skill"
        question_column: 问题列名，默认为"question"
        seperator: 多技能分隔符，默认为"_"

    返回:
        处理后的题目信息数据 DataFrame
    """
    data = cleared_data.copy()

    # 检查技能列是否包含多技能（以_分隔）
    if (
        data[skill_column].dtype == "object"
        and data[skill_column].str.contains(seperator).any()
    ):
        # 技能ID列是seperator分隔的多个技能组成，将其展开为多行
        data_expanded = (
            data.assign(**{skill_column: data[skill_column].str.split(seperator)})
            .explode(skill_column)
            .drop_duplicates(subset=[question_column, skill_column])
            .reset_index(drop=True)
        )
        # 转换为整数类型
        data_expanded[skill_column] = data_expanded[skill_column].astype(int)
    else:
        # 直接去重
        data_expanded = data.drop_duplicates(
            subset=[question_column, skill_column]
        ).reset_index(drop=True)

    # 将技能映射为连续ID
    data_expanded = map_to_continuous_ids(data_expanded, columns=[skill_column])

    return data_expanded


__all__ = [
    "DataSource",
    "restrains_sequence_length",
    "map_to_continuous_ids",
    "build_question_data_from_cleared",
]
