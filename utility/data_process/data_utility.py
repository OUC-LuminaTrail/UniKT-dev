from abc import ABC, abstractmethod
import os
from sklearn.model_selection import KFold


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
        self.processed_data = None
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
                print("Server does not support range requests, using single-threaded download")
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
                        end = start + chunk_size - 1 if i < num_threads - 1 else total_bytes - 1
                        chunk_path = os.path.join(temp_dir, f"chunk_{i}")
                        
                        future = executor.submit(
                            self._download_chunk,
                            self.data_url,
                            start,
                            end,
                            chunk_path,
                            pbar
                        )
                        futures.append((i, future, chunk_path))
                    
                    # 等待所有任务完成
                    for i, future, chunk_path in futures:
                        future.result()  # 如果有异常会在这里抛出
            
            # 合并所有分块文件
            print("Merging downloaded chunks...")
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
            print(f"Dataset already exists, skip downloading: {archive_path}")
        else:
            if force_download and os.path.exists(archive_path):
                print(f"Force download enabled, removing existing file: {archive_path}")
                os.remove(archive_path)
            
            print(f"Downloading data from {self.data_url}")
            
            # 尝试下载，支持重试
            for attempt in range(max_retries):
                try:
                    if requests is not None:
                        self._download_with_requests(
                            archive_path, num_threads, attempt + 1, max_retries
                        )
                    else:
                        # urllib 回退
                        print("Using urllib as fallback (no multi-threading support)")
                        urllib.request.urlretrieve(self.data_url, archive_path)
                    
                    print(f"Download finished: {archive_path}")
                    break  # 下载成功，跳出重试循环
                    
                except Exception as e:
                    # 清理失败的下载文件
                    if os.path.exists(archive_path):
                        os.remove(archive_path)
                        print(f"Removed incomplete download: {archive_path}")
                    
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # 指数退避
                        print(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"Retrying in {wait_time} seconds...")
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

        # 判断是否已经解压
        if any(Path(extract_target).iterdir()):
            print(f"Raw data directory not empty, skip extraction: {extract_target}")
            return extract_target

        print(f"Extracting archive: {archive_path}")
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
                with gzip.open(archive_path, "rb") as f_in, open(
                    target_file, "wb"
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            else:
                # 非压缩文件，直接复制
                dest_path = os.path.join(extract_target, file_name)
                if archive_path != dest_path:
                    shutil.copy2(archive_path, dest_path)
        except Exception as e:
            raise RuntimeError(f"Failed to extract archive: {e}")

        print(f"Extraction finished: {extract_target}")
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
        """
        import pandas

        self.load_metadata()
        data_processed_path = os.path.join(
            self.data_folder, f"{self.dataset}_processed.parquet"
        )
        if not os.path.exists(data_processed_path):
            raise FileNotFoundError(
                f"Cannot find processed data file: {data_processed_path}"
            )
        # 检测文件的MD5值是否匹配
        md5_hash = self.compute_md5(data_processed_path)
        if (
            "processed_data_md5" in self.metadata
            and md5_hash != self.metadata["processed_data_md5"]
        ):
            raise ValueError(
                "Processed data file integrity check failed (MD5 mismatch)."
            )
        self.processed_data = pandas.read_parquet(data_processed_path)

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
        print("Saving processed data...")
        if self.processed_data is None:
            raise ValueError("Please run clear_data() before saving processed data.")

        data_processed_path = os.path.join(
            self.data_folder, f"{self.dataset}_processed.parquet"
        )
        self.processed_data.to_parquet(data_processed_path, index=False)
        md5_hash = self.compute_md5(data_processed_path)
        self.add_metadata("processed_data_md5", md5_hash)
        self.save_metadata()
        print("Processed data saved to:", data_processed_path)

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

    def get_processed_data(self):
        """
        获取预处理后的数据

        返回:
            预处理后的数据
        """
        if self.processed_data is None:
            try:
                self.load_processed_data()
            except FileNotFoundError:
                raise ValueError(
                    "No processed data available. Please run clear_data() first."
                )
        return self.processed_data

    def add_metadata(self, key: str, value):
        """
        添加数据元信息

        参数:
            key: 元信息键
            value: 元信息值
        """
        self.metadata[key] = value

    def save_metadata(self):
        """
        保存数据元信息

        必须保存的元信息:
        - num_users: 学生总数
        - num_questions: 题目总数
        - num_skills: 技能总数
        - max_seq_len: 最大序列长度
        - min_seq_len: 最小序列长度
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

        if self.processed_data is None:
            raise ValueError(
                "No processed data available. Please call load_processed_data() or clear_data() first."
            )

        print(f"Adding K-Fold labels with n_splits={n_splits}")

        # 复制数据以避免修改原始数据
        data = self.processed_data.copy()
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
        self.processed_data = data

        # 更新元数据
        self.add_metadata("kfold_n_splits", n_splits)

        return data

    @staticmethod
    def restrains_sequence_length(data, min_seq_len: int, max_seq_len: int):
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

    @staticmethod
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


class ModelData:
    r"""
    模型数据基类

    参数:
        data_src: 数据源对象
    """

    def __init__(self, data_src: DataSource):
        self.data_src = data_src
        self.data_src.load_processed_data()

    def get_kfold_split_data(self, *arrays, fold_idx: int):
        r"""
        根据K折交叉验证的fold索引获取训练集和验证集

        参数:
            *arrays: 任意个数、首维为样本数的数组或张量（与 split_data 一致）。
                     例如：
                     - GIKT: (sequences, responses, masks)
                     - SQGKT: (sequences, responses, masks, user_id_sequence)
            fold_idx: 当前的fold索引（关键字参数，必填）。

        返回:
            train_data: 与输入相同结构的元组，包含训练集切片
            val_data:   与输入相同结构的元组，包含验证集切片

        说明:
            - 需要数据源中已添加K折标签（通过 add_kfold_labels）
            - 验证集为指定fold的数据，训练集为其他fold的数据
            - 需要数据源中有用户到行索引的映射信息
        """
        from tqdm import tqdm
        import numpy as np

        if len(arrays) == 0:
            raise ValueError(
                "get_kfold_split_data requires at least one input array/tensor"
            )

        # 加载数据以获取折信息
        data = self.data_src.get_processed_data()

        # 检查是否已添加fold列
        if "fold" not in data.columns:
            raise ValueError(
                "K-fold labels not found in data. Please call data_src.add_kfold_labels() first."
            )

        # 获取有效的用户索引（基于序列中实际存在的用户）
        num_users = arrays[0].shape[0]

        # 校验所有输入的首维一致
        for i, arr in enumerate(arrays):
            if arr.shape[0] != num_users:
                raise ValueError(
                    f"第 {i} 个输入首维为 {arr.shape[0]}，与预期的 {num_users} 不一致"
                )

        # 创建用户fold信息映射
        user_folds = np.ones(num_users, dtype=int) * -1
        for row in tqdm(
            data.itertuples(),
            total=data.shape[0],
            desc=f"Mapping users to fold {fold_idx}",
        ):
            user_idx = row.user
            fold_label = row.fold
            if user_idx < num_users:
                user_folds[user_idx] = fold_label

        # 根据fold标签分割用户数据
        val_user_indices = np.where(user_folds == fold_idx)[0]
        train_user_indices = np.where(user_folds != fold_idx)[0]

        # 过滤掉fold标签为-1的用户（不在fold中的用户）
        val_user_indices = val_user_indices[val_user_indices < num_users]
        train_user_indices = train_user_indices[train_user_indices < num_users]

        # 索引列表
        val_idx_list = val_user_indices.tolist()
        train_idx_list = train_user_indices.tolist()

        train_slices = []
        val_slices = []
        for arr in arrays:
            # 识别 torch.Tensor
            is_torch_tensor = False
            try:
                import torch  # noqa: F401

                is_torch_tensor = hasattr(arr, "dim") and hasattr(arr, "index_select")
            except Exception:
                is_torch_tensor = False

            if is_torch_tensor:
                import torch

                train_idx = torch.tensor(
                    train_idx_list, dtype=torch.long, device=arr.device
                )
                val_idx = torch.tensor(
                    val_idx_list, dtype=torch.long, device=arr.device
                )
                train_slices.append(arr.index_select(0, train_idx))
                val_slices.append(arr.index_select(0, val_idx))
            else:
                train_slices.append(arr[train_idx_list])
                val_slices.append(arr[val_idx_list])

        return tuple(train_slices), tuple(val_slices)

    def split_data(self, *arrays, val_ratio: float = 0.2):
        r"""
        随机划分训练集和验证集（支持可变数量的输入数组/张量）。

        参数:
            *arrays: 任意个数、首维为样本数的数组或张量。
                     例如：
                     - GIKT: (sequences, responses, masks)
                     - SQGKT: (sequences, responses, masks, user_id_sequence)
            val_ratio: 验证集比例(默认为0.2)

        返回:
            (train_data, val_data):
                - train_data: 与输入相同结构的元组，包含训练集切片
                - val_data:   与输入相同结构的元组，包含验证集切片

        说明:
            - 将依据第一个输入的首维作为样本维度进行打乱与划分。
            - 要求所有输入的首维大小一致。
            - 同时兼容 numpy.ndarray 与 torch.Tensor（若可用）。
            - GIKT 可仅传三项；SQGKT 可传四项（包含 user_id_sequence）。
        """
        import numpy as np

        if len(arrays) == 0:
            raise ValueError("split_data requires at least one input array/tensor")

        num_users = arrays[0].shape[0]

        # 校验所有数组首维一致
        for i, arr in enumerate(arrays):
            if arr.shape[0] != num_users:
                raise ValueError(
                    f"第 {i} 个输入首维为 {arr.shape[0]}，与预期的 {num_users} 不一致"
                )

        indices = np.arange(num_users)
        np.random.shuffle(indices)
        indices = indices.tolist()

        val_size = int(num_users * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # 兼容 numpy 与 torch 的索引切片
        train_slices = []
        val_slices = []
        for arr in arrays:
            # 尝试识别 torch.Tensor
            is_torch_tensor = False
            try:
                import torch  # noqa: F401

                is_torch_tensor = hasattr(arr, "dim") and hasattr(arr, "index_select")
            except Exception:
                is_torch_tensor = False

            if is_torch_tensor:
                import torch

                train_idx = torch.tensor(
                    train_indices, dtype=torch.long, device=arr.device
                )
                val_idx = torch.tensor(val_indices, dtype=torch.long, device=arr.device)
                train_slices.append(arr.index_select(0, train_idx))
                val_slices.append(arr.index_select(0, val_idx))
            else:
                # 视作 numpy 数组或支持 list 索引的结构
                train_slices.append(arr[train_indices])
                val_slices.append(arr[val_indices])

        train_data = tuple(train_slices)
        val_data = tuple(val_slices)

        return train_data, val_data

    @abstractmethod
    def prepare_data(self, args):
        """
        准备模型所需的数据
        """
        raise NotImplementedError("Subclasses should implement prepare_data method")

    def build_sequence_data(self, max_seq_len: int, min_seq_len: int):
        from tqdm import tqdm
        import numpy as np

        data = self.data_src.get_processed_data()
        num_users = self.data_src.get_metadata("num_users")

        # 构建用户答题序列
        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        # 构建用户ID序列
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        # 用户作答正确与否序列
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        # 序列掩码，用于区分是否存在作答数据
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)
        # 用户序列长度计数器，用于索引
        num_sequence = [0] * num_users

        for row in tqdm(
            data.itertuples(), total=data.shape[0], desc="Building user sequences"
        ):
            # 获取用户ID、问题ID和作答正确与否
            user_idx = row.user
            question_idx = row.question
            label = row.label
            # 如果当前用户的序列长度未达到最大长度，则添加数据
            if num_sequence[user_idx] < max_seq_len:
                user_sequence[user_idx, num_sequence[user_idx]] = question_idx
                user_id_sequence[user_idx, num_sequence[user_idx]] = user_idx
                user_response[user_idx, num_sequence[user_idx]] = label
                user_mask[user_idx, num_sequence[user_idx]] = 1
                # 自增对应的用户序列长度
                num_sequence[user_idx] += 1

        return user_sequence, user_response, user_mask, user_id_sequence

    def build_data_matrix(
        self, edge_type: tuple[str, str, str], value_type: str = "binary"
    ):
        """
        构建实体之间的关系矩阵

        参数:
            edge_type: 边类型三元组 (源节点类型, 边关系名, 目标节点类型)
                      节点类型对应数据中的列名（如 'user', 'question', 'skill', 'template', 'assignment'等）
                      例如: ('user', 'answers', 'question')
                           ('question', 'has', 'skill')
                           ('question', 'belongs_to', 'template')
                           ('skill', 'related_to', 'assignment')
            value_type: 矩阵值类型，可选:
                       'binary': 二值矩阵,表示是否存在关系 (默认)
                       'count': 计数矩阵,表示关系出现的次数

        返回:
            data_matrix: numpy数组,形状为 (源节点数量, 目标节点数量)

        示例:
            # 构建用户-问题二值关系矩阵
            matrix = model_data.build_data_matrix(('user', 'answers', 'question'))

            # 构建问题-技能关系矩阵
            matrix = model_data.build_data_matrix(('question', 'has', 'skill'))

            # 构建问题-模板关系矩阵
            matrix = model_data.build_data_matrix(('question', 'belongs_to', 'template'))

            # 构建技能-作业关系矩阵
            matrix = model_data.build_data_matrix(('skill', 'related_to', 'assignment'))

            # 构建用户-问题计数矩阵
            matrix = model_data.build_data_matrix(('user', 'answers', 'question'), value_type='count')
        """
        import numpy as np
        from tqdm import tqdm

        data = self.data_src.get_processed_data()

        src_type, _, dst_type = edge_type
        
        # 直接使用节点类型作为列名
        src_col = src_type
        dst_col = dst_type

        # 验证列是否存在
        if src_col not in data.columns or dst_col not in data.columns:
            raise ValueError(
                f"Required columns '{src_col}' or '{dst_col}' not found in data. "
                f"Available columns: {data.columns.tolist()}"
            )

        # 获取节点数量
        # 首先尝试从元数据获取
        src_meta_key = f"num_{src_type}s"
        dst_meta_key = f"num_{dst_type}s"
        
        try:
            num_src = self.data_src.get_metadata(src_meta_key)
        except (KeyError, AttributeError):
            # 如果元数据中没有，从数据中计算
            num_src = data[src_col].nunique()
            print(f"Warning: {src_meta_key} not found in metadata, calculated from data: {num_src}")
        
        try:
            num_dst = self.data_src.get_metadata(dst_meta_key)
        except (KeyError, AttributeError):
            # 如果元数据中没有，从数据中计算
            num_dst = data[dst_col].nunique()
            print(f"Warning: {dst_meta_key} not found in metadata, calculated from data: {num_dst}")

        # 初始化矩阵
        data_matrix = np.zeros((num_src, num_dst), dtype=int)

        # 填充矩阵
        for row in tqdm(
            data.itertuples(),
            total=data.shape[0],
            desc=f"Building {src_type}-{dst_type} matrix",
        ):
            src_idx = getattr(row, src_col)
            dst_idx = getattr(row, dst_col)
            
            # 跳过无效索引（NaN或超出范围）
            if (
                src_idx is None or dst_idx is None or
                np.isnan(src_idx) or np.isnan(dst_idx) or
                src_idx < 0 or dst_idx < 0 or
                src_idx >= num_src or dst_idx >= num_dst
            ):
                continue

            if value_type == "binary":
                data_matrix[int(src_idx), int(dst_idx)] = 1
            elif value_type == "count":
                data_matrix[int(src_idx), int(dst_idx)] += 1
            else:
                raise ValueError(
                    f"Unsupported value_type: {value_type}. "
                    f"Supported types: 'binary', 'count'"
                )

        return data_matrix

    def build_hetero_graph(
        self,
        edge_types: list[tuple[str, str, str]],
        edge_attrs: dict[tuple[str, str, str], list[str]] = None,
        directed: bool = False,
        node_features: dict[str, any] = None,
    ):
        """
        构建异构图，支持灵活配置节点类型和边类型

        参数:
            edge_types: 边类型列表，每个元素为三元组 (源节点类型, 边关系名, 目标节点类型)
                       例如: [('user', 'answers', 'question'), ('question', 'has', 'skill')]
            edge_attrs: 边属性字典，键为边类型三元组，值为属性列名列表
                       例如: {('user', 'answers', 'question'): ['label', 'order_id']}
                       默认为 None（不添加边属性）
            directed: 是否构建有向图，默认为 False（无向图）
            node_features: 节点特征字典，键为节点类型，值为特征张量或None
                          例如: {'question': question_difficulty_tensor}
                          默认使用节点ID作为特征

        返回:
            HeteroData: PyTorch Geometric 异构图对象

        示例:
            # 示例1: 构建问题-技能无向图
            graph = model_data.build_hetero_graph(
                edge_types=[('question', 'has', 'skill')],
                directed=False
            )

            # 示例2: 构建学生-问题和问题-技能的组合图
            graph = model_data.build_hetero_graph(
                edge_types=[
                    ('user', 'answers', 'question'),
                    ('question', 'has', 'skill')
                ],
                directed=False
            )

            # 示例3: 构建带边属性的图
            graph = model_data.build_hetero_graph(
                edge_types=[('user', 'answers', 'question')],
                edge_attrs={('user', 'answers', 'question'): ['label', 'order_id']},
                directed=True
            )
        """
        from tqdm import tqdm
        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import ToUndirected
        import numpy as np
        import torch

        if edge_attrs is None:
            edge_attrs = {}

        # 获取数据
        data = self.data_src.get_processed_data()
        graph = HeteroData()

        # 收集所有需要的节点类型
        node_types = set()
        for src_type, _, dst_type in edge_types:
            node_types.add(src_type)
            node_types.add(dst_type)

        # 获取每种节点类型的数量
        node_counts = {}
        for node_type in node_types:
            if node_type in data.columns:
                node_counts[node_type] = data[node_type].nunique()
            else:
                # 尝试从元数据获取
                meta_key = f"num_{node_type}s"
                node_counts[node_type] = self.data_src.get_metadata(meta_key)

        # 设置节点数量和特征
        for node_type in node_types:
            graph[node_type].num_nodes = node_counts[node_type]

            # 设置节点特征
            if node_features and node_type in node_features:
                graph[node_type].x = node_features[node_type]
            else:
                # 默认使用节点ID作为特征
                graph[node_type].x = (
                    torch.arange(node_counts[node_type]).view(-1, 1).float()
                )

        # 为每种边类型构建边
        for edge_type in edge_types:
            src_type, relation, dst_type = edge_type
            src_col = f"{src_type}"
            dst_col = f"{dst_type}"

            # 检查列是否存在
            if src_col not in data.columns or dst_col not in data.columns:
                print(
                    f"Warning: Columns {src_col} or {dst_col} not found in data. Skipping edge type {edge_type}"
                )
                continue

            # 收集边和边属性
            edge_dict = {}  # 使用字典存储边，key为(src, dst)，value为属性字典

            # 需要提取的属性列
            attr_cols = edge_attrs.get(edge_type, [])

            # 选择需要的列
            cols_to_select = [src_col, dst_col] + attr_cols

            # 遍历数据构建边
            for row in tqdm(
                data[cols_to_select].itertuples(index=False),
                total=data.shape[0],
                desc=f"Building {src_type}-{relation}-{dst_type} edges",
            ):
                src_id = getattr(row, src_col)
                dst_id = getattr(row, dst_col)
                edge_key = (src_id, dst_id)

                # 如果边已存在，更新属性（取最后一次）
                if attr_cols:
                    edge_attrs_dict = {attr: getattr(row, attr) for attr in attr_cols}
                    edge_dict[edge_key] = edge_attrs_dict
                else:
                    edge_dict[edge_key] = None

            # 转换为张量格式
            edge_list = list(edge_dict.keys())
            if len(edge_list) == 0:
                print(f"Warning: No edges found for {edge_type}")
                continue

            edge_index = np.array(edge_list, dtype=np.int64).T
            edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

            # 添加边索引到图
            graph[src_type, relation, dst_type].edge_index = edge_index

            # 添加边属性
            if attr_cols:
                for attr in attr_cols:
                    attr_values = [edge_dict[edge][attr] for edge in edge_list]
                    attr_tensor = torch.tensor(attr_values, dtype=torch.float32)
                    # 边属性存储为 edge_attr_<attr_name>
                    setattr(
                        graph[src_type, relation, dst_type],
                        f"edge_attr_{attr}",
                        attr_tensor,
                    )

        # 如果需要无向图，应用转换
        if not directed:
            graph = ToUndirected()(graph)

        return graph

    def save_graph(self, graph, file_path: str):
        """
        保存异构图到文件

        参数:
            graph: HeteroData 图对象
            file_path: 保存路径（.pt 文件）
        """
        import torch
        import os

        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 保存图
        torch.save(graph, file_path)
        print(f"Graph saved to: {file_path}")

    def load_graph(self, file_path: str):
        """
        从文件加载异构图

        参数:
            file_path: 图文件路径（.pt 文件）

        返回:
            HeteroData: 加载的图对象
        """
        import torch
        import os

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Graph file not found: {file_path}")

        graph = torch.load(file_path)
        print(f"Graph loaded from: {file_path}")

        return graph

    def build_hypergraph(
        self,
        edge_type: tuple[str, str, str],
        vertex_type: str = None,
    ):
        """
        构建超图，支持灵活配置超边类型
        
        超图定义：
            - 顶点(vertices): 通常是问题(question)节点
            - 超边(hyperedges): 每个超边连接一组相关的顶点
              例如：具有相同知识点/技能的题目、属于相同模板的题目等
        
        参数:
            edge_type: 边类型三元组 (顶点类型, 边关系名, 超边类型)
                      例如: ('question', 'has', 'skill') - 知识点超边
                           ('question', 'belongs_to', 'template') - 模板超边
                           ('question', 'in', 'assignment') - 作业超边
            vertex_type: 顶点类型（可选），默认使用 edge_type 的第一个元素
                        通常是 'question'
        
        返回:
            dhg.Hypergraph: DHG框架的超图对象
        
        工作原理:
            1. 从数据中提取顶点-超边的关联关系
            2. 将相同超边类型(如相同skill_id)的所有顶点分组
            3. 每组顶点形成一个超边
            4. 使用DHG框架创建超图对象
        
        示例:
            # 构建知识点超图：每个知识点连接包含它的所有题目
            skill_hg = model_data.build_hypergraph(
                ('question', 'has', 'skill')
            )
            
            # 构建模板超图：每个模板连接属于它的所有题目
            template_hg = model_data.build_hypergraph(
                ('question', 'belongs_to', 'template')
            )
            
            # 构建作业超图：每个作业连接其中的所有题目
            assignment_hg = model_data.build_hypergraph(
                ('question', 'in', 'assignment')
            )
        """
        from dhg import Hypergraph
        from tqdm import tqdm
        import numpy as np

        vertex_node_type, relation, hyperedge_node_type = edge_type
        
        # 如果未指定顶点类型，使用边类型的第一个元素
        if vertex_type is None:
            vertex_type = vertex_node_type
        
        # 获取关联矩阵
        H = self.build_data_matrix(edge_type, value_type="binary")
        
        # 获取顶点数量
        num_vertices = H.shape[0]
        
        # 将关联矩阵转换为超边列表
        rows, cols = np.nonzero(H)
        
        # 按列（超边类型）分组，每个超边类型对应一个超边
        # 使用字典收集每个超边包含的顶点
        edge_dict = {}
        for vertex_idx, hyperedge_idx in tqdm(
            zip(rows, cols), 
            total=len(rows), 
            desc=f"Building {hyperedge_node_type} hyperedges"
        ):
            if hyperedge_idx not in edge_dict:
                edge_dict[hyperedge_idx] = []
            edge_dict[hyperedge_idx].append(int(vertex_idx))
        
        # 转换为超边列表（过滤空超边）
        e_list = [vertices for vertices in edge_dict.values() if len(vertices) > 0]
        
        # 处理没有超边的情况
        if len(e_list) == 0:
            print(f"Warning: No hyperedges found for {edge_type}. Creating self-loop hypergraph.")
            # 创建自环超图：每个顶点自成一个超边
            e_list = [[i] for i in range(num_vertices)]
        
        # 使用 DHG 框架创建超图
        hypergraph = Hypergraph(num_v=num_vertices, e_list=e_list)
        
        print(f"{hyperedge_node_type.capitalize()} Hypergraph constructed:")
        print(f"  - Number of vertices ({vertex_type}s): {hypergraph.num_v}")
        print(f"  - Number of hyperedges ({hyperedge_node_type}s): {hypergraph.num_e}")
        
        return hypergraph

    def build_multiple_hypergraphs(
        self,
        edge_types: list[tuple[str, str, str]],
        vertex_type: str = None,
    ):
        """
        批量构建多个超图
        
        参数:
            edge_types: 边类型列表，每个元素为三元组 (顶点类型, 边关系名, 超边类型)
                       例如: [
                           ('question', 'has', 'skill'),
                           ('question', 'belongs_to', 'template'),
                           ('question', 'in', 'assignment')
                       ]
            vertex_type: 顶点类型（可选），默认使用每个edge_type的第一个元素
        
        返回:
            dict: 字典，键为超边类型名称，值为对应的超图对象
                 例如: {
                     'skill': skill_hypergraph,
                     'template': template_hypergraph,
                     'assignment': assignment_hypergraph
                 }
        
        示例:
            # 批量构建多个超图
            hypergraphs = model_data.build_multiple_hypergraphs([
                ('question', 'has', 'skill'),
                ('question', 'belongs_to', 'template'),
            ])
            
            skill_hg = hypergraphs['skill']
            template_hg = hypergraphs['template']
        """
        hypergraphs = {}
        
        for edge_type in edge_types:
            _, _, hyperedge_type = edge_type
            hypergraph = self.build_hypergraph(edge_type, vertex_type=vertex_type)
            hypergraphs[hyperedge_type] = hypergraph
        
        return hypergraphs
        return graph

    @staticmethod
    def save_numpy_data(file_path: str, data: tuple):
        """
        保存numpy数据到文件

        参数:
            file_path: 文件路径
            data: 需要保存的数据元组
        """
        import numpy as np

        np.savez_compressed(file_path, *data)
