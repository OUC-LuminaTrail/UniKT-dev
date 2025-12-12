from .data_source import *
import os
import pandas as pd
from tqdm import tqdm
from typing_extensions import override
import pyarrow.csv as pv
import pyarrow as pa
import concurrent.futures
import itertools
import multiprocessing
import threading


class EdNetKT1Data(DataSource):
    """
    EdNetKT1数据集处理类
    """

    def __init__(self, args):
        super().__init__(
            dataset="ednet",
            data_base_path=args.data_base_path,
            data_url="http://cdn.lionhao.top/KTDataset/EdNetKT1.zip",
            seed=args.seed,
        )
        self.args = args
        # 原始数据文件夹路径
        self.raw_data_folder = os.path.join(self.data_folder, "raw")

    @override
    def load_src_data(self):
        # 实现数据加载逻辑
        if not os.path.exists(self.raw_data_folder):
            raise FileNotFoundError(f"Cannot find: {self.raw_data_folder}")
        print("Loading raw data from:", self.raw_data_folder)

        # 读取题目信息
        self.question_data_path = os.path.join(
            self.raw_data_folder, "EdNet-Contents", "questions.csv"
        )
        if not os.path.exists(self.question_data_path):
            raise FileNotFoundError(f"Cannot find: {self.question_data_path}")
        self.question_data_raw = pd.read_csv(self.question_data_path, low_memory=False)

        # 记录响应数据文件夹路径
        self.response_data_path = os.path.join(self.raw_data_folder, "EdNet-KT1", "KT1")
        if not os.path.exists(self.response_data_path):
            raise FileNotFoundError(f"Cannot find: {self.response_data_path}")

        # 检查是否开启调试模式
        is_debug = hasattr(self.args, "debug") and self.args.debug
        debug_limit = 200 if is_debug else None
        if is_debug:
            print(f"Debug mode enabled: processing only {debug_limit} files.")

        # 统计文件总数
        total_files = 0
        with os.scandir(self.response_data_path) as it:
            for entry in it:
                if entry.name.endswith(".csv") and entry.is_file():
                    total_files += 1
                    if debug_limit and total_files >= debug_limit:
                        break

        # 并行处理文件
        processed_batches = []

        # 使用os.scandir生成文件路径
        def file_path_generator():
            count = 0
            with os.scandir(self.response_data_path) as it:
                for entry in it:
                    if entry.name.endswith(".csv") and entry.is_file():
                        yield entry.path
                        count += 1
                        if debug_limit and count >= debug_limit:
                            break

        # 生成chunk的辅助函数
        def chunked(iterable, size):
            it = iter(iterable)
            while True:
                chunk = tuple(itertools.islice(it, size))
                if not chunk:
                    break
                yield chunk

        # 配置并行
        max_workers = os.cpu_count()
        if max_workers is None:
            max_workers = 4

        # 增大chunk size以减少开销
        CHUNK_SIZE = 5000

        print(f"Starting parallel processing with {max_workers} workers...")

        # 使用Manager Queue进行进程间通信
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        # 启动进度条更新线程
        pbar = tqdm(total=total_files, desc="Loading files", unit="file")
        stop_event = threading.Event()

        def update_progress():
            while not stop_event.is_set() or not progress_queue.empty():
                try:
                    # 尝试获取进度更新，超时时间短一点以免阻塞退出
                    count = progress_queue.get(timeout=0.1)
                    pbar.update(count)
                except:
                    continue

        progress_thread = threading.Thread(target=update_progress)
        progress_thread.start()

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = []

                # 提交任务
                for chunk in chunked(file_path_generator(), CHUNK_SIZE):
                    future = executor.submit(process_chunk, chunk, progress_queue)
                    futures.append(future)

                # 获取结果
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res = future.result()
                        if res is not None:
                            processed_batches.append(res)
                    except Exception as e:
                        print(f"Error in worker: {e}")
        finally:
            # 停止进度条线程
            stop_event.set()
            progress_thread.join()
            pbar.close()

        if not processed_batches:
            raise ValueError("No data processed from EdNet files.")

        print("Concatenating all batches...")
        self.sequence_data_raw = pa.concat_tables(processed_batches).to_pandas()
        print(f"Loaded {len(self.sequence_data_raw)} raw interactions.")

    @override
    def clear_data(self):
        print("Processing Data...")

        # 加载原始数据
        if (
            not hasattr(self, "sequence_data_raw")
            or not hasattr(self, "question_data_raw")
            or self.sequence_data_raw is None
            or self.question_data_raw is None
        ):
            self.load_src_data()

        # 处理题目信息
        question_data = self.question_data_raw.copy()  # 复制一份以防修改原数据
        # 重命名列
        question_data = question_data.rename(
            columns={
                "tags": "skill",
                "question_id": "question",
                "bundle_id": "assignment",
            }
        )
        question_data["question"] = question_data["question"].astype(str)
        # 移除缺失值
        question_data.dropna(subset=["correct_answer", "skill"], inplace=True)
        # 将bundle_id列转换为连续的整数ID
        bundles = question_data["assignment"].unique()
        bundles_id_map = {skill: idx for idx, skill in enumerate(bundles)}
        question_data["assignment"] = question_data["assignment"].map(bundles_id_map)
        # 将question_id列转换为连续的整数id
        questions = question_data["question"].unique()
        questions_id_map = {q: idx for idx, q in enumerate(questions)}
        question_data["question"] = question_data["question"].map(questions_id_map)

        # 处理用户回答数据
        sequence_data = self.sequence_data_raw.copy()
        # 将question_id列映射为整数id
        sequence_data["question_id"] = sequence_data["question_id"].map(
            questions_id_map
        )
        # 从question_data中构建question到correct_answer的映射
        q_ans_map = question_data.set_index("question")["correct_answer"].to_dict()
        # 过滤掉不在题目元数据中的题目
        valid_questions = sequence_data["question_id"].isin(q_ans_map.keys())
        sequence_data = sequence_data[valid_questions]
        # 按照question_id映射正确答案
        sequence_data["correct_answer"] = sequence_data["question_id"].map(q_ans_map)
        # 计算label
        sequence_data["label"] = (
            sequence_data["user_answer"] == sequence_data["correct_answer"]
        ).astype(int)
        # 抛弃不需要的列
        sequence_data = sequence_data[["user_id", "question_id", "label", "timestamp"]]
        # 重命名
        sequence_data.rename(
            columns={"question_id": "question", "user_id": "user"}, inplace=True
        )
        # 移除缺失值
        sequence_data.dropna(subset=["user", "question", "label"], inplace=True)
        # 排序
        sequence_data.sort_values(by=["user", "timestamp"], inplace=True)

        # 其他数据清理步骤
        # 限制序列长度
        sequence_data = restrains_sequence_length(
            sequence_data, self.args.min_seq_len, self.args.max_seq_len
        )
        # 映射ID
        sequence_data = map_to_continuous_ids(
            sequence_data, columns=["user", "question"]
        )
        self.sequence_data = sequence_data.copy()

        # 构建question_data
        self.question_data = build_question_data_from_cleared(
            question_data,
            skill_column="skill",
            question_column="question",
            seperator=";",
        )

        print(f"Processed {len(sequence_data)} interactions.")

        self.add_metadatas(
            {
                "num_users": sequence_data["user"].nunique(),
                "num_questions": self.question_data["question"].nunique(),
                "num_skills": self.question_data["skill"].nunique(),
                "num_assignments": self.question_data["assignment"].nunique(),
                "max_seq_len": self.args.max_seq_len,
                "min_seq_len": self.args.min_seq_len,
                "sequence_columns": sequence_data.columns.tolist(),
                "question_columns": self.question_data.columns.tolist(),
            }
        )


def process_chunk(file_paths, progress_queue=None):
    dfs = []
    processed_count = 0

    for file_path in file_paths:
        try:
            # 从文件名中提取user_id
            basename = os.path.basename(file_path)
            user_id_str = basename.split(".")[0]
            if user_id_str.startswith("u"):
                user_id = int(user_id_str[1:])
            else:
                continue

            # 读取CSV文件
            try:
                table = pv.read_csv(file_path)
                df = table.to_pandas()
            except Exception:
                continue

            if df.empty:
                continue

            df["user_id"] = user_id

            # 保留必要的列
            keep_cols = ["user_id", "question_id", "user_answer", "timestamp"]
            existing_cols = [c for c in keep_cols if c in df.columns]
            df = df[existing_cols]

            dfs.append(df)
        except Exception:
            pass
        finally:
            processed_count += 1
            # 每处理100个文件更新一次进度
            if progress_queue and processed_count >= 100:
                progress_queue.put(processed_count)
                processed_count = 0

    # 发送剩余的处理计数
    if progress_queue and processed_count > 0:
        progress_queue.put(processed_count)

    if not dfs:
        return None

    batch_df = pd.concat(dfs, ignore_index=True)

    return pa.Table.from_pandas(batch_df, preserve_index=False)
