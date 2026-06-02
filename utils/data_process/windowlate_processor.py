"""Windowlate 数据处理器"""

import os
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import tqdm

from utils.core import get_logger

logger = get_logger(__name__)


class WindowlateProcessor:
    """构建 windowlate 数据。

    - 每个目标 KC 仅生成一个可评估窗口（窗口末位是目标 KC）
    - 历史位置仅作为上下文，不参与评估（mask=0）
    - 目标位置参与评估（mask=1）
    - 当序列长度超过 max_seq_len 时，仅保留"以目标位置结尾"的最后一个窗口
    """

    # ===== 类常量：数据结构定义 =====
    DTYPE_MAP: dict[str, pl.DataType] = {
        "sample_id": pl.Int64,
        "position": pl.Int32,
        "skill": pl.Int32,
        "question": pl.Int32,
        "response": pl.Int8,
        "mask": pl.Int8,
        "user_id": pl.Int32,
        "group_id": pl.Int64,
        "true_label": pl.Int8,
        "fold": pl.Int32,
    }
    COLUMN_ORDER: list[str] = list(DTYPE_MAP.keys())
    SAMPLE_COLUMNS: list[str] = [col for col in COLUMN_ORDER if col != "fold"]
    CHUNK_ROW_LIMIT: int = 500_000

    # ===== 核心算法 =====

    @staticmethod
    def count_user_samples(skills_list: list[list[int]], max_seq_len: int) -> int:
        """计算单个用户的样本数量，无需实际生成行。

        Args:
            skills_list: 每个交互的技能列表
            max_seq_len: 最大序列长度

        Returns:
            该用户将生成的样本数量
        """
        _ = max_seq_len  # 接口保持兼容，计数与窗口长度无关
        return sum(len(q_skills) for q_skills in skills_list)

    @classmethod
    def generate_user_samples(
        cls,
        user_id: int,
        labels: list[int],
        skills_list: list[list[int]],
        questions: list[int],
        sample_id_start: int,
        group_id_start: int,
        max_seq_len: int,
    ) -> Iterator[list[tuple]]:
        """生成单个用户的所有样本数据。

        Args:
            user_id: 用户ID
            labels: 每个交互的正确性标签
            skills_list: 每个交互的技能列表
            questions: 每个交互的题目ID
            sample_id_start: 起始样本ID
            group_id_start: 起始组ID
            max_seq_len: 最大序列长度

        Yields:
            list[tuple]: 完整样本的所有行，每行格式为
                (sample_id, position, skill, response, mask, user_id, group_id, true_label)
        """
        # 展开技能和标签
        expanded_skills = []
        expanded_questions = []
        expanded_labels = []
        expanded_group_ids = []
        inter_boundaries = [0]

        for i, (q_skills, label) in enumerate(zip(skills_list, labels)):
            question_id = questions[i]
            for skill in q_skills:
                expanded_skills.append(skill)
                expanded_questions.append(question_id)
                expanded_labels.append(label)
                expanded_group_ids.append(group_id_start + i)
            inter_boundaries.append(inter_boundaries[-1] + len(q_skills))

        if not expanded_skills:
            return

        sample_id = sample_id_start
        num_interactions = len(skills_list)

        for inter_idx in range(num_interactions):
            n_skills = len(skills_list[inter_idx])
            history_end = inter_boundaries[inter_idx]
            for skill_offset in range(n_skills):
                current_skill_pos = inter_boundaries[inter_idx] + skill_offset
                current_skill = expanded_skills[current_skill_pos]
                current_question = expanded_questions[current_skill_pos]
                current_label = expanded_labels[current_skill_pos]
                current_group_id = expanded_group_ids[current_skill_pos]

                # 统一构建预测序列：历史 + 当前技能（目标位 response 置 0 防泄漏）
                full_skills = expanded_skills[:history_end] + [current_skill]
                full_questions = expanded_questions[:history_end] + [current_question]
                full_labels = expanded_labels[:history_end] + [0]
                full_group_ids = expanded_group_ids[:history_end] + [current_group_id]
                full_true_labels = expanded_labels[:history_end] + [current_label]

                # 仅保留"以目标位结尾"的窗口
                if len(full_skills) > max_seq_len:
                    win_skills = full_skills[-max_seq_len:]
                    win_questions = full_questions[-max_seq_len:]
                    win_labels = full_labels[-max_seq_len:]
                    win_group_ids = full_group_ids[-max_seq_len:]
                    win_true_labels = full_true_labels[-max_seq_len:]
                else:
                    win_skills = full_skills
                    win_questions = full_questions
                    win_labels = full_labels
                    win_group_ids = full_group_ids
                    win_true_labels = full_true_labels

                target_pos = len(win_skills) - 1
                rows = []
                for pos in range(len(win_skills)):
                    rows.append(
                        (
                            sample_id,
                            pos,
                            win_skills[pos],
                            win_questions[pos],
                            win_labels[pos],
                            1 if pos == target_pos else 0,
                            user_id,
                            win_group_ids[pos],
                            win_true_labels[pos],
                        )
                    )
                yield rows
                sample_id += 1

    # ===== 批量处理 =====

    @classmethod
    def process_user_batch(
        cls,
        args: tuple,
    ) -> tuple[int, str | None, int]:
        """处理一批用户，流式写入单个parquet文件。

        Args:
            args: (batch_idx, batch_users, max_seq_len, chunk_row_limit, output_dir)

        Returns:
            tuple: (batch_idx, output_path | None, total_rows)
        """
        batch_idx, batch_users, max_seq_len, chunk_row_limit, output_dir = args

        if not batch_users:
            return batch_idx, None, 0

        output_path = os.path.join(
            output_dir, f"windowlate_worker_{batch_idx:05d}.parquet"
        )
        writer = None
        total_rows = 0

        # 初始化缓冲区
        buffers = {col: [] for col in cls.DTYPE_MAP}

        try:
            for (
                user_id,
                labels,
                skills_list,
                questions,
                sample_id_start,
                group_id_start,
            ) in batch_users:
                for sample_rows in cls.generate_user_samples(
                    user_id,
                    labels,
                    skills_list,
                    questions,
                    sample_id_start,
                    group_id_start,
                    max_seq_len,
                ):
                    for row in sample_rows:
                        for i, col in enumerate(cls.SAMPLE_COLUMNS):
                            buffers[col].append(row[i])

                    if len(buffers["sample_id"]) >= chunk_row_limit:
                        writer = cls._flush_buffers(buffers, writer, output_path)
                        total_rows += len(buffers["sample_id"])
                        for col in buffers:
                            buffers[col].clear()

            # 最终刷新
            if buffers["sample_id"]:
                writer = cls._flush_buffers(buffers, writer, output_path)
                total_rows += len(buffers["sample_id"])

        finally:
            if writer is not None:
                writer.close()

        return batch_idx, (output_path if total_rows > 0 else None), total_rows

    @classmethod
    def _flush_buffers(
        cls,
        buffers: dict[str, list],
        writer: pq.ParquetWriter | None,
        output_path: str,
    ) -> pq.ParquetWriter:
        """将缓冲区数据写入parquet文件。"""
        chunk_df = pl.DataFrame(
            {
                "sample_id": np.asarray(buffers["sample_id"], dtype=np.int64),
                "position": np.asarray(buffers["position"], dtype=np.int32),
                "skill": np.asarray(buffers["skill"], dtype=np.int32),
                "question": np.asarray(buffers["question"], dtype=np.int32),
                "response": np.asarray(buffers["response"], dtype=np.int8),
                "mask": np.asarray(buffers["mask"], dtype=np.int8),
                "user_id": np.asarray(buffers["user_id"], dtype=np.int32),
                "group_id": np.asarray(buffers["group_id"], dtype=np.int64),
                "true_label": np.asarray(buffers["true_label"], dtype=np.int8),
                "fold": np.asarray([-1] * len(buffers["sample_id"]), dtype=np.int32),
            },
            schema=cls.DTYPE_MAP,
        )
        chunk_table = chunk_df.to_arrow()

        if writer is None:
            writer = pq.ParquetWriter(
                output_path, chunk_table.schema, compression="NONE"
            )
        writer.write_table(chunk_table)
        return writer

    # ===== 高层接口 =====

    @classmethod
    def build(
        cls,
        test_data: pl.DataFrame,
        question_data: pl.DataFrame,
        max_seq_len: int,
        output_path: str,
        num_workers: int = 0,
        users_per_batch: int = 64,
    ) -> pl.LazyFrame:
        """构建windowlate数据并直接写入文件。

        Args:
            test_data: 测试集序列数据
            question_data: 题目数据（包含技能映射）
            max_seq_len: 最大序列长度
            output_path: 输出文件路径（流式写入）
            num_workers: 并行worker数量（0或负数表示自动）
            users_per_batch: 每批处理的用户数
        """
        # 构建题目到技能列表的映射
        q_skill_map = (
            question_data.sort("question", "skill")
            .group_by("question")
            .agg(pl.col("skill").sort().alias("skills"))
        )

        # 将技能列表映射到测试数据
        test_data = test_data.join(q_skill_map, on="question", how="inner")
        sorted_test_data = test_data.sort(["user", "timestamp"])

        # 确定worker数量
        if num_workers <= 0:
            num_workers = max(1, os.cpu_count() or 1)

        # 将中间分块写到输出文件同级目录，避免落在系统临时目录。
        tmp_base_dir = os.path.dirname(os.path.abspath(output_path)) or "."
        os.makedirs(tmp_base_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix="windowlate_chunks_", dir=tmp_base_dir
        ) as tmp_dir:
            # 预处理：提取用户数据并计算偏移
            user_records, global_sample_id, global_group_id = cls._prepare_user_records(
                sorted_test_data, max_seq_len
            )

            if not user_records:
                raise ValueError("No valid windowlate evaluation samples for test set")

            # 构建批次输入
            batch_inputs = cls._build_batch_inputs(
                user_records, max_seq_len, users_per_batch, tmp_dir
            )

            logger.debug(
                f"Windowlate parallel workers={num_workers}, batches={len(batch_inputs)}, "
                f"users={len(user_records)}"
            )

            # 并行处理
            worker_results = cls._parallel_process(batch_inputs, num_workers)

            # 直接合并到最终输出路径
            total_rows = cls._merge_results(worker_results, output_path)

        logger.debug(
            f"Built windowlate data: {global_sample_id} samples, {total_rows} rows"
        )

    @classmethod
    def _prepare_user_records(
        cls,
        sorted_test_data: pl.DataFrame,
        max_seq_len: int,
    ) -> tuple[list, int, int]:
        """预处理用户数据并计算ID偏移。"""
        user_records = []
        global_sample_id = 0
        global_group_id = 0

        def _normalize_group_key(group_key):
            # Polars group_by iterator returns tuple keys even for single grouping col.
            if isinstance(group_key, tuple):
                return group_key[0]
            return group_key

        user_groups = sorted_test_data.group_by("user", maintain_order=True)
        for group_key, user_df in user_groups:
            user = _normalize_group_key(group_key)
            labels = user_df["label"].to_list()
            skills_list = user_df["skills"].to_list()
            questions = user_df["question"].to_list()

            sample_count = cls.count_user_samples(skills_list, max_seq_len)
            user_records.append(
                (
                    user,
                    labels,
                    skills_list,
                    questions,
                    global_sample_id,
                    global_group_id,
                )
            )
            global_sample_id += sample_count
            global_group_id += len(skills_list)

        return user_records, global_sample_id, global_group_id

    @classmethod
    def _build_batch_inputs(
        cls,
        user_records: list,
        max_seq_len: int,
        users_per_batch: int,
        tmp_dir: str,
    ) -> list:
        """构建批次输入参数。"""
        batch_inputs = []
        for idx in range(0, len(user_records), users_per_batch):
            batch_idx = len(batch_inputs)
            batch_users = user_records[idx : idx + users_per_batch]
            batch_inputs.append(
                (batch_idx, batch_users, max_seq_len, cls.CHUNK_ROW_LIMIT, tmp_dir)
            )
        return batch_inputs

    @classmethod
    def _parallel_process(
        cls,
        batch_inputs: list,
        num_workers: int,
    ) -> list:
        """并行处理批次。"""
        from concurrent.futures import as_completed

        worker_results = [None] * len(batch_inputs)

        if num_workers == 1 or len(batch_inputs) == 1:
            for res in tqdm.tqdm(
                map(cls.process_user_batch, batch_inputs),
                total=len(batch_inputs),
                desc="Processing windowlate batches",
            ):
                worker_results[res[0]] = res
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                futures = {
                    executor.submit(cls.process_user_batch, inp): inp[0]
                    for inp in batch_inputs
                }
                # 按完成顺序收集结果
                for future in tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing windowlate batches",
                ):
                    res = future.result()
                    worker_results[res[0]] = res

        return worker_results

    @classmethod
    def _merge_results(
        cls,
        worker_results: list,
        output_path: str,
    ) -> int:
        """合并所有worker结果到最终文件。"""
        tmp_path = output_path + ".tmp"
        final_writer = None
        total_rows = 0
        has_written_rows = False

        logger.info("Saving windowlate data to output path")

        try:
            for item in worker_results:
                if item is None:
                    continue
                _, worker_path, worker_rows = item
                if worker_path is None:
                    continue

                pq_file = pq.ParquetFile(worker_path)
                for rg_idx in range(pq_file.num_row_groups):
                    table = pq_file.read_row_group(rg_idx)
                    if final_writer is None:
                        final_writer = pq.ParquetWriter(tmp_path, table.schema)
                    final_writer.write_table(table)
                    has_written_rows = True
                total_rows += worker_rows
        finally:
            if final_writer is not None:
                final_writer.close()

        if not has_written_rows:
            raise ValueError("No valid windowlate evaluation samples generated")

        os.replace(tmp_path, output_path)
        return total_rows
