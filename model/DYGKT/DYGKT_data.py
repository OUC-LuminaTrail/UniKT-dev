"""
DYGKT 数据处理模块（严格按照原始 pyedmine 实现）

关键改动：
1. 维护全局交互索引 n
2. 实现用户和问题的历史邻居查找
3. 使用 Q-table 计算问题相似度
4. 用户ID重新编号：user_id = num_question + original_user_id
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class DYGKTDataset(Dataset):
    """DYGKT 数据集类（严格复刻原始实现）。"""
    
    def __init__(
        self,
        dataset_config: dict[str, Any],
        data_all: list[dict[str, Any]],
        q_table: np.ndarray,
    ):
        """
        Args:
            dataset_config: 配置字典，包含 num_question, num_neighbor
            data_all: 用户序列数据列表
            q_table: 问题-知识点矩阵 [num_question, num_concept]
            device: 设备
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.data_all = data_all
        self.q_table = q_table
        
        # 原始格式数据（与 pyedmine 完全一致）
        self.dataset_converted = {
            "idx": [],
            "user": [],
            "question": [],
            "idx_in_seq": [],
            "time": [],
            "correctness": [],
            "user_his_seq": [],
            # pyedmine 原始代码初始化为 snq，但后续写入使用 snd。
            # 这里同时保留两个键，确保兼容严格复现和当前框架测试。
            "user_his_snq_seq": [],
            "user_his_snd_seq": [],
            "user_his_snk_seq": [],
            "que_his_seq": [],
            "que_his_qn_seq": [],
        }
        
        self.dataset = None
        self.process_dataset()
        
    def __len__(self):
        return self.dataset["user"].shape[0]
        
    def __getitem__(self, index):
        """返回一个交互的完整信息（严格复刻原始实现）。"""
        result = dict()
        num_neighbor = self.dataset_config["num_neighbor"]
        
        for key in self.dataset_converted.keys():
            if key in ["user_his_seq", "que_his_seq"]:
                # 历史序列：padding + 索引对应的时间和正确率
                key_data = self.dataset_converted[key][index]
                padding = [0] * (num_neighbor - len(key_data))
                neighbor_idx = torch.tensor(key_data + padding, dtype=torch.long)
                neighbor_time = self.dataset["time"][neighbor_idx]
                neighbor_edge = self.dataset["correctness"][neighbor_idx]
                neighbor_last_idx = torch.tensor(len(key_data), dtype=torch.long)
                
                if key == "user_his_seq":
                    result["user_his_time_seq"] = neighbor_time
                    result["user_his_correctness_seq"] = neighbor_edge
                    result["user_his_last_idx"] = neighbor_last_idx
                else:
                    result["que_his_time_seq"] = neighbor_time
                    result["que_his_correctness_seq"] = neighbor_edge
                    result["que_his_last_idx"] = neighbor_last_idx
                    
            elif key in [
                "user_his_snq_seq",
                "user_his_snd_seq",
                "user_his_snk_seq",
                "que_his_qn_seq",
            ]:
                key_data = self.dataset_converted[key][index]
                padding = [0] * (num_neighbor - len(key_data))
                result[key] = torch.tensor(key_data + padding, dtype=torch.long)
            else:
                result[key] = self.dataset[key][index]
                
        return result
        
    def process_dataset(self):
        """处理数据集。"""
        self.convert_dataset()
        self.dataset2tensor()
        
    def convert_dataset(self):
        """核心数据转换逻辑（严格复刻 pyedmine L78-123）。"""
        # 计算问题相似度矩阵（基于共享知识点）
        que_sim_by_concept = ((self.q_table @ self.q_table.T) > 0).astype(int)
        
        num_question = self.dataset_config["num_question"]
        num_neighbor = self.dataset_config["num_neighbor"]

        logger.info("DYGKT: building interaction records and user histories...")
        
        # 全局交互计数器（关键！）
        n = 0
        que_his_seqs = {}
        
        # 第一遍遍历：构建用户历史
        for user_data in self.data_all:
            # 用户ID重新编号（关键！）
            user_id = num_question + user_data["user_id"]
            seq_len = user_data["seq_len"]
            question_seq = user_data["question_seq"][:seq_len]
            correctness_seq = user_data["correctness_seq"][:seq_len]
            time_seq = user_data["time_seq"][:seq_len]
            
            for i, (q_id, t, c) in enumerate(zip(question_seq, time_seq, correctness_seq)):
                # 记录问题历史
                if q_id not in que_his_seqs:
                    que_his_seqs[q_id] = []
                que_his_seqs[q_id].append((n, t))
                
                # 基础信息
                self.dataset_converted["idx"].append(n)
                self.dataset_converted["user"].append(user_id)
                self.dataset_converted["question"].append(q_id)
                self.dataset_converted["idx_in_seq"].append(i)
                self.dataset_converted["time"].append(t)
                self.dataset_converted["correctness"].append(c)
                
                # 用户历史序列（严格复刻 L101）
                user_his_seq = list(range(n-i, n)) if i < num_neighbor else list(range(n-num_neighbor, n))
                self.dataset_converted["user_his_seq"].append(user_his_seq)
                
                # 用户历史中的问题序列（严格复刻 L103）
                question_seq_ = question_seq[0 if (i <= num_neighbor) else (i-num_neighbor):i]
                
                # 相同问题指示器（严格复刻 L104）
                user_his_snd_seq = list(map(lambda q: int(q == q_id), question_seq_))
                self.dataset_converted["user_his_snq_seq"].append(user_his_snd_seq)
                self.dataset_converted["user_his_snd_seq"].append(user_his_snd_seq)
                
                # 相似知识点指示器（严格复刻 L105）
                user_his_snk_seq = list(map(lambda q: int(que_sim_by_concept[q, q_id]), question_seq_))
                self.dataset_converted["user_his_snk_seq"].append(user_his_snk_seq)
                
                # 问题历史序列（第二遍填充）
                self.dataset_converted["que_his_seq"].append(None)
                self.dataset_converted["que_his_qn_seq"].append(None)
                
                n += 1

        logger.info("DYGKT: building question histories (optimized, strict time ordering)...")
        # 第二遍遍历：填充问题历史序列
        # 原始实现为：对每个交互 i，筛选同题中满足 time < t 的全部历史并排序。
        # 这里按题目分组并按时间增量维护历史，语义等价，复杂度从 O(N^2) 显著下降。
        for q_id, events in que_his_seqs.items():
            # events: list[(global_idx, timestamp)]
            # 先按时间排序；同时间按索引排序保证稳定性
            sorted_events = sorted(events, key=lambda x: (x[1], x[0]))

            # 只保留“严格更早时间”的历史，因此同一时间组内不互相可见
            history_indices: list[int] = []
            pos = 0
            total = len(sorted_events)
            while pos < total:
                current_t = sorted_events[pos][1]
                group_end = pos
                while group_end < total and sorted_events[group_end][1] == current_t:
                    group_end += 1

                # 当前时间组的每个交互都共享同一批“更早时间”历史
                recent_history = (
                    history_indices
                    if len(history_indices) < num_neighbor
                    else history_indices[-num_neighbor:]
                )
                for k in range(pos, group_end):
                    idx_k = sorted_events[k][0]
                    self.dataset_converted["que_his_seq"][idx_k] = list(recent_history)
                    self.dataset_converted["que_his_qn_seq"][idx_k] = []

                # 将当前时间组加入历史，供后续更晚时间使用
                history_indices.extend(sorted_events[k][0] for k in range(pos, group_end))
                pos = group_end

        # 兜底：若某些交互未被填充（理论上不会发生），置为空序列
        for i in range(n):
            if self.dataset_converted["que_his_seq"][i] is None:
                self.dataset_converted["que_his_seq"][i] = []
            if self.dataset_converted["que_his_qn_seq"][i] is None:
                self.dataset_converted["que_his_qn_seq"][i] = []

        logger.info("DYGKT: question history construction finished.")
    
    def dataset2tensor(self):
        """转换为 Tensor（严格复刻 L126-130）。"""
        self.dataset = {}
        for k in self.dataset_converted.keys():
            # 这些键长度可变，保留在 dataset_converted 里按样本动态 padding。
            if k not in [
                "user_his_seq",
                "que_his_seq",
                "user_his_snq_seq",
                "user_his_snd_seq",
                "user_his_snk_seq",
                "que_his_qn_seq",
            ]:
                self.dataset[k] = torch.tensor(self.dataset_converted[k], dtype=torch.long)


class DYGKTModelData(QuestionModelData):
    """DYGKT 模型数据处理类（适配 kt-exp-graph 框架）。"""
    
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        """准备 DYGKT 数据。"""
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        num_neighbor = getattr(args, 'num_neighbor', 50)
        
        # 使用框架底层关系矩阵机制构建 Q-table（question-skill）
        q_table = self.build_relationship_matrix(("question", "has", "skill"))
        if q_table is None or q_table.size == 0:
            raise ValueError("DYGKT requires a non-empty Q-table.")

        num_questions = int(q_table.shape[0])
        
        dataset_config = {
            "num_question": num_questions,
            "num_neighbor": num_neighbor,
        }
        
        # 加载序列数据并按框架 K-fold 标签划分
        question_sequences, user_responses, user_masks, user_id_sequences = (
            self.load_sequence_data()
        )
        time_sequences = self._load_time_sequences(question_sequences.shape)
        
        # K-fold 划分
        if fold_idx is not None:
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(f"fold_idx {fold_idx} out of range [0, {kfold_n_splits})")
            
            logger.info(f"K-fold: fold {fold_idx + 1}/{kfold_n_splits}")
            train_data, val_data, test_data = self.split_kfold_data(
                question_sequences,
                user_responses,
                user_masks,
                time_sequences,
                user_id_sequences,
                fold_idx=fold_idx,
            )
        else:
            raise ValueError("fold_idx must be specified")

        train_records = self._build_interaction_records(*train_data)
        val_records = self._build_interaction_records(*val_data)
        test_records = self._build_interaction_records(*test_data)

        # 构建数据集
        train_dataset = DYGKTDataset(dataset_config, train_records, q_table)
        val_dataset = DYGKTDataset(dataset_config, val_records, q_table)
        test_dataset = DYGKTDataset(dataset_config, test_records, q_table)

        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset
    
    def _load_time_sequences(self, target_shape: tuple[int, int]) -> np.ndarray:
        """从 split_question_sequence 加载时间序列。"""
        num_users, max_seq_len = target_shape
        timestamps = np.zeros((num_users, max_seq_len), dtype=np.float32)

        split_data = self.data_src.get_split_question_sequence_data().to_pandas()
        if "timestamp" not in split_data.columns:
            logger.warning("No timestamp column found, using synthetic timestamps.")
            for u in range(num_users):
                timestamps[u, :] = np.arange(max_seq_len, dtype=np.float32) * 3600.0
            return timestamps

        users = split_data["user"].to_numpy(dtype=np.int64)
        seq_pos = split_data["seq_pos"].to_numpy(dtype=np.int64)
        ts = split_data["timestamp"].to_numpy(dtype=np.float32)

        valid = (
            (users >= 0)
            & (users < num_users)
            & (seq_pos >= 0)
            & (seq_pos < max_seq_len)
        )
        timestamps[users[valid], seq_pos[valid]] = ts[valid]
        return timestamps

    def _build_interaction_records(
        self,
        question_sequences: np.ndarray,
        user_responses: np.ndarray,
        user_masks: np.ndarray,
        time_sequences: np.ndarray,
        user_id_sequences: np.ndarray,
    ) -> list[dict[str, Any]]:
        """将框架序列数据转换为 pyedmine DyGKT 所需记录格式。"""
        records: list[dict[str, Any]] = []

        for idx, (q_seq, r_seq, mask_seq, t_seq, uid_seq) in enumerate(
            zip(
                question_sequences,
                user_responses,
                user_masks,
                time_sequences,
                user_id_sequences,
            )
        ):
            seq_len = int(np.asarray(mask_seq).sum())
            if seq_len <= 0:
                continue

            valid_uid = np.asarray(uid_seq)[:seq_len]
            if valid_uid.size > 0:
                user_id = int(valid_uid[0])
            else:
                user_id = idx

            records.append(
                {
                    "user_id": user_id,
                    "seq_len": seq_len,
                    "question_seq": np.asarray(q_seq)[:seq_len].astype(np.int64).tolist(),
                    "correctness_seq": np.asarray(r_seq)[:seq_len]
                    .astype(np.int64)
                    .tolist(),
                    "time_seq": np.asarray(t_seq)[:seq_len].astype(np.float32).tolist(),
                }
            )

        return records
