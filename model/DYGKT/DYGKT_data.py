"""
DYGKT 数据处理模块（严格按照原始 pyedmine 实现）

关键改动：
1. 维护全局交互索引 n
2. 实现用户和问题的历史邻居查找
3. 使用 Q-table 计算问题相似度
4. 用户ID重新编号：user_id = num_question + original_user_id
"""

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
    
    def __init__(self, dataset_config, data_all, q_table, device='cpu'):
        """
        Args:
            dataset_config: 配置字典，包含 num_question, num_neighbor
            data_all: 用户序列数据列表
            q_table: 问题-知识点矩阵 [num_question, num_concept]
            device: 设备
        """
        super(DYGKTDataset, self).__init__()
        self.dataset_config = dataset_config
        self.data_all = data_all
        self.q_table = q_table
        self.device = device
        
        # 原始格式数据（与 pyedmine 完全一致）
        self.dataset_converted = {
            "idx": [],
            "user": [],
            "question": [],
            "idx_in_seq": [],
            "time": [],
            "correctness": [],
            "user_his_seq": [],
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
                neighbor_idx = torch.tensor(key_data + padding).long().to(self.device)
                neighbor_time = self.dataset["time"][neighbor_idx]
                neighbor_edge = self.dataset["correctness"][neighbor_idx]
                neighbor_last_idx = torch.tensor(len(key_data)).long().to(self.device)
                
                if key == "user_his_seq":
                    result["user_his_time_seq"] = neighbor_time
                    result["user_his_correctness_seq"] = neighbor_edge
                    result["user_his_last_idx"] = neighbor_last_idx
                else:
                    result["que_his_time_seq"] = neighbor_time
                    result["que_his_correctness_seq"] = neighbor_edge
                    result["que_his_last_idx"] = neighbor_last_idx
                    
            elif key in ["user_his_snd_seq", "user_his_snk_seq", "que_his_qn_seq"]:
                key_data = self.dataset_converted[key][index]
                padding = [0] * (num_neighbor - len(key_data))
                result[key] = torch.tensor(key_data + padding).long().to(self.device)
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
                self.dataset_converted["user_his_snd_seq"].append(user_his_snd_seq)
                
                # 相似知识点指示器（严格复刻 L105）
                user_his_snk_seq = list(map(lambda q: int(que_sim_by_concept[q, q_id]), question_seq_))
                self.dataset_converted["user_his_snk_seq"].append(user_his_snk_seq)
                
                # 问题历史序列（第二遍填充）
                self.dataset_converted["que_his_seq"].append(None)
                self.dataset_converted["que_his_qn_seq"].append(None)
                
                n += 1
        
        # 第二遍遍历：填充问题历史序列（严格复刻 L112-123）
        for i in range(n):
            q_id = self.dataset_converted["question"][i]
            t = self.dataset_converted["time"][i]
            
            que_his_seq = list(map(
                lambda x: x[0],
                sorted(
                    list(filter(
                        lambda y: y[1] < t,
                        que_his_seqs[q_id]
                    )),
                    key=lambda z: z[1]
                )
            ))
            
            self.dataset_converted["que_his_seq"][i] = \
                que_his_seq if len(que_his_seq) < num_neighbor else que_his_seq[-num_neighbor:]
            
            # que_his_qn_seq 暂时设为空列表（原始实现中未使用）
            self.dataset_converted["que_his_qn_seq"][i] = []
    
    def dataset2tensor(self):
        """转换为 Tensor（严格复刻 L126-130）。"""
        self.dataset = {}
        for k in self.dataset_converted.keys():
            if k not in ["user_his_seq", "que_his_seq", "user_his_snd_seq", "user_his_snk_seq", "que_his_qn_seq"]:
                self.dataset[k] = torch.tensor(self.dataset_converted[k]).long().to(self.device)


class DYGKTModelData(QuestionModelData):
    """DYGKT 模型数据处理类（适配 kt-exp-graph 框架）。"""
    
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        """准备 DYGKT 数据。"""
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        num_questions = self.data_src.get_metadata("num_questions")
        num_neighbor = getattr(args, 'num_neighbor', 50)
        device = getattr(args, 'device', 'cpu')
        
        # 获取 Q-table
        q_table = self.data_src.get_q_table()
        if q_table is None:
            raise ValueError("DYGKT requires Q-table. Please ensure data source provides get_q_table()")
        
        dataset_config = {
            "num_question": num_questions,
            "num_neighbor": num_neighbor,
            "device": device
        }
        
        # 加载用户序列数据
        data_all = self._load_user_sequences()
        
        # K-fold 划分
        if fold_idx is not None:
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(f"fold_idx {fold_idx} out of range [0, {kfold_n_splits})")
            
            logger.info(f"K-fold: fold {fold_idx + 1}/{kfold_n_splits}")
            
            train_users, val_users, test_users = self._split_users_kfold(
                len(data_all), fold_idx, kfold_n_splits
            )
            
            train_data = [data_all[i] for i in train_users]
            val_data = [data_all[i] for i in val_users]
            test_data = [data_all[i] for i in test_users]
        else:
            raise ValueError("fold_idx must be specified")

        # 构建数据集
        train_dataset = DYGKTDataset(dataset_config, train_data, q_table, device)
        val_dataset = DYGKTDataset(dataset_config, val_data, q_table, device)
        test_dataset = DYGKTDataset(dataset_config, test_data, q_table, device)

        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset
    
    def _load_user_sequences(self):
        """加载用户序列数据（转换为原始格式）。"""
        question_sequences, user_responses, user_masks, user_ids = self.load_sequence_data()
        
        # 获取或生成时间序列
        if hasattr(self.data_src, 'get_time_sequences'):
            time_sequences = self.data_src.get_time_sequences()
        else:
            logger.warning("No timestamps, generating simulated ones")
            time_sequences = []
            for seq in question_sequences:
                time_seq = np.arange(len(seq), dtype=np.float32) * 3600.0
                time_seq += np.random.uniform(0, 1800, size=len(seq))
                time_sequences.append(time_seq.tolist())
        
        # 转换为原始格式
        data_all = []
        for i, (q_seq, r_seq, mask, time_seq) in enumerate(zip(
            question_sequences, user_responses, user_masks, time_sequences
        )):
            user_id = user_ids[i] if user_ids is not None else i
            seq_len = int(mask.sum()) if hasattr(mask, 'sum') else len(q_seq)
            
            data_all.append({
                "user_id": user_id,
                "seq_len": seq_len,
                "question_seq": q_seq[:seq_len] if isinstance(q_seq, list) else q_seq[:seq_len].tolist(),
                "correctness_seq": r_seq[:seq_len] if isinstance(r_seq, list) else r_seq[:seq_len].tolist(),
                "time_seq": time_seq[:seq_len] if isinstance(time_seq, list) else time_seq[:seq_len].tolist()
            })
        
        return data_all
    
    def _split_users_kfold(self, num_users, fold_idx, kfold_n_splits):
        """K-fold 用户划分。"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=kfold_n_splits, shuffle=True, random_state=42)
        user_indices = list(range(num_users))
        
        for i, (train_val_idx, test_idx) in enumerate(kf.split(user_indices)):
            if i == fold_idx:
                val_size = max(1, len(train_val_idx) // 10)
                train_idx = train_val_idx[:-val_size]
                val_idx = train_val_idx[-val_size:]
                return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
        
        raise ValueError(f"fold_idx {fold_idx} not found")
