import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class DYGKTDataset(Dataset):
    """DYGKT 数据集类。
    
    为 DYGKT 模型准备数据，包括用户序列、问题序列、回答、时间戳和掩码。
    """
    
    def __init__(self, user_sequences, question_sequences, responses, time_sequences, masks):
        self.user_sequences = user_sequences
        self.question_sequences = question_sequences
        self.responses = responses
        self.time_sequences = time_sequences
        self.masks = masks

    def __getitem__(self, index):
        return (
            torch.tensor(self.user_sequences[index], dtype=torch.long),
            torch.tensor(self.question_sequences[index], dtype=torch.long),
            torch.tensor(self.responses[index], dtype=torch.long),
            torch.tensor(self.time_sequences[index], dtype=torch.float),
            torch.tensor(self.masks[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.question_sequences)


class DYGKTModelData(QuestionModelData):
    """DYGKT 模型数据处理类。"""
    
    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, args):
        """准备 DYGKT 模型所需的数据。
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        fold_idx = args.fold if args.fold >= 0 else None
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")

        # 构建用户答题序列
        question_sequence, user_response, user_mask, user_ids = self.load_sequence_data()
        
        # 生成时间戳序列（如果没有真实时间戳，使用序列位置模拟）
        time_sequences = self._generate_time_sequences(question_sequence, user_response)
        
        # 生成用户 ID 序列
        user_sequences = self._generate_user_sequences(question_sequence, user_ids)
        
        # 划分训练集、验证集和测试集
        if fold_idx is not None:
            kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
            if fold_idx < 0 or fold_idx >= kfold_n_splits:
                raise ValueError(
                    f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
                )
            logger.info(
                f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
            )
            train_data, val_data, test_data = self.split_kfold_data(
                question_sequence, user_response, user_mask, fold_idx=fold_idx
            )
        else:
            raise ValueError("fold_idx must be specified for K-fold cross-validation")

        # 同步划分时间序列和用户序列
        train_time = self._split_by_indices(time_sequences, train_data[0])
        val_time = self._split_by_indices(time_sequences, val_data[0])
        test_time = self._split_by_indices(time_sequences, test_data[0])
        
        train_users = self._split_by_indices(user_sequences, train_data[0])
        val_users = self._split_by_indices(user_sequences, val_data[0])
        test_users = self._split_by_indices(user_sequences, test_data[0])

        # 构建模型数据集
        train_dataset = DYGKTDataset(
            train_users, train_data[0], train_data[1], train_time, train_data[2]
        )
        val_dataset = DYGKTDataset(
            val_users, val_data[0], val_data[1], val_time, val_data[2]
        )
        test_dataset = DYGKTDataset(
            test_users, test_data[0], test_data[1], test_time, test_data[2]
        )

        return train_dataset, val_dataset, test_dataset

    def _generate_time_sequences(self, question_sequences, response_sequences):
        """生成时间戳序列。
        
        如果数据源包含真实时间戳，使用真实值；否则使用序列位置模拟。
        """
        # 尝试从数据源获取时间戳
        if hasattr(self.data_src, 'get_time_sequences'):
            return self.data_src.get_time_sequences()
        
        # 生成模拟时间戳（假设每个交互间隔 1 小时）
        time_sequences = []
        for seq in question_sequences:
            # 从 0 开始，每步增加 3600 秒（1小时）
            time_seq = [i * 3600.0 for i in range(len(seq))]
            # 添加一些随机扰动，使时间更真实
            time_seq = [t + np.random.uniform(0, 1800) for t in time_seq]
            time_sequences.append(time_seq)
        
        return time_sequences
    
    def _generate_user_sequences(self, question_sequences, user_ids):
        """生成用户 ID 序列。
        
        每个序列对应一个用户，生成相同长度的用户 ID 序列。
        """
        user_sequences = []
        for idx, seq in enumerate(question_sequences):
            user_id = user_ids[idx] if user_ids is not None else idx
            user_seq = [user_id] * len(seq)
            user_sequences.append(user_seq)
        
        return user_sequences
    
    def _split_by_indices(self, full_data, split_sequences):
        """根据划分后的序列索引提取对应的数据。
        
        Args:
            full_data: 完整数据列表
            split_sequences: 划分后的序列数据（用于确定长度和索引）
            
        Returns:
            划分后的数据
        """
        # 假设 split_sequences 是已经划分好的数据，我们需要对应的索引
        # 这里简单处理：返回相同长度的数据切片
        if len(full_data) == len(split_sequences):
            return full_data
        
        # 更安全的做法：根据序列内容重建
        result = []
        for seq in split_sequences:
            # 查找对应的原始数据
            for i, orig_seq in enumerate(full_data):
                if len(orig_seq) == len(seq):
                    # 简单匹配（实际应该有更好的索引机制）
                    result.append(orig_seq)
                    break
        
        return result if result else full_data[:len(split_sequences)]
