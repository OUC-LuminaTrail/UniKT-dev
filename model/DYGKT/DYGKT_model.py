from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.core import register_model


class TimeDualDecayEncoder(nn.Module):
    """时间双衰减编码器。
    
    使用短期和长期两种时间衰减机制，对时间间隔进行编码。
    - 短期衰减：处理 24 小时内的时间间隔
    - 长期衰减：处理超过 24 小时的时间间隔
    
    Args:
        dim_time: 时间编码维度
        parameter_requires_grad: 是否允许参数梯度更新
        
    Example:
        >>> encoder = TimeDualDecayEncoder(dim_time=64)
        >>> timestamps = torch.randn(32, 100)  # [B, S]
        >>> time_encoding = encoder(timestamps)  # [B, S, 64]
    """
    
    def __init__(self, dim_time: int, parameter_requires_grad: bool = True) -> None:
        super().__init__()
        self.time_dim = dim_time
        
        # 短期时间衰减权重
        self.w_short = nn.Linear(1, dim_time)
        self.w_short.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time, dtype=np.float32)).reshape(dim_time, -1)
        )
        self.w_short.bias = nn.Parameter(torch.zeros(dim_time))
        
        # 长期时间衰减权重
        self.w_long = nn.Linear(1, dim_time)
        self.w_long.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time, dtype=np.float32)).reshape(dim_time, -1)
        )
        self.w_long.bias = nn.Parameter(torch.zeros(dim_time))
        
        # 输出投影层
        self.w_o = nn.Linear(dim_time, dim_time)
        self.w_o.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time * dim_time, dtype=np.float32)).reshape(dim_time, -1)
        )
        self.w_o.bias = nn.Parameter(torch.zeros(dim_time))
        
        self.f = nn.ReLU()
        
        # 控制参数是否可训练
        if not parameter_requires_grad:
            self.w_short.weight.requires_grad = False
            self.w_short.bias.requires_grad = False
            self.w_long.weight.requires_grad = False
            self.w_long.bias.requires_grad = False
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            timestamps: 时间戳序列 [B, S]
            
        Returns:
            时间编码 [B, S, dim_time]
        """
        timestamps = timestamps.unsqueeze(dim=2)  # [B, S, 1]
        
        # 计算相邻时间步的时间差
        timestamps_right = timestamps.clone()
        timestamps_right = torch.cat(
            [timestamps_right[:, 1:, :], timestamps_right[:, -1, :].unsqueeze(1)],
            dim=1
        )  # [B, S, 1]
        timestamps_diff = timestamps_right - timestamps  # [B, S, 1]
        
        # 区分短期（24小时内）和长期（超过24小时）
        timestamps_mask = (timestamps_diff > 3600 * 24).float()  # [B, S, 1]
        
        # 短期和长期衰减编码
        timestamps_short = self.f(self.w_short(timestamps_diff * timestamps_mask))
        timestamps_long = self.f(self.w_long(timestamps_diff * (1 - timestamps_mask)))
        
        # 融合输出
        output = self.w_o(timestamps_short + timestamps_long)  # [B, S, dim_time]
        
        return output


class DyKT_Seq(nn.Module):
    """动态知识追踪序列更新模块。
    
    使用 GRU 对历史交互序列进行编码，更新节点（用户/问题）的动态表示。
    
    Args:
        edge_dim: 边特征维度（交互特征维度）
        node_dim: 节点隐藏状态维度
        
    Example:
        >>> updater = DyKT_Seq(edge_dim=128, node_dim=64)
        >>> edge_features = torch.randn(1, 10, 128)  # [1, seq_len, edge_dim]
        >>> node_states = updater.update(edge_features)  # [seq_len, node_dim]
    """
    
    def __init__(self, edge_dim: int, node_dim: int) -> None:
        super().__init__()
        self.patch_enc_layer = nn.Linear(edge_dim, node_dim)
        self.hid_node_updater = nn.GRU(
            input_size=edge_dim,
            hidden_size=node_dim,
            batch_first=True
        )
    
    def update(self, x: torch.Tensor) -> torch.Tensor:
        """更新节点状态。
        
        Args:
            x: 边特征序列 [1, seq_len, edge_dim]
            
        Returns:
            更新后的节点状态 [seq_len, node_dim]
        """
        outputs, _ = self.hid_node_updater(x)
        return torch.squeeze(outputs, dim=0)


@register_model("DYGKT")
class DYGKT(nn.Module):
    """动态图知识追踪模型 (Dynamic Graph-based Knowledge Tracing)。
    
    基于动态图的知识追踪模型，通过建模用户和问题的历史交互序列，
    捕捉用户知识状态和问题难度的动态变化。
    
    核心特点：
    - 双时间衰减机制：区分短期和长期记忆
    - 用户-问题双向建模：同时追踪用户状态和问题特征的演化
    - GRU 序列更新：动态更新节点表示
    
    Args:
        args: 模型参数配置
        data_metadata: 数据集元数据，包含 num_questions, num_users 等
        **kwargs: 额外的关键字参数
        
    Example:
        >>> model = DYGKT(args, data_metadata)
        >>> logits = model(user_seq, question_seq, response_seq, time_seq, mask)
    """
    
    def __init__(self, args: Any, data_metadata: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.args = args
        self.data_metadata = data_metadata
        
        # 模型参数
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.dim_time = getattr(args, 'dim_time', 64)
        self.dropout = args.dropout
        
        num_questions = data_metadata["num_questions"]
        num_users = data_metadata.get("num_users", 1000)  # 默认值
        
        # Embedding 层
        self.question_embedding = nn.Embedding(
            num_embeddings=num_questions,
            embedding_dim=self.embedding_dim
        )
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=self.embedding_dim
        )
        self.answer_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.embedding_dim
        )
        
        # 表现编码器（将正确率编码为特征）
        self.performance_encoder = nn.Linear(1, 64)
        
        # 双时间衰减编码器
        self.dual_time_encoder = TimeDualDecayEncoder(self.dim_time)
        
        # 多集指示器
        self.multiset_indicator = nn.Linear(1, 64)
        
        # 用户和问题的 GRU 更新器
        self.gru_linear4user = nn.Linear(self.embedding_dim, 64)
        self.gru4user = nn.GRU(self.embedding_dim, 64, batch_first=True)
        
        self.gru_linear4que = nn.Linear(self.embedding_dim, 64)
        self.gru4que = nn.GRU(self.embedding_dim, 64, batch_first=True)
        
        # 预测层
        self.fc_hidden = nn.Linear(64 + 64 + self.dim_time, self.hidden_dim)
        self.fc_output = nn.Linear(self.hidden_dim, 1)
        
        self.embedding_dropout = nn.Dropout(p=self.dropout)
    
    def get_user_que_embedding(
        self,
        user_seq: torch.Tensor,
        question_seq: torch.Tensor,
        response_seq: torch.Tensor,
        time_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取用户和问题的动态嵌入表示。
        
        Args:
            user_seq: 用户 ID 序列 [B, S] 或 [B, S, ...]
            question_seq: 问题 ID 序列 [B, S]
            response_seq: 回答正确性序列 [B, S]
            time_seq: 时间戳序列 [B, S]
            
        Returns:
            user_emb: 用户嵌入 [B, S, 64]
            que_emb: 问题嵌入 [B, S, 64]
            time_emb: 时间编码 [B, S, dim_time]
        """
        # 智能处理维度：保留前两个维度 [B, S]
        def normalize_dim(tensor):
            """将输入张量规范化为 [B, S] 形状"""
            if tensor.dim() == 1:
                # [S] -> [1, S]
                return tensor.unsqueeze(0)
            elif tensor.dim() == 2:
                # [B, S] - 已经正确
                return tensor
            elif tensor.dim() == 3:
                # [1, B, S] or [B, S, 1] -> [B, S]
                # 检查哪个维度是1
                if tensor.size(0) == 1:
                    return tensor.squeeze(0)  # [1, B, S] -> [B, S]
                elif tensor.size(2) == 1:
                    return tensor.squeeze(2)  # [B, S, 1] -> [B, S]
                else:
                    # 假设是 [B, S, ?]，取前两维
                    return tensor[:, :, 0]
            else:
                # 对于更高维度，尽量保留前两维
                while tensor.dim() > 2:
                    if tensor.size(-1) == 1:
                        tensor = tensor.squeeze(-1)
                    elif tensor.size(0) == 1:
                        tensor = tensor.squeeze(0)
                    else:
                        break
                return tensor
        
        user_seq = normalize_dim(user_seq)
        question_seq = normalize_dim(question_seq)
        response_seq = normalize_dim(response_seq)
        time_seq = normalize_dim(time_seq)
            
        B, S = user_seq.size()
        
        # 获取基础嵌入
        user_base_emb = self.user_embedding(user_seq)  # [B, S, E]
        que_base_emb = self.question_embedding(question_seq)  # [B, S, E]
        ans_emb = self.answer_embedding(response_seq)  # [B, S, E]
        
        # 组合练习嵌入（问题 + 回答）
        exercise_emb = que_base_emb + ans_emb  # [B, S, E]
        exercise_emb = self.embedding_dropout(exercise_emb)
        
        # GRU 更新用户状态
        user_gru_out, _ = self.gru4user(exercise_emb)  # [B, S, 64]
        
        # GRU 更新问题状态
        que_gru_out, _ = self.gru4que(exercise_emb)  # [B, S, 64]
        
        # 时间编码
        time_emb = self.dual_time_encoder(time_seq)  # [B, S, dim_time]
        
        return user_gru_out, que_gru_out, time_emb
    
    def forward(
        self,
        user_sequence: torch.Tensor,
        user_response: torch.Tensor,
        user_mask: torch.Tensor,
        question_sequence: torch.Tensor,
        time_sequence: torch.Tensor,
        return_states: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。
        
        Args:
            user_sequence: 用户 ID 序列 [B, S]（注：在 KT 中通常一个 batch 内同一用户）
            user_response: 用户回答序列 [B, S]
            user_mask: 有效位置掩码 [B, S]
            question_sequence: 问题 ID 序列 [B, S]
            time_sequence: 时间戳序列 [B, S]
            return_states: 是否额外返回中间状态
            
        Returns:
            若 return_states=False: 预测 logits [B, S]
            若 return_states=True: (logits, user_emb, que_emb)
        """
        # 获取用户、问题和时间的嵌入（内部会处理维度）
        user_emb, que_emb, time_emb = self.get_user_que_embedding(
            user_sequence, question_sequence, user_response, time_sequence
        )  # [B, S, 64], [B, S, 64], [B, S, dim_time]
        
        # 从嵌入结果获取批次大小和序列长度
        B, S = user_emb.size()[:2]
        
        # 拼接所有特征
        combined_features = torch.cat([user_emb, que_emb, time_emb], dim=-1)  # [B, S, 64+64+dim_time]
        
        # 通过全连接层
        hidden = F.relu(self.fc_hidden(combined_features))  # [B, S, hidden_dim]
        hidden = self.embedding_dropout(hidden)
        
        # 预测层
        logits = self.fc_output(hidden).squeeze(-1)  # [B, S]
        
        if return_states:
            return logits, user_emb, que_emb
        
        return logits
